// src/core/polygon_reader.cpp
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <curl/curl.h>
#include <zstd.h>
#include <chrono>
#include <vector>
#include <memory>
#include <unordered_map>
#include "polygon_reader.h"
#include "data_structures.h"

// Nanosecond timestamp handling
using NanoTime = std::chrono::nanoseconds;
using TimePoint = std::chrono::time_point<std::chrono::system_clock, NanoTime>;

struct Trade {
    TimePoint timestamp;
    double price;
    uint64_t volume;
    uint8_t conditions[4];
    char exchange;
};

struct Quote {
    TimePoint timestamp;
    double bid_price;
    double ask_price;
    uint64_t bid_size;
    uint64_t ask_size;
    char bid_exchange;
    char ask_exchange;
};

class PolygonDataReader {
private:
    std::string api_key;
    CURL* curl;
    
    // Write callback for CURL
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    
    // Download and decompress data
    std::vector<uint8_t> downloadAndDecompress(const std::string& url) {
        std::string compressed_data;
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &compressed_data);
        
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            throw std::runtime_error("CURL error: " + std::string(curl_easy_strerror(res)));
        }
        
        // Decompress using zstd
        size_t decompressed_size = ZSTD_getFrameContentSize(
            compressed_data.data(), compressed_data.size()
        );
        
        std::vector<uint8_t> decompressed_data(decompressed_size);
        
        size_t actual_size = ZSTD_decompress(
            decompressed_data.data(), decompressed_size,
            compressed_data.data(), compressed_data.size()
        );
        
        if (ZSTD_isError(actual_size)) {
            throw std::runtime_error("ZSTD decompression error");
        }
        
        decompressed_data.resize(actual_size);
        return decompressed_data;
    }
    
public:
    PolygonDataReader(const std::string& key) : api_key(key) {
        curl = curl_easy_init();
        if (!curl) {
            throw std::runtime_error("Failed to initialize CURL");
        }
    }
    
    ~PolygonDataReader() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }
    
    // Read trades data from Polygon flat files
    std::vector<Trade> readTrades(
        const std::string& symbol,
        const std::string& date
    ) {
        // Construct URL for trades flat file
        std::string url = "https://files.polygon.io/flatfiles/us_stocks_sip/trades_v1/" 
                         + date + "/" + symbol + ".csv.zst";
        
        auto data = downloadAndDecompress(url);
        
        // Parse CSV data with nanosecond precision
        std::vector<Trade> trades;
        trades.reserve(1000000);  // Pre-allocate for performance
        
        // Create Arrow CSV reader for efficient parsing
        arrow::io::IOContext io_context;
        auto input = std::make_shared<arrow::io::BufferReader>(data);
        
        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.column_names = {
            "participant_timestamp", "price", "size", 
            "conditions", "exchange"
        };
        
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        parse_options.delimiter = ',';
        
        auto convert_options = arrow::csv::ConvertOptions::Defaults();
        
        auto maybe_reader = arrow::csv::StreamingReader::Make(
            io_context, input, read_options, parse_options, convert_options
        );
        
        if (!maybe_reader.ok()) {
            throw std::runtime_error("Failed to create CSV reader");
        }
        
        auto reader = *maybe_reader;
        
        while (true) {
            auto maybe_batch = reader->Next();
            if (!maybe_batch.ok()) break;
            
            auto batch = *maybe_batch;
            if (batch == nullptr) break;
            
            auto timestamp_array = batch->column(0);
            auto price_array = batch->column(1);
            auto size_array = batch->column(2);
            
            for (int64_t i = 0; i < batch->num_rows(); ++i) {
                Trade trade;
                
                // Parse nanosecond timestamp
                auto timestamp_value = std::static_pointer_cast<arrow::Int64Array>(
                    timestamp_array)->Value(i);
                trade.timestamp = TimePoint(NanoTime(timestamp_value));
                
                // Parse price and volume
                trade.price = std::static_pointer_cast<arrow::DoubleArray>(
                    price_array)->Value(i);
                trade.volume = std::static_pointer_cast<arrow::UInt64Array>(
                    size_array)->Value(i);
                
                trades.push_back(trade);
            }
        }
        
        return trades;
    }
    
    // Read quotes data from Polygon flat files
    std::vector<Quote> readQuotes(
        const std::string& symbol,
        const std::string& date
    ) {
        // Construct URL for quotes flat file
        std::string url = "https://files.polygon.io/flatfiles/us_stocks_sip/quotes_v1/" 
                         + date + "/" + symbol + ".csv.zst";
        
        auto data = downloadAndDecompress(url);
        
        std::vector<Quote> quotes;
        quotes.reserve(2000000);  // Pre-allocate
        
        // Similar Arrow CSV parsing for quotes
        arrow::io::IOContext io_context;
        auto input = std::make_shared<arrow::io::BufferReader>(data);
        
        auto read_options = arrow::csv::ReadOptions::Defaults();
        read_options.column_names = {
            "participant_timestamp", "bid_price", "bid_size",
            "bid_exchange", "ask_price", "ask_size", "ask_exchange"
        };
        
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        auto convert_options = arrow::csv::ConvertOptions::Defaults();
        
        auto maybe_reader = arrow::csv::StreamingReader::Make(
            io_context, input, read_options, parse_options, convert_options
        );
        
        if (!maybe_reader.ok()) {
            throw std::runtime_error("Failed to create CSV reader");
        }
        
        auto reader = *maybe_reader;
        
        while (true) {
            auto maybe_batch = reader->Next();
            if (!maybe_batch.ok()) break;
            
            auto batch = *maybe_batch;
            if (batch == nullptr) break;
            
            for (int64_t i = 0; i < batch->num_rows(); ++i) {
                Quote quote;
                
                auto timestamp_value = std::static_pointer_cast<arrow::Int64Array>(
                    batch->column(0))->Value(i);
                quote.timestamp = TimePoint(NanoTime(timestamp_value));
                
                quote.bid_price = std::static_pointer_cast<arrow::DoubleArray>(
                    batch->column(1))->Value(i);
                quote.bid_size = std::static_pointer_cast<arrow::UInt64Array>(
                    batch->column(2))->Value(i);
                
                quote.ask_price = std::static_pointer_cast<arrow::DoubleArray>(
                    batch->column(4))->Value(i);
                quote.ask_size = std::static_pointer_cast<arrow::UInt64Array>(
                    batch->column(5))->Value(i);
                
                quotes.push_back(quote);
            }
        }
        
        return quotes;
    }
    
    // Compute returns at nanosecond resolution
    std::vector<double> computeReturns(
        const std::vector<Trade>& trades,
        int64_t interval_ns
    ) {
        if (trades.size() < 2) return {};
        
        std::vector<double> returns;
        returns.reserve(trades.size() - 1);
        
        auto current_interval_start = trades[0].timestamp;
        double interval_start_price = trades[0].price;
        
        for (size_t i = 1; i < trades.size(); ++i) {
            auto time_diff = trades[i].timestamp - current_interval_start;
            
            if (time_diff.count() >= interval_ns) {
                // Calculate return for this interval
                double return_val = std::log(trades[i].price / interval_start_price);
                returns.push_back(return_val);
                
                // Reset for next interval
                current_interval_start = trades[i].timestamp;
                interval_start_price = trades[i].price;
            }
        }
        
        return returns;
    }
    
    // Compute bid-ask spread
    std::vector<double> computeBidAskSpread(const std::vector<Quote>& quotes) {
        std::vector<double> spreads;
        spreads.reserve(quotes.size());
        
        for (const auto& quote : quotes) {
            double spread = (quote.ask_price - quote.bid_price) / 
                           ((quote.ask_price + quote.bid_price) / 2.0);
            spreads.push_back(spread);
        }
        
        return spreads;
    }
};

// C interface for Python bindings
extern "C" {
    void* create_polygon_reader(const char* api_key) {
        return new PolygonDataReader(std::string(api_key));
    }
    
    void destroy_polygon_reader(void* reader) {
        delete static_cast<PolygonDataReader*>(reader);
    }
    
    void read_trades_data(
        void* reader,
        const char* symbol,
        const char* date,
        double** prices,
        uint64_t** volumes,
        int64_t** timestamps,
        size_t* count
    ) {
        auto polygon_reader = static_cast<PolygonDataReader*>(reader);
        auto trades = polygon_reader->readTrades(symbol, date);
        
        *count = trades.size();
        *prices = new double[*count];
        *volumes = new uint64_t[*count];
        *timestamps = new int64_t[*count];
        
        for (size_t i = 0; i < *count; ++i) {
            (*prices)[i] = trades[i].price;
            (*volumes)[i] = trades[i].volume;
            (*timestamps)[i] = trades[i].timestamp.time_since_epoch().count();
        }
    }
    
    void compute_returns_from_trades(
        void* reader,
        const char* symbol,
        const char* date,
        int64_t interval_ns,
        double** returns,
        size_t* count
    ) {
        auto polygon_reader = static_cast<PolygonDataReader*>(reader);
        auto trades = polygon_reader->readTrades(symbol, date);
        auto ret_vec = polygon_reader->computeReturns(trades, interval_ns);
        
        *count = ret_vec.size();
        *returns = new double[*count];
        std::copy(ret_vec.begin(), ret_vec.end(), *returns);
    }
}

