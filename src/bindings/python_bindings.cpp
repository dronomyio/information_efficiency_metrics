// Simple C interface for Python ctypes (no pybind11 required)
#include <cstring>
#include <cstdlib>
#include <cstdint>  // Add this for uint64_t and int64_t

extern "C" {
    // Forward declarations of the actual functions
    void* create_vr_calculator();
    void destroy_vr_calculator(void* calculator);
    void compute_variance_ratios(void* calculator, const double* returns,
                                 int n_returns, const int* horizons,
                                 int n_horizons, double* results);
    
    void* create_autocorr_processor(int num_threads);
    void destroy_autocorr_processor(void* processor);
    void compute_autocorrelations(void* processor, const double* returns,
                                  size_t n_returns, size_t max_lag, double* results);
    
    void* create_polygon_reader(const char* api_key);
    void destroy_polygon_reader(void* reader);
    void compute_returns_from_trades(void* reader, const char* symbol,
                                     const char* date, int64_t interval_ns,
                                     double** returns, size_t* count);
    
    void read_trades_data(void* reader, const char* symbol, const char* date,
                         double** prices, uint64_t** volumes, 
                         int64_t** timestamps, size_t* count);
    
    // Export these functions directly for ctypes
    extern "C" __attribute__((visibility("default"))) void* cpp_create_vr_calculator() {
        return create_vr_calculator();
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_destroy_vr_calculator(void* calc) {
        destroy_vr_calculator(calc);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_compute_variance_ratios(
        void* calc, const double* returns, int n_returns,
        const int* horizons, int n_horizons, double* results) {
        compute_variance_ratios(calc, returns, n_returns, horizons, n_horizons, results);
    }
    
    extern "C" __attribute__((visibility("default"))) void* cpp_create_autocorr_processor(int n) {
        return create_autocorr_processor(n);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_destroy_autocorr_processor(void* p) {
        destroy_autocorr_processor(p);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_compute_autocorrelations(
        void* proc, const double* returns, size_t n_returns,
        size_t max_lag, double* results) {
        compute_autocorrelations(proc, returns, n_returns, max_lag, results);
    }
    
    extern "C" __attribute__((visibility("default"))) void* cpp_create_polygon_reader(const char* key) {
        return create_polygon_reader(key);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_destroy_polygon_reader(void* reader) {
        destroy_polygon_reader(reader);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_compute_returns_from_trades(
        void* reader, const char* symbol, const char* date, 
        int64_t interval_ns, double** returns, size_t* count) {
        compute_returns_from_trades(reader, symbol, date, interval_ns, returns, count);
    }
    
    extern "C" __attribute__((visibility("default"))) void cpp_read_trades_data(
        void* reader, const char* symbol, const char* date,
        double** prices, uint64_t** volumes, int64_t** timestamps, size_t* count) {
        read_trades_data(reader, symbol, date, prices, volumes, timestamps, count);
    }
}
