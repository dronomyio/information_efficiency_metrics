// src/simd/simd_operations.cpp
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "simd_operations.h"

// AVX-512 autocorrelation computation
class AutocorrelationSIMD {
private:
    static constexpr int SIMD_WIDTH = 8;  // AVX-512 processes 8 doubles
    
    // Compute mean using AVX-512
    double computeMeanAVX512(const double* data, size_t n) {
        __m512d sum_vec = _mm512_setzero_pd();
        size_t simd_end = n - (n % SIMD_WIDTH);
        
        // Main SIMD loop
        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
            __m512d data_vec = _mm512_loadu_pd(&data[i]);
            sum_vec = _mm512_add_pd(sum_vec, data_vec);
        }
        
        // Horizontal sum
        double sum = _mm512_reduce_add_pd(sum_vec);
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            sum += data[i];
        }
        
        return sum / n;
    }
    
    // Compute variance using AVX-512
    double computeVarianceAVX512(const double* data, size_t n, double mean) {
        __m512d mean_vec = _mm512_set1_pd(mean);
        __m512d sum_sq_vec = _mm512_setzero_pd();
        size_t simd_end = n - (n % SIMD_WIDTH);
        
        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
            __m512d data_vec = _mm512_loadu_pd(&data[i]);
            __m512d diff_vec = _mm512_sub_pd(data_vec, mean_vec);
            __m512d sq_vec = _mm512_mul_pd(diff_vec, diff_vec);
            sum_sq_vec = _mm512_add_pd(sum_sq_vec, sq_vec);
        }
        
        double sum_sq = _mm512_reduce_add_pd(sum_sq_vec);
        
        for (size_t i = simd_end; i < n; ++i) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
        
        return sum_sq / (n - 1);
    }
    
public:
    // Compute autocorrelation for multiple lags using SIMD
    std::vector<double> computeAutocorrelation(
        const double* returns,
        size_t n,
        size_t max_lag
    ) {
        std::vector<double> acf(max_lag + 1);
        
        double mean = computeMeanAVX512(returns, n);
        double variance = computeVarianceAVX512(returns, n, mean);
        
        acf[0] = 1.0;  // Lag 0 autocorrelation is always 1
        
        // Parallel computation of autocorrelations for different lags
        #pragma omp parallel for schedule(dynamic)
        for (size_t lag = 1; lag <= max_lag; ++lag) {
            __m512d mean_vec = _mm512_set1_pd(mean);
            __m512d cov_vec = _mm512_setzero_pd();
            
            size_t simd_end = (n - lag) - ((n - lag) % SIMD_WIDTH);
            
            // SIMD loop for covariance computation
            for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
                __m512d x_vec = _mm512_loadu_pd(&returns[i]);
                __m512d y_vec = _mm512_loadu_pd(&returns[i + lag]);
                
                __m512d x_centered = _mm512_sub_pd(x_vec, mean_vec);
                __m512d y_centered = _mm512_sub_pd(y_vec, mean_vec);
                
                __m512d prod = _mm512_mul_pd(x_centered, y_centered);
                cov_vec = _mm512_add_pd(cov_vec, prod);
            }
            
            double covariance = _mm512_reduce_add_pd(cov_vec);
            
            // Handle remaining elements
            for (size_t i = simd_end; i < n - lag; ++i) {
                covariance += (returns[i] - mean) * (returns[i + lag] - mean);
            }
            
            acf[lag] = covariance / ((n - lag) * variance);
        }
        
        return acf;
    }
    
    // Compute decay parameters using SIMD-accelerated least squares
    std::pair<double, double> computeDecayParameters(
        const std::vector<double>& acf
    ) {
        size_t n = acf.size() - 1;  // Exclude lag 0
        
        // Prepare data for regression (log of absolute ACF values)
        std::vector<double> log_acf(n);
        std::vector<double> lags(n);
        
        #pragma omp parallel for simd
        for (size_t i = 0; i < n; ++i) {
            log_acf[i] = std::log(std::abs(acf[i + 1]));
            lags[i] = static_cast<double>(i + 1);
        }
        
        // Compute regression coefficients using SIMD
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        
        size_t simd_end = n - (n % SIMD_WIDTH);
        
        __m512d sum_x_vec = _mm512_setzero_pd();
        __m512d sum_y_vec = _mm512_setzero_pd();
        __m512d sum_xx_vec = _mm512_setzero_pd();
        __m512d sum_xy_vec = _mm512_setzero_pd();
        
        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
            __m512d x_vec = _mm512_loadu_pd(&lags[i]);
            __m512d y_vec = _mm512_loadu_pd(&log_acf[i]);
            
            sum_x_vec = _mm512_add_pd(sum_x_vec, x_vec);
            sum_y_vec = _mm512_add_pd(sum_y_vec, y_vec);
            sum_xx_vec = _mm512_fmadd_pd(x_vec, x_vec, sum_xx_vec);
            sum_xy_vec = _mm512_fmadd_pd(x_vec, y_vec, sum_xy_vec);
        }
        
        sum_x = _mm512_reduce_add_pd(sum_x_vec);
        sum_y = _mm512_reduce_add_pd(sum_y_vec);
        sum_xx = _mm512_reduce_add_pd(sum_xx_vec);
        sum_xy = _mm512_reduce_add_pd(sum_xy_vec);
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            sum_x += lags[i];
            sum_y += log_acf[i];
            sum_xx += lags[i] * lags[i];
            sum_xy += lags[i] * log_acf[i];
        }
        
        // Calculate regression coefficients
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        double phi = std::exp(slope);
        double half_life = std::log(0.5) / slope;
        
        return {phi, half_life};
    }
};

// Multi-threaded batch processing
class BatchProcessor {
private:
    int num_threads;
    
public:
    BatchProcessor(int threads = 0) {
        num_threads = threads > 0 ? threads : omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
    
    // Process multiple time series in parallel
    std::vector<std::vector<double>> processBatch(
        const std::vector<std::vector<double>>& batch_returns,
        size_t max_lag
    ) {
        size_t batch_size = batch_returns.size();
        std::vector<std::vector<double>> results(batch_size);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < batch_size; ++i) {
            AutocorrelationSIMD processor;
            results[i] = processor.computeAutocorrelation(
                batch_returns[i].data(),
                batch_returns[i].size(),
                max_lag
            );
        }
        
        return results;
    }
};

// C interface for Python bindings
extern "C" {
    void* create_autocorr_processor(int num_threads) {
        return new BatchProcessor(num_threads);
    }
    
    void destroy_autocorr_processor(void* processor) {
        delete static_cast<BatchProcessor*>(processor);
    }
    
    void compute_autocorrelations(
        void* processor,
        const double* returns,
        size_t n_returns,
        size_t max_lag,
        double* results
    ) {
        AutocorrelationSIMD acf_processor;
        auto acf = acf_processor.computeAutocorrelation(returns, n_returns, max_lag);
        std::copy(acf.begin(), acf.end(), results);
    }
    
    void compute_batch_autocorrelations(
        void* processor,
        const double** batch_returns,
        const size_t* batch_sizes,
        size_t batch_count,
        size_t max_lag,
        double** results
    ) {
        auto batch_proc = static_cast<BatchProcessor*>(processor);
        
        std::vector<std::vector<double>> input_batch(batch_count);
        for (size_t i = 0; i < batch_count; ++i) {
            input_batch[i] = std::vector<double>(
                batch_returns[i], 
                batch_returns[i] + batch_sizes[i]
            );
        }
        
        auto output_batch = batch_proc->processBatch(input_batch, max_lag);
        
        for (size_t i = 0; i < batch_count; ++i) {
            std::copy(output_batch[i].begin(), output_batch[i].end(), results[i]);
        }
    }
}
