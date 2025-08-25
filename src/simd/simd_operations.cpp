// src/simd/simd_operations.cpp - Full SIMD implementation with AVX-512
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <numeric>

// Check for AVX-512 support - only define if not already defined
#ifndef USE_AVX512
    #ifdef __AVX512F__
        #define USE_AVX512
        #define SIMD_WIDTH 8  // 8 doubles per AVX-512 register
    #elif __AVX2__
        #define USE_AVX2
        #define SIMD_WIDTH 4  // 4 doubles per AVX2 register
    #else
        #define SIMD_WIDTH 1  // Fallback to scalar
    #endif
#else
    #define SIMD_WIDTH 8
#endif

// AVX-512 autocorrelation computation class
class AutocorrelationSIMD {
private:
    int num_threads;
    
    // Compute mean using AVX-512
    double computeMeanAVX512(const double* data, size_t n) {
        #ifdef USE_AVX512
        __m512d sum_vec = _mm512_setzero_pd();
        size_t simd_end = n - (n % SIMD_WIDTH);
        
        // Main SIMD loop
        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
            __m512d data_vec = _mm512_loadu_pd(&data[i]);
            sum_vec = _mm512_add_pd(sum_vec, data_vec);
        }
        
        // Horizontal sum of vector elements
        double sum = _mm512_reduce_add_pd(sum_vec);
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n; ++i) {
            sum += data[i];
        }
        
        return sum / n;
        
        #elif defined(USE_AVX2)
        __m256d sum_vec = _mm256_setzero_pd();
        size_t simd_end = n - (n % 4);
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            sum_vec = _mm256_add_pd(sum_vec, data_vec);
        }
        
        // Horizontal sum for AVX2
        __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
        __m128d sum_128 = _mm_add_pd(sum_low, sum_high);
        double sum_arr[2];
        _mm_storeu_pd(sum_arr, sum_128);
        double sum = sum_arr[0] + sum_arr[1];
        
        for (size_t i = simd_end; i < n; ++i) {
            sum += data[i];
        }
        
        return sum / n;
        
        #else
        // Scalar fallback
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += data[i];
        }
        return sum / n;
        #endif
    }
    
    // Compute variance using AVX-512
    double computeVarianceAVX512(const double* data, size_t n, double mean) {
        #ifdef USE_AVX512
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
        
        #elif defined(USE_AVX2)
        __m256d mean_vec = _mm256_set1_pd(mean);
        __m256d sum_sq_vec = _mm256_setzero_pd();
        size_t simd_end = n - (n % 4);
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            __m256d diff_vec = _mm256_sub_pd(data_vec, mean_vec);
            __m256d sq_vec = _mm256_mul_pd(diff_vec, diff_vec);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_vec);
        }
        
        // Horizontal sum for AVX2
        __m128d sum_high = _mm256_extractf128_pd(sum_sq_vec, 1);
        __m128d sum_low = _mm256_castpd256_pd128(sum_sq_vec);
        __m128d sum_128 = _mm_add_pd(sum_low, sum_high);
        double sum_arr[2];
        _mm_storeu_pd(sum_arr, sum_128);
        double sum_sq = sum_arr[0] + sum_arr[1];
        
        for (size_t i = simd_end; i < n; ++i) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
        
        return sum_sq / (n - 1);
        
        #else
        // Scalar fallback
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
        return sum_sq / (n - 1);
        #endif
    }
    
    // Compute covariance for a specific lag using SIMD
    double computeCovarianceSIMD(const double* data, size_t n, size_t lag, double mean) {
        if (lag >= n) return 0.0;
        
        #ifdef USE_AVX512
        __m512d mean_vec = _mm512_set1_pd(mean);
        __m512d cov_vec = _mm512_setzero_pd();
        size_t simd_end = (n - lag) - ((n - lag) % SIMD_WIDTH);
        
        for (size_t i = 0; i < simd_end; i += SIMD_WIDTH) {
            __m512d x_vec = _mm512_loadu_pd(&data[i]);
            __m512d y_vec = _mm512_loadu_pd(&data[i + lag]);
            
            __m512d x_centered = _mm512_sub_pd(x_vec, mean_vec);
            __m512d y_centered = _mm512_sub_pd(y_vec, mean_vec);
            
            __m512d prod = _mm512_mul_pd(x_centered, y_centered);
            cov_vec = _mm512_add_pd(cov_vec, prod);
        }
        
        double covariance = _mm512_reduce_add_pd(cov_vec);
        
        // Handle remaining elements
        for (size_t i = simd_end; i < n - lag; ++i) {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }
        
        return covariance / (n - lag);
        
        #elif defined(USE_AVX2)
        __m256d mean_vec = _mm256_set1_pd(mean);
        __m256d cov_vec = _mm256_setzero_pd();
        size_t simd_end = (n - lag) - ((n - lag) % 4);
        
        for (size_t i = 0; i < simd_end; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&data[i]);
            __m256d y_vec = _mm256_loadu_pd(&data[i + lag]);
            
            __m256d x_centered = _mm256_sub_pd(x_vec, mean_vec);
            __m256d y_centered = _mm256_sub_pd(y_vec, mean_vec);
            
            __m256d prod = _mm256_mul_pd(x_centered, y_centered);
            cov_vec = _mm256_add_pd(cov_vec, prod);
        }
        
        // Horizontal sum
        __m128d sum_high = _mm256_extractf128_pd(cov_vec, 1);
        __m128d sum_low = _mm256_castpd256_pd128(cov_vec);
        __m128d sum_128 = _mm_add_pd(sum_low, sum_high);
        double sum_arr[2];
        _mm_storeu_pd(sum_arr, sum_128);
        double covariance = sum_arr[0] + sum_arr[1];
        
        for (size_t i = simd_end; i < n - lag; ++i) {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }
        
        return covariance / (n - lag);
        
        #else
        // Scalar fallback
        double covariance = 0.0;
        for (size_t i = 0; i < n - lag; ++i) {
            covariance += (data[i] - mean) * (data[i + lag] - mean);
        }
        return covariance / (n - lag);
        #endif
    }
    
public:
    AutocorrelationSIMD(int threads = 0) {
        num_threads = threads > 0 ? threads : omp_get_max_threads();
        omp_set_num_threads(num_threads);
        
        printf("AutocorrelationSIMD initialized with %d threads\n", num_threads);
        #ifdef USE_AVX512
        printf("Using AVX-512 instructions (8 doubles per operation)\n");
        #elif defined(USE_AVX2)
        printf("Using AVX2 instructions (4 doubles per operation)\n");
        #else
        printf("Using scalar operations (no SIMD)\n");
        #endif
    }
    
    // Compute autocorrelation for multiple lags using SIMD
    std::vector<double> computeAutocorrelation(
        const double* returns,
        size_t n,
        size_t max_lag
    ) {
        std::vector<double> acf(max_lag + 1);
        
        // Compute mean
        double mean = computeMeanAVX512(returns, n);
        
        // Compute variance
        double variance = computeVarianceAVX512(returns, n, mean);
        
        if (variance == 0.0) {
            // If variance is zero, all autocorrelations are undefined
            std::fill(acf.begin(), acf.end(), 0.0);
            acf[0] = 1.0;
            return acf;
        }
        
        acf[0] = 1.0;  // Lag 0 autocorrelation is always 1
        
        // Sequential computation for autocorrelations
        // OpenMP doesn't work well with complex loop conditions
        size_t max_compute_lag = std::min(max_lag, n - 1);
        
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t lag = 1; lag <= max_compute_lag; ++lag) {
            double covariance = computeCovarianceSIMD(returns, n, lag, mean);
            acf[lag] = covariance / variance;
        }
        
        // Fill remaining lags with zero if max_lag >= n
        for (size_t lag = max_compute_lag + 1; lag <= max_lag; ++lag) {
            acf[lag] = 0.0;
        }
        
        return acf;
    }
    
    // Compute decay parameters using SIMD-accelerated least squares
    std::pair<double, double> computeDecayParameters(
        const std::vector<double>& acf
    ) {
        if (acf.size() < 2) return {0.0, 0.0};
        
        size_t n = acf.size() - 1;  // Exclude lag 0
        
        // Prepare data for regression (log of absolute ACF values)
        std::vector<double> log_acf(n);
        std::vector<double> lags(n);
        
        for (size_t i = 0; i < n; ++i) {
            log_acf[i] = std::log(std::max(std::abs(acf[i + 1]), 1e-10));
            lags[i] = static_cast<double>(i + 1);
        }
        
        // Compute regression coefficients using SIMD
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        
        #ifdef USE_AVX512
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
        #else
        // Scalar computation
        for (size_t i = 0; i < n; ++i) {
            sum_x += lags[i];
            sum_y += log_acf[i];
            sum_xx += lags[i] * lags[i];
            sum_xy += lags[i] * log_acf[i];
        }
        #endif
        
        // Calculate regression coefficients
        double denominator = n * sum_xx - sum_x * sum_x;
        if (std::abs(denominator) < 1e-10) {
            return {1.0, INFINITY};
        }
        
        double slope = (n * sum_xy - sum_x * sum_y) / denominator;
        double phi = std::exp(slope);
        double half_life = (slope != 0.0) ? std::log(0.5) / slope : INFINITY;
        
        return {phi, half_life};
    }
    
    // Compute partial autocorrelation function using Levinson-Durbin recursion
    std::vector<double> computePartialAutocorrelation(
        const std::vector<double>& acf,
        size_t max_lag
    ) {
        std::vector<double> pacf(max_lag + 1, 0.0);
        if (acf.empty()) return pacf;
        
        pacf[0] = 1.0;
        if (max_lag == 0 || acf.size() < 2) return pacf;
        
        pacf[1] = acf[1];
        
        std::vector<double> phi(max_lag + 1);
        std::vector<double> phi_new(max_lag + 1);
        
        phi[1] = acf[1];
        
        for (size_t k = 2; k <= max_lag && k < acf.size(); ++k) {
            double numerator = acf[k];
            double denominator = 1.0;
            
            for (size_t j = 1; j < k; ++j) {
                numerator -= phi[j] * acf[k - j];
                denominator -= phi[j] * acf[j];
            }
            
            if (std::abs(denominator) < 1e-10) {
                pacf[k] = 0.0;
            } else {
                pacf[k] = numerator / denominator;
            }
            
            // Update phi coefficients
            for (size_t j = 1; j < k; ++j) {
                phi_new[j] = phi[j] - pacf[k] * phi[k - j];
            }
            phi_new[k] = pacf[k];
            
            std::swap(phi, phi_new);
        }
        
        return pacf;
    }
};

// Multi-threaded batch processor
class BatchProcessor {
private:
    int num_threads;
    std::vector<AutocorrelationSIMD> processors;
    
public:
    BatchProcessor(int threads = 0) {
        num_threads = threads > 0 ? threads : omp_get_max_threads();
        processors.reserve(num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            processors.emplace_back(1);  // Each processor uses 1 thread
        }
        
        printf("BatchProcessor initialized with %d processors\n", num_threads);
    }
    
    // Process multiple time series in parallel
    std::vector<std::vector<double>> processBatch(
        const std::vector<std::vector<double>>& batch_returns,
        size_t max_lag
    ) {
        size_t batch_size = batch_returns.size();
        std::vector<std::vector<double>> results(batch_size);
        
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            int thread_id = omp_get_thread_num();
            results[i] = processors[thread_id].computeAutocorrelation(
                batch_returns[i].data(),
                batch_returns[i].size(),
                max_lag
            );
        }
        
        return results;
    }
    
    // Compute statistics across batch
    std::vector<double> computeBatchStatistics(
        const std::vector<std::vector<double>>& batch_acf
    ) {
        if (batch_acf.empty() || batch_acf[0].empty()) {
            return std::vector<double>();
        }
        
        size_t max_lag = batch_acf[0].size() - 1;
        std::vector<double> mean_acf(max_lag + 1, 0.0);
        
        // Compute mean ACF across batch
        // Use simple parallel reduction without array syntax
        #pragma omp parallel for
        for (size_t lag = 0; lag <= max_lag; ++lag) {
            double sum = 0.0;
            for (const auto& acf : batch_acf) {
                if (lag < acf.size()) {
                    sum += acf[lag];
                }
            }
            mean_acf[lag] = sum / batch_acf.size();
        }
        
        return mean_acf;
    }
};

// Global instances
static AutocorrelationSIMD* g_acf_processor = nullptr;
static BatchProcessor* g_batch_processor = nullptr;

// C interface for external use
extern "C" {
    void* create_autocorr_processor(int num_threads) {
        if (!g_acf_processor) {
            g_acf_processor = new AutocorrelationSIMD(num_threads);
        }
        return g_acf_processor;
    }
    
    void destroy_autocorr_processor(void* processor) {
        if (g_acf_processor) {
            delete g_acf_processor;
            g_acf_processor = nullptr;
        }
    }
    
    void compute_autocorrelations(
        void* processor,
        const double* returns,
        size_t n_returns,
        size_t max_lag,
        double* results
    ) {
        if (!g_acf_processor) {
            g_acf_processor = new AutocorrelationSIMD();
        }
        
        auto acf = g_acf_processor->computeAutocorrelation(returns, n_returns, max_lag);
        
        for (size_t i = 0; i <= max_lag && i < acf.size(); ++i) {
            results[i] = acf[i];
        }
    }
    
    void compute_batch_autocorrelations(
        void* processor,
        const double** batch_returns,
        const size_t* batch_sizes,
        size_t batch_count,
        size_t max_lag,
        double** results
    ) {
        if (!g_batch_processor) {
            g_batch_processor = new BatchProcessor();
        }
        
        std::vector<std::vector<double>> input_batch(batch_count);
        for (size_t i = 0; i < batch_count; ++i) {
            input_batch[i] = std::vector<double>(
                batch_returns[i], 
                batch_returns[i] + batch_sizes[i]
            );
        }
        
        auto output_batch = g_batch_processor->processBatch(input_batch, max_lag);
        
        for (size_t i = 0; i < batch_count; ++i) {
            for (size_t j = 0; j <= max_lag && j < output_batch[i].size(); ++j) {
                results[i][j] = output_batch[i][j];
            }
        }
    }
    
    // Additional test function
    void test_autocorrelation() {
        printf("Testing Autocorrelation SIMD Implementation...\n");
        
        // Generate test data
        std::vector<double> test_returns(10000);
        for (int i = 0; i < 10000; i++) {
            test_returns[i] = (rand() % 200 - 100) / 10000.0;
        }
        
        AutocorrelationSIMD acf_processor;
        auto acf = acf_processor.computeAutocorrelation(test_returns.data(), test_returns.size(), 100);
        
        printf("Autocorrelation Results:\n");
        for (size_t i = 0; i <= 10; i++) {
            printf("  ACF(%zu) = %.4f\n", i, acf[i]);
        }
        
        auto [phi, half_life] = acf_processor.computeDecayParameters(acf);
        printf("Decay parameter (phi): %.4f\n", phi);
        printf("Half-life: %.2f periods\n", half_life);
    }
}
