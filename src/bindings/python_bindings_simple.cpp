// Simple C interface for Python ctypes (no pybind11 required)
#include <cstring>

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
    
    // Export these functions directly for ctypes
    __attribute__((visibility("default"))) void* cpp_create_vr_calculator() {
        return create_vr_calculator();
    }
    
    __attribute__((visibility("default"))) void cpp_destroy_vr_calculator(void* calc) {
        destroy_vr_calculator(calc);
    }
    
    __attribute__((visibility("default"))) void cpp_compute_variance_ratios(
        void* calc, const double* returns, int n_returns,
        const int* horizons, int n_horizons, double* results) {
        compute_variance_ratios(calc, returns, n_returns, horizons, n_horizons, results);
    }
    
    __attribute__((visibility("default"))) void* cpp_create_autocorr_processor(int n) {
        return create_autocorr_processor(n);
    }
    
    __attribute__((visibility("default"))) void cpp_destroy_autocorr_processor(void* p) {
        destroy_autocorr_processor(p);
    }
    
    __attribute__((visibility("default"))) void cpp_compute_autocorrelations(
        void* proc, const double* returns, size_t n_returns,
        size_t max_lag, double* results) {
        compute_autocorrelations(proc, returns, n_returns, max_lag, results);
    }
    
    __attribute__((visibility("default"))) void* cpp_create_polygon_reader(const char* key) {
        return create_polygon_reader(key);
    }
    
    __attribute__((visibility("default"))) void cpp_destroy_polygon_reader(void* reader) {
        destroy_polygon_reader(reader);
    }
}
