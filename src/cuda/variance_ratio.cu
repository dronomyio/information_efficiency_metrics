// src/cuda/variance_ratio.cu - Full CUDA implementation
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <cmath>

namespace cg = cooperative_groups;

// Constants
#define WARP_SIZE 32
#define MAX_THREADS 1024
#define SHARED_MEM_SIZE 49152  // 48KB shared memory

// Custom atomicAdd for double precision (for older GPUs)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Native atomicAdd for compute capability 6.0+
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Warp-level reduction for variance computation
template<typename T>
__device__ T warpReduce(T val) {
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<typename T, int BLOCK_SIZE>
__device__ T blockReduce(T val) {
    __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warpReduce(val);
    
    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Final warp reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduce(val);
    
    return val;
}

// Kernel for computing h-period returns
__global__ void computeHPeriodReturns(
    const double* __restrict__ returns,
    double* __restrict__ h_returns,
    int n,
    int h
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n - h + 1; i += stride) {
        double sum = 0.0;
        
        // Unrolled loop for better performance
        #pragma unroll 8
        for (int j = 0; j < h; ++j) {
            sum += returns[i + j];
        }
        h_returns[i] = sum;
    }
}

// Kernel for computing mean using Welford's algorithm
__global__ void computeMeanKernel(
    const double* __restrict__ data,
    double* __restrict__ mean,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    double local_sum = 0.0;
    int local_count = 0;
    
    // Grid-stride loop
    for (int i = tid; i < n; i += stride) {
        local_sum += data[i];
        local_count++;
    }
    
    // Block-level reduction
    local_sum = blockReduce<double, MAX_THREADS>(local_sum);
    local_count = blockReduce<int, MAX_THREADS>(local_count);
    
    // Write result
    if (threadIdx.x == 0) {
        atomicAdd(mean, local_sum / n);
    }
}

// Kernel for computing variance with Welford's algorithm
__global__ void computeVarianceWelford(
    const double* __restrict__ data,
    double* __restrict__ variance,
    double mean,
    int n
) {
    extern __shared__ double shared_data[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    double local_variance = 0.0;
    
    // Grid-stride loop for variance computation
    for (int i = tid; i < n; i += stride) {
        double diff = data[i] - mean;
        local_variance += diff * diff;
    }
    
    // Block-level reduction
    local_variance = blockReduce<double, MAX_THREADS>(local_variance);
    
    // Write result
    if (threadIdx.x == 0) {
        atomicAdd(variance, local_variance / (n - 1));
    }
}

// Optimized kernel for small horizons
__global__ void computeVarianceRatioSmallH(
    const double* __restrict__ returns,
    double* __restrict__ results,
    int n,
    int h
) {
    extern __shared__ double shared_mem[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Compute statistics in shared memory
    if (tid < n && tid < blockDim.x) {
        shared_mem[tid] = returns[bid * blockDim.x + tid];
    }
    __syncthreads();
    
    // Compute local variance ratio
    if (tid == 0) {
        double mean = 0.0;
        for (int i = 0; i < min(n, blockDim.x); i++) {
            mean += shared_mem[i];
        }
        mean /= min(n, blockDim.x);
        
        double var_1 = 0.0;
        for (int i = 0; i < min(n, blockDim.x); i++) {
            double diff = shared_mem[i] - mean;
            var_1 += diff * diff;
        }
        var_1 /= (min(n, blockDim.x) - 1);
        
        // Store partial result
        results[bid] = var_1;
    }
}

// Multi-GPU variance ratio computation class
class VarianceRatioMultiGPU {
private:
    int num_gpus;
    std::vector<cudaStream_t> streams;
    std::vector<cublasHandle_t> cublas_handles;
    std::vector<double*> d_workspace;
    size_t workspace_size;
    
public:
    VarianceRatioMultiGPU() {
        cudaGetDeviceCount(&num_gpus);
        printf("Initializing with %d GPUs\n", num_gpus);
        
        streams.resize(num_gpus);
        cublas_handles.resize(num_gpus);
        d_workspace.resize(num_gpus);
        workspace_size = 10000000 * sizeof(double);  // 10M doubles workspace
        
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
            cublasCreate(&cublas_handles[i]);
            cublasSetStream(cublas_handles[i], streams[i]);
            cudaMalloc(&d_workspace[i], workspace_size);
            
            // Print GPU info
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("GPU %d: %s (SM %d.%d, %d SMs, %.1f GB)\n", 
                   i, prop.name, prop.major, prop.minor,
                   prop.multiProcessorCount,
                   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    
    ~VarianceRatioMultiGPU() {
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamDestroy(streams[i]);
            cublasDestroy(cublas_handles[i]);
            cudaFree(d_workspace[i]);
        }
    }
    
    std::vector<double> computeVR(
        const std::vector<double>& returns,
        const std::vector<int>& horizons
    ) {
        int n = returns.size();
        std::vector<double> vr_results(horizons.size());
        
        // Distribute work across GPUs
        int chunk_size = (horizons.size() + num_gpus - 1) / num_gpus;
        
        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            cudaSetDevice(gpu);
            
            int start_idx = gpu * chunk_size;
            int end_idx = std::min((gpu + 1) * chunk_size, (int)horizons.size());
            
            if (start_idx >= horizons.size()) continue;
            
            // Allocate device memory
            double *d_returns, *d_h_returns, *d_mean, *d_variance;
            cudaMalloc(&d_returns, n * sizeof(double));
            cudaMemcpyAsync(d_returns, returns.data(), n * sizeof(double), 
                           cudaMemcpyHostToDevice, streams[gpu]);
            
            // Process each horizon on this GPU
            for (int i = start_idx; i < end_idx; i++) {
                int h = horizons[i];
                int h_size = n - h + 1;
                
                if (h_size <= 0) {
                    vr_results[i] = 1.0;
                    continue;
                }
                
                cudaMalloc(&d_h_returns, h_size * sizeof(double));
                cudaMalloc(&d_mean, sizeof(double));
                cudaMalloc(&d_variance, sizeof(double));
                
                cudaMemsetAsync(d_mean, 0, sizeof(double), streams[gpu]);
                cudaMemsetAsync(d_variance, 0, sizeof(double), streams[gpu]);
                
                // Compute h-period returns
                int block_size = 256;
                int grid_size = (h_size + block_size - 1) / block_size;
                computeHPeriodReturns<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_returns, d_h_returns, n, h
                );
                
                // Compute mean of 1-period returns
                computeMeanKernel<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_returns, d_mean, n
                );
                
                // Get mean from device
                double mean_1;
                cudaMemcpyAsync(&mean_1, d_mean, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                cudaStreamSynchronize(streams[gpu]);
                
                // Compute variance of 1-period returns
                cudaMemsetAsync(d_variance, 0, sizeof(double), streams[gpu]);
                computeVarianceWelford<<<grid_size, block_size, 
                                        block_size * sizeof(double), streams[gpu]>>>(
                    d_returns, d_variance, mean_1, n
                );
                
                double var_1;
                cudaMemcpyAsync(&var_1, d_variance, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                
                // Compute mean of h-period returns
                cudaMemsetAsync(d_mean, 0, sizeof(double), streams[gpu]);
                computeMeanKernel<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_h_returns, d_mean, h_size
                );
                
                double mean_h;
                cudaMemcpyAsync(&mean_h, d_mean, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                cudaStreamSynchronize(streams[gpu]);
                
                // Compute variance of h-period returns
                cudaMemsetAsync(d_variance, 0, sizeof(double), streams[gpu]);
                computeVarianceWelford<<<grid_size, block_size,
                                        block_size * sizeof(double), streams[gpu]>>>(
                    d_h_returns, d_variance, mean_h, h_size
                );
                
                double var_h;
                cudaMemcpyAsync(&var_h, d_variance, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                cudaStreamSynchronize(streams[gpu]);
                
                // Calculate variance ratio
                vr_results[i] = (var_1 > 0) ? (var_h / (h * var_1)) : 1.0;
                
                // Cleanup
                cudaFree(d_h_returns);
                cudaFree(d_mean);
                cudaFree(d_variance);
            }
            
            cudaFree(d_returns);
        }
        
        return vr_results;
    }
    
    // Batch processing for multiple securities
    std::vector<std::vector<double>> computeBatchVR(
        const std::vector<std::vector<double>>& batch_returns,
        const std::vector<int>& horizons
    ) {
        std::vector<std::vector<double>> results(batch_returns.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < batch_returns.size(); i++) {
            results[i] = computeVR(batch_returns[i], horizons);
        }
        
        return results;
    }
};

// Global instance
static VarianceRatioMultiGPU* g_vr_calculator = nullptr;

// C interface for external use
extern "C" {
    void* create_vr_calculator() {
        if (!g_vr_calculator) {
            g_vr_calculator = new VarianceRatioMultiGPU();
        }
        return g_vr_calculator;
    }
    
    void destroy_vr_calculator(void* calculator) {
        if (g_vr_calculator) {
            delete g_vr_calculator;
            g_vr_calculator = nullptr;
        }
    }
    
    void compute_variance_ratios(
        void* calculator,
        const double* returns,
        int n_returns,
        const int* horizons,
        int n_horizons,
        double* results
    ) {
        if (!g_vr_calculator) {
            g_vr_calculator = new VarianceRatioMultiGPU();
        }
        
        std::vector<double> ret_vec(returns, returns + n_returns);
        std::vector<int> hor_vec(horizons, horizons + n_horizons);
        
        auto vr_results = g_vr_calculator->computeVR(ret_vec, hor_vec);
        
        for (int i = 0; i < n_horizons; i++) {
            results[i] = vr_results[i];
        }
    }
    
    // Additional functions for testing
    void test_variance_ratio() {
        printf("Testing Variance Ratio CUDA Implementation...\n");
        
        // Generate test data
        std::vector<double> test_returns(10000);
        for (int i = 0; i < 10000; i++) {
            test_returns[i] = (rand() % 200 - 100) / 10000.0;
        }
        
        std::vector<int> test_horizons = {2, 5, 10, 20, 50, 100};
        
        VarianceRatioMultiGPU vr_calc;
        auto results = vr_calc.computeVR(test_returns, test_horizons);
        
        printf("Variance Ratio Results:\n");
        for (size_t i = 0; i < test_horizons.size(); i++) {
            printf("  VR(%d) = %.4f\n", test_horizons[i], results[i]);
        }
    }
}
