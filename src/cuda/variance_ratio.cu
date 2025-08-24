// src/cuda/variance_ratio.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cooperative_groups.h>
#include "kernels.cuh"

namespace cg = cooperative_groups;

// Warp-level reduction for variance computation
template<typename T>
__device__ T warpReduce(T val) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Block-level reduction
template<typename T, int BLOCK_SIZE>
__device__ T blockReduce(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduce(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
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
    
    if (tid < n - h + 1) {
        double sum = 0.0;
        #pragma unroll 4
        for (int i = 0; i < h; ++i) {
            sum += returns[tid + i];
        }
        h_returns[tid] = sum;
    }
}

// Kernel for computing variance with Welford's algorithm
__global__ void computeVarianceWelford(
    const double* __restrict__ data,
    double* __restrict__ variance,
    double* __restrict__ mean,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    double local_mean = 0.0;
    double local_m2 = 0.0;
    int local_count = 0;
    
    // Welford's online algorithm
    for (int i = tid; i < n; i += stride) {
        local_count++;
        double delta = data[i] - local_mean;
        local_mean += delta / local_count;
        double delta2 = data[i] - local_mean;
        local_m2 += delta * delta2;
    }
    
    // Reduce within block
    local_mean = blockReduce<double, 256>(local_mean * local_count) / n;
    local_m2 = blockReduce<double, 256>(local_m2);
    
    if (threadIdx.x == 0) {
        atomicAdd(mean, local_mean);
        atomicAdd(variance, local_m2 / (n - 1));
    }
}

// Multi-GPU variance ratio computation
class VarianceRatioMultiGPU {
private:
    int num_gpus;
    std::vector<cudaStream_t> streams;
    std::vector<cublasHandle_t> cublas_handles;
    
public:
    VarianceRatioMultiGPU() {
        cudaGetDeviceCount(&num_gpus);
        streams.resize(num_gpus);
        cublas_handles.resize(num_gpus);
        
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
            cublasCreate(&cublas_handles[i]);
            cublasSetStream(cublas_handles[i], streams[i]);
        }
    }
    
    ~VarianceRatioMultiGPU() {
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamDestroy(streams[i]);
            cublasDestroy(cublas_handles[i]);
        }
    }
    
    std::vector<double> computeVR(
        const std::vector<double>& returns,
        const std::vector<int>& horizons
    ) {
        int n = returns.size();
        std::vector<double> vr_results(horizons.size());
        
        // Distribute work across GPUs
        int chunk_size = horizons.size() / num_gpus;
        
        #pragma omp parallel for num_threads(num_gpus)
        for (int gpu = 0; gpu < num_gpus; gpu++) {
            cudaSetDevice(gpu);
            
            int start_idx = gpu * chunk_size;
            int end_idx = (gpu == num_gpus - 1) ? horizons.size() : (gpu + 1) * chunk_size;
            
            // Allocate device memory
            double *d_returns, *d_h_returns, *d_variance, *d_mean;
            cudaMalloc(&d_returns, n * sizeof(double));
            cudaMemcpyAsync(d_returns, returns.data(), n * sizeof(double), 
                           cudaMemcpyHostToDevice, streams[gpu]);
            
            for (int i = start_idx; i < end_idx; i++) {
                int h = horizons[i];
                int h_size = n - h + 1;
                
                cudaMalloc(&d_h_returns, h_size * sizeof(double));
                cudaMalloc(&d_variance, sizeof(double));
                cudaMalloc(&d_mean, sizeof(double));
                
                cudaMemsetAsync(d_variance, 0, sizeof(double), streams[gpu]);
                cudaMemsetAsync(d_mean, 0, sizeof(double), streams[gpu]);
                
                // Compute h-period returns
                int block_size = 256;
                int grid_size = (h_size + block_size - 1) / block_size;
                computeHPeriodReturns<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_returns, d_h_returns, n, h
                );
                
                // Compute variance of 1-period returns
                double var_1;
                computeVarianceWelford<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_returns, d_variance, d_mean, n
                );
                cudaMemcpyAsync(&var_1, d_variance, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                
                // Compute variance of h-period returns
                double var_h;
                cudaMemsetAsync(d_variance, 0, sizeof(double), streams[gpu]);
                cudaMemsetAsync(d_mean, 0, sizeof(double), streams[gpu]);
                computeVarianceWelford<<<grid_size, block_size, 0, streams[gpu]>>>(
                    d_h_returns, d_variance, d_mean, h_size
                );
                cudaMemcpyAsync(&var_h, d_variance, sizeof(double), 
                               cudaMemcpyDeviceToHost, streams[gpu]);
                
                cudaStreamSynchronize(streams[gpu]);
                
                // Calculate variance ratio
                vr_results[i] = var_h / (h * var_1);
                
                cudaFree(d_h_returns);
                cudaFree(d_variance);
                cudaFree(d_mean);
            }
            
            cudaFree(d_returns);
        }
        
        return vr_results;
    }
};

// Extern C interface for Python binding
extern "C" {
    void* create_vr_calculator() {
        return new VarianceRatioMultiGPU();
    }
    
    void destroy_vr_calculator(void* calculator) {
        delete static_cast<VarianceRatioMultiGPU*>(calculator);
    }
    
    void compute_variance_ratios(
        void* calculator,
        const double* returns,
        int n_returns,
        const int* horizons,
        int n_horizons,
        double* results
    ) {
        auto calc = static_cast<VarianceRatioMultiGPU*>(calculator);
        std::vector<double> ret_vec(returns, returns + n_returns);
        std::vector<int> hor_vec(horizons, horizons + n_horizons);
        auto vr_results = calc->computeVR(ret_vec, hor_vec);
        std::copy(vr_results.begin(), vr_results.end(), results);
    }
}
