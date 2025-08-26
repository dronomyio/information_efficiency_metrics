#pragma once
#include <cuda_runtime.h>

// Device function templates
template<typename T>
__device__ T warpReduce(T val);

template<typename T, int BLOCK_SIZE>
__device__ T blockReduce(T val);

// Kernel declarations
__global__ void computeHPeriodReturns(
    const double* __restrict__ returns, 
    double* __restrict__ h_returns, 
    int n, 
    int h
);

__global__ void computeMeanKernel(
    const double* __restrict__ data,
    double* __restrict__ mean,
    int n
);

__global__ void computeVarianceWelford(
    const double* __restrict__ data,
    double* __restrict__ variance,
    double mean,  // Value, not pointer
    int n
);

__global__ void computeVarianceRatioSmallH(
    const double* __restrict__ returns,
    double* __restrict__ results,
    int n,
    int h
);
