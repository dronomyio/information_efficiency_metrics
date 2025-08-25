#pragma once

#include <cuda_runtime.h>

template<typename T>
__device__ T warpReduce(T val);

template<typename T, int BLOCK_SIZE>
__device__ T blockReduce(T val);

__global__ void computeHPeriodReturns(const double* returns, double* h_returns, int n, int h);
__global__ void computeVarianceWelford(const double* data, double* variance, double* mean, int n);
