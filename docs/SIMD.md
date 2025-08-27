# Full Demo: CUDA GPU + SIMD CPU Processing Flow

## How cuda or simd impl is determined

Based on the CUDA library implementation, the GPU and SIMD path is determined through runtime capability detection and compile-time feature flags that automatically select the optimal execution backend - the system checks available GPUs via cudaGetDeviceCount() for CUDA path selection, while SIMD path determination uses preprocessor macros (__AVX512F__, __AVX2__) to compile the highest available instruction set with fallback to scalar operations.

# why VR is only caculated in cuda and Autocorellation in SIMD

The separation isn't arbitrary - it reflects the computational characteristics and optimal hardware utilization for each algorithm:

**CUDA handles Variance Ratios because:**
- VR computation requires multiple independent variance calculations across different time horizons
- Each horizon (h=2, 5, 10, 20, 50, 100) can be computed in parallel across different GPUs
- The algorithm is embarrassingly parallel - thousands of h-period returns can be computed simultaneously
- GPU's massive parallelism (2560+ cores per RTX 3070) excels at these independent calculations
- Memory bandwidth-intensive operations (reading large return series) benefit from GPU's high-bandwidth memory

**SIMD handles Autocorrelations because:**
- ACF computation is inherently sequential for each lag (lag k depends on completing lag k-1 calculations)
- The algorithm benefits more from vectorization than massive parallelism
- AVX2/AVX-512 can process 4-8 doubles simultaneously, perfect for the dot-product operations in covariance calculations
- CPU's better branch prediction and cache hierarchy suit the more complex control flow
- OpenMP threading provides sufficient parallelism for the workload

**The design choice also reflects practical constraints:**
- GPU context switching overhead would hurt ACF's fine-grained sequential nature
- CPU would be inefficient for VR's massively parallel variance calculations
- This division maximizes utilization of both processing units simultaneously

The implementation could theoretically do both on either platform, but this separation achieves optimal performance by matching algorithmic characteristics to hardware strengths.

## Input Data
- 4 Market Types × 100K Returns Each
- Efficient Market, Mean Reverting (φ=-0.6), Trending (φ=0.6), Volatility Clustering

---

## Parallel Processing Paths

### GPU Path: CUDA Variance Ratios
```
vr_calc = cuda_lib.create_vr_calculator()
└── Initialize 4x RTX 3070 GPUs

GPU Distribution:
├── GPU 0: Horizons [2, 5]
├── GPU 1: Horizons [10, 20]  
├── GPU 2: Horizons [50, 100]
└── GPU 3: Horizon [200]

For Each Market Type:
├── Efficient Market
│   ├── 5 CUDA kernels per horizon
│   ├── computeHPeriodReturns<<<>>>
│   ├── computeMeanKernel<<<>>> (×2)
│   ├── computeVarianceWelford<<<>>> (×2)
│   └── VR = var_h / (h × var_1)
│
├── Mean Reverting (φ=-0.6)
│   ├── Same 5 kernels
│   └── Results: VR < 1.0 (mean reversion)
│
├── Trending (φ=0.6)
│   ├── Same 5 kernels  
│   └── Results: VR > 1.0 (momentum)
│
└── Volatility Clustering
    ├── GARCH pattern
    └── Results: VR ≈ 1.0 (efficient but heteroskedastic)

Performance: 173K-280K points/ms aggregate
Output: VR arrays for dashboard LEFT panel
```

### CPU Path: SIMD Autocorrelations
```
acf_proc = simd_lib.create_autocorr_processor(8)
└── 8 OpenMP threads with AVX2 (4 doubles/operation)
    └── Using 8 of 48 available CPU cores

Thread Distribution:
├── Thread 0: Efficient Market ACF
├── Thread 1: Mean Reverting ACF  
├── Thread 2: Trending ACF
├── Thread 3: Volatility Clustering ACF
└── Threads 4-7: Load balancing/idle

AVX2 SIMD Operations:
├── __m256d data_vec = _mm256_loadu_pd(&data[i])
├── 4 doubles loaded simultaneously
├── __m256d result = _mm256_add_pd(vec1, vec2)  
└── 4 additions in single instruction

For Each Market Type:
├── 1. computeMeanAVX2(returns, n) → μ
├── 2. computeVarianceAVX2(returns, n, μ) → σ²
└── 3. For lag k=1 to 50:
    ├── computeCovarianceSIMD(returns, n, k, μ)
    └── ACF[k] = Cov(r_t, r_{t-k}) / σ²

Results by Market:
├── Efficient Market: ACF ≈ 0 (no correlation)
├── Mean Reverting: ACF[1] < 0 (negative correlation)
├── Trending: ACF[1] > 0 (positive correlation)  
└── Vol Clustering: ACF ≈ 0 but different variance pattern

Performance: 66K-234K points/ms
Output: ACF arrays for dashboard RIGHT panel
```

---

## Dashboard Creation
**Synchronized Results**: GPU Variance Ratios + CPU Autocorrelations
- Left Panel: VR(h) from CUDA GPUs
- Right Panel: ACF(k) from SIMD CPUs

## Key Insights
- **GPU Utilization**: Optimal - all 4 GPUs working on different horizons
- **CPU Utilization**: Suboptimal - only 8/48 cores used  
- **Memory**: GPU memory-bound, CPU could use more threads

- **Performance**: Near-linear scaling across GPUs, underutilized CPU capacity
