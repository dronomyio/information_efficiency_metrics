
# Three distinct types of CUDA functions and their calling hierarchy:

Check the class
https://github.com/dronomyio/information_efficiency_metrics/blob/main/src/cuda/variance_ratio.cu

**__global__ functions** (red): Entry points launched from host CPU with `<<<gridSize, blockSize>>>` syntax. Examples: `computeHPeriodReturns()`, `computeMeanKernel()`. These cannot call other global functions or return to host until completion.

**__device__ functions** (green): GPU utility functions called only by kernels or other device functions. Examples: `blockReduce()`, `warpReduce()`. These enable code reuse and modularity within GPU execution.

**__host__ functions** (blue): CPU functions that manage GPU operations, allocate memory, and launch kernels. These orchestrate the entire computation but cannot directly call device functions.

**Key differences**:
- **Global functions**: GPU entry points with massive parallelism (thousands of threads)
- **Device functions**: GPU utilities for intra-kernel operations 
- **Host functions**: CPU orchestration with single-threaded execution

**Data flow**: Host allocates memory → launches global kernel → kernel processes data using device functions → results copied back to host. The calling restrictions prevent illegal cross-boundary calls (host cannot call device functions, kernels cannot call other kernels).

Your variance ratio implementation follows this pattern: host code distributes work across GPUs, launches reduction kernels, which then call device functions for parallel summation, with results aggregated back on the host.



