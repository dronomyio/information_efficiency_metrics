# computeVR


This visualization shows the complete multi-GPU execution flow where:

**OpenMP creates 4 parallel CPU threads**, each managing one GPU with a chunk of horizons (GPU 0 gets [2,5], GPU 1 gets [10,20], GPU 2 gets [50,100], GPU 3 gets [200]).

**Each GPU processes its horizons sequentially** but **all GPUs work concurrently**. For each horizon h, there are 5 kernel launches: computeHPeriodReturns, computeMeanKernel (twice), and computeVarianceWelford (twice), followed by host-side VR calculation.

**Load imbalance occurs** because GPU 3 has only 1 horizon while others have 2, and larger horizons require more computation. The function waits for all GPUs to complete before returning the complete variance ratio vector.

**Key performance aspects**: 35 total kernel launches across 4 GPUs, peak memory usage under 2MB per GPU, and memory bandwidth as the primary bottleneck rather than compute capacity, achieving near-linear 95% scaling efficiency.


a. The horizons [2,5,10,20,50,100,200] are chosen to test the mathematical relationship VR(h) = 1 + 2Σ(1-k/h)ρ(k) across different time scales, where small h values (2,5) detect short-term autocorrelations and mean reversion, medium h values (10,20,50) capture medium-term momentum or trending patterns, and large h values (100,200) identify long-term dependencies - essentially sampling the autocorrelation function ρ(k) at logarithmically spaced intervals to comprehensively test market efficiency across multiple time horizons.

#chunkSize

The distribution formula chunk_size = (7 + 4 - 1) / 4 = 2 assigns horizons by GPU index ranges (GPU 0: horizons[0:2], GPU 1: horizons[2:4], etc.), and the different h values [2,5,10,20,50,100,200] represent various time windows for computing variance ratios - testing market efficiency across multiple time scales from 2-period (short-term) to 200-period (long-term) returns to detect different types of market inefficiencies or patterns.

# gridSize

The grid size is calculated as (h_size + block_size - 1) / block_size where h_size = n - h + 1, so for h=2: (100000-2+1 + 255)/256 = 99999/256 = 391 blocks, while for h=5: (100000-5+1 + 255)/256 = 99996/256 = 390 blocks - the different grid sizes reflect the decreasing number of h-period returns that can be computed as h increases.

# kernels

The mathematical relationship VR(h) = 1 + 2Σ(1-k/h)ρ(k) and autocorrelation theory are not explicitly implemented in the code - the implementation uses the direct computational definition VR(h) = Var(r_t^(h)) / (h × Var(r_t^(1))) through kernel calculations, while the autocorrelation relationship serves as the theoretical foundation explaining why different VR(h) values indicate trending (VR > 1) or mean reversion (VR < 1) patterns, but the code doesn't compute the individual ρ(k) autocorrelation coefficients that would be needed for the mathematical formula.

# OpenMP sync

At the OpenMP synchronization point, VR(2), VR(5), VR(10), VR(20), VR(50), VR(100), and VR(200) are the computed variance ratio values for each horizon - numerical results like VR(10) = 1.000 indicating random walk behavior, VR(2) = 0.650 suggesting mean reversion, or VR(200) = 1.350 indicating long-term momentum, which represent the ratio of h-period return variance to h times the single-period return variance for detecting different market efficiency patterns across time scales.

# math of VR

The mathematical relationship VR(h) = 1 + 2Σ(1-k/h)ρ(k) and autocorrelation theory are not explicitly implemented in the code - the implementation uses the direct computational definition VR(h) = Var(r_t^(h)) / (h × Var(r_t^(1))) through kernel calculations, while the autocorrelation relationship serves as the theoretical foundation explaining why different VR(h) values indicate trending (VR > 1) or mean reversion (VR < 1) patterns, but the code doesn't compute the individual ρ(k) autocorrelation coefficients that would be needed for the mathematical formula.

#auto correlation

No, implementing the autocorrelation relationship is not necessary for your current variance ratio implementation - the direct computational approach VR(h) = Var(r_t^(h)) / (h × Var(r_t^(1))) already captures all the statistical information needed and is computationally more efficient than calculating individual autocorrelation coefficients ρ(k) and then applying the theoretical formula, though adding autocorrelation computation could provide additional insights into the underlying market microstructure patterns if desired for deeper analysis.
