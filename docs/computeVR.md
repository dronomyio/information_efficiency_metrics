# computeVR

a. The horizons [2,5,10,20,50,100,200] are chosen to test the mathematical relationship VR(h) = 1 + 2Σ(1-k/h)ρ(k) across different time scales, where small h values (2,5) detect short-term autocorrelations and mean reversion, medium h values (10,20,50) capture medium-term momentum or trending patterns, and large h values (100,200) identify long-term dependencies - essentially sampling the autocorrelation function ρ(k) at logarithmically spaced intervals to comprehensively test market efficiency across multiple time horizons.

# gridSize

The grid size is calculated as (h_size + block_size - 1) / block_size where h_size = n - h + 1, so for h=2: (100000-2+1 + 255)/256 = 99999/256 = 391 blocks, while for h=5: (100000-5+1 + 255)/256 = 99996/256 = 390 blocks - the different grid sizes reflect the decreasing number of h-period returns that can be computed as h increases.

# OpenMP sync

At the OpenMP synchronization point, VR(2), VR(5), VR(10), VR(20), VR(50), VR(100), and VR(200) are the computed variance ratio values for each horizon - numerical results like VR(10) = 1.000 indicating random walk behavior, VR(2) = 0.650 suggesting mean reversion, or VR(200) = 1.350 indicating long-term momentum, which represent the ratio of h-period return variance to h times the single-period return variance for detecting different market efficiency patterns across time scales.
