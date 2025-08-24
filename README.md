# The intuition behind information efficiency metrics 

variance ratios and return autocorrelation decay centers on how quickly markets incorporate new information into prices.

## Core Intuition

Think of a market as an information processing machine. When news hits (a "shock"), an efficient market should rapidly adjust prices to reflect this new information. The speed of this adjustment tells us how informationally efficient the market is.

## Variance Ratio VR(h)

The variance ratio compares price variability over different time horizons. In an efficient market with uncorrelated returns:
- The variance of h-period returns should be exactly h times the variance of 1-period returns
- VR(h) = Var(r_t^h) / (h × Var(r_t)) should equal 1

**Why this matters**: If VR(h) deviates from 1, it suggests returns are predictable:
- VR(h) > 1: Positive autocorrelation (trends/momentum) - shocks get amplified before correcting
- VR(h) < 1: Negative autocorrelation (mean reversion) - prices overshoot then reverse

Markets with VR(h) closer to 1 absorb information more cleanly without over- or under-reaction.

## Return Autocorrelation Decay

This measures how quickly the correlation between returns at different lags drops to zero. In efficient markets:
- Today's return shouldn't predict tomorrow's (autocorrelation ≈ 0)
- Any predictability should decay rapidly

**The decay pattern reveals**: 
- Fast decay → Quick information absorption
- Slow decay → Information gets incorporated gradually (perhaps due to trading frictions, behavioral biases, or institutional constraints)

## Real-World Example

Imagine earnings news hits a stock:
- **Efficient market**: Price jumps immediately to new fair value, subsequent returns uncorrelated
- **Inefficient market**: Price drifts upward over days/weeks as information slowly spreads

The variance ratio would deviate from 1 during the drift period, and autocorrelations would remain significant for multiple lags, both signaling slow information absorption.

These metrics essentially quantify how "sticky" or "smooth" price discovery is - whether information gets priced in one clean jump or through a prolonged adjustment process.

# Mathematical Framework for Information Efficiency

### Variance Ratio VR(h)

The variance ratio tests whether returns follow a random walk (efficient market hypothesis).

**Definition:**
$$VR(h) = \frac{Var(r_t + r_{t-1} + ... + r_{t-h+1})}{h \cdot Var(r_t)}$$

Or equivalently:
$$VR(h) = \frac{Var(r_t^{(h)})}{h \cdot Var(r_t^{(1)})}$$

where $r_t^{(h)}$ is the h-period return.

**Under the null hypothesis** (random walk with uncorrelated increments):
- $VR(h) = 1$ for all horizons h
- Returns are unpredictable

**Relationship to autocorrelations:**
$$VR(h) = 1 + 2\sum_{k=1}^{h-1}\left(1 - \frac{k}{h}\right)\rho(k)$$

where $\rho(k)$ is the k-th order autocorrelation of returns.

**Interpretation:**
- VR(h) > 1: Positive autocorrelation dominates (trending/momentum)
- VR(h) < 1: Negative autocorrelation dominates (mean reversion)
- VR(h) → 1 as h increases: Information gets absorbed

### Return Autocorrelation Decay

**Autocorrelation function (ACF):**
$$\rho(k) = \frac{Cov(r_t, r_{t-k})}{Var(r_t)} = \frac{E[(r_t - \mu)(r_{t-k} - \mu)]}{\sigma^2}$$

**Decay patterns:**

1. **Exponential decay** (AR(1) process):
   $$\rho(k) = \phi^k$$
   where $|\phi| < 1$ is the AR coefficient

2. **Power-law decay** (long memory):
   $$\rho(k) \sim k^{2d-1}$$
   where d is the fractional differencing parameter

**Information absorption speed:**
- **Half-life**: $h_{1/2} = \frac{\log(0.5)}{\log(|\phi|)}$
- Measures lags until autocorrelation drops by half

### Testing Framework

**Lo-MacKinlay VR Test Statistic:**
$$Z(h) = \frac{VR(h) - 1}{\sqrt{V(h)}} \sim N(0,1)$$

where $V(h)$ is the asymptotic variance:
$$V(h) = \frac{2(2h-1)(h-1)}{3hT}$$

for homoskedastic returns, or a heteroskedasticity-robust version.

**Ljung-Box Test for Autocorrelations:**
$$Q(m) = T(T+2)\sum_{k=1}^{m}\frac{\rho^2(k)}{T-k} \sim \chi^2_m$$

Tests joint significance of first m autocorrelations.

### Microstructure Noise Adjustment

In practice, high-frequency returns contain microstructure noise:

**Observed returns:**
$$r_t^{obs} = r_t^{true} + \epsilon_t$$

where $\epsilon_t$ is noise (bid-ask bounce, etc.)

**Bias in VR:**
$$VR_{obs}(h) = VR_{true}(h) + \frac{2\sigma^2_\epsilon}{h \cdot \sigma^2_r}$$

The noise term decreases with h, so longer horizons give cleaner efficiency measures.

### Price Discovery Metrics

**Hasbrouck Information Share:**
Decomposes price discovery across venues/time using vector error correction model:

$$\Delta p_t = \alpha(p_{t-1} - p^*_{t-1}) + \sum_{i=1}^{k}\Gamma_i\Delta p_{t-i} + \epsilon_t$$

**Beveridge-Nelson decomposition:**
$$p_t = p^*_t + z_t$$

where $p^*_t$ is the efficient price (random walk) and $z_t$ is the transitory component.

The speed of $z_t$ mean reversion indicates information efficiency:
$$z_t = \phi z_{t-1} + u_t$$

Smaller $|\phi|$ means faster absorption.

### Practical Implementation

```python
# Variance Ratio
def variance_ratio(returns, h):
    T = len(returns)
    mu = np.mean(returns)
    
    # h-period returns
    r_h = np.sum([returns[i:T-h+i+1] for i in range(h)], axis=0)
    
    # Variances
    var_1 = np.var(returns)
    var_h = np.var(r_h)
    
    VR = var_h / (h * var_1)
    return VR

# Autocorrelation decay
def autocorr_decay(returns, max_lag):
    acf = [np.corrcoef(returns[:-k], returns[k:])[0,1] 
           for k in range(1, max_lag+1)]
    
    # Fit exponential decay
    k = np.arange(1, max_lag+1)
    log_acf = np.log(np.abs(acf))
    phi = np.exp(np.polyfit(k, log_acf, 1)[0])
    
    half_life = np.log(0.5) / np.log(phi)
    return acf, half_life
```

The mathematics reveals that information efficiency is fundamentally about the **temporal dependence structure** of returns - efficient markets should have minimal predictability, leading to VR(h) ≈ 1 and rapidly decaying autocorrelations.












# Project Strucure

project-structure/
├── CMakeLists.txt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── src/
│   ├── cuda/
│   │   ├── variance_ratio.cu
│   │   ├── autocorrelation.cu
│   │   ├── kernels.cuh
│   │   └── multi_gpu_manager.cu
│   ├── simd/
│   │   ├── simd_operations.cpp
│   │   ├── simd_operations.h
│   │   └── avx_kernels.cpp
│   ├── core/
│   │   ├── data_structures.h
│   │   ├── polygon_reader.cpp
│   │   ├── polygon_reader.h
│   │   └── time_utils.h
│   └── bindings/
│       └── python_bindings.cpp
├── python/
│   ├── __init__.py
│   ├── api.py
│   ├── data_loader.py
│   └── visualization.py
├── tests/
│   ├── test_variance_ratio.py
│   ├── test_autocorrelation.py
│   └── benchmark.py
└── config/
    └── config.yaml
