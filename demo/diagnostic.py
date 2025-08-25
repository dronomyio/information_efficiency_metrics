# diagnostic.py - Check if variance ratios are computed correctly
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add build to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'build'))

def manual_variance_ratio(returns, h):
    """Manually compute variance ratio for verification"""
    n = len(returns)
    
    # 1-period returns variance
    var_1 = np.var(returns, ddof=1)
    
    # h-period returns
    h_returns = np.array([np.sum(returns[i:i+h]) for i in range(n-h+1)])
    var_h = np.var(h_returns, ddof=1)
    
    # Variance ratio
    vr = var_h / (h * var_1)
    return vr

# Test with known patterns
np.random.seed(42)
n = 10000

# Test 1: Pure random walk (VR should be ~1)
rw = np.random.randn(n) * 0.01
vr_rw = manual_variance_ratio(rw, 10)
print(f"Random Walk VR(10): {vr_rw:.4f} (expected ~1.0)")

# Test 2: Strong mean reversion (VR should be < 1)
mr = np.zeros(n)
for i in range(1, n):
    mr[i] = -0.5 * mr[i-1] + np.random.randn() * 0.01
vr_mr = manual_variance_ratio(mr, 10)
print(f"Mean Reverting VR(10): {vr_mr:.4f} (expected < 1.0)")

# Test 3: Strong trending (VR should be > 1)
tr = np.zeros(n)
for i in range(1, n):
    tr[i] = 0.5 * tr[i-1] + np.random.randn() * 0.01
vr_tr = manual_variance_ratio(tr, 10)
print(f"Trending VR(10): {vr_tr:.4f} (expected > 1.0)")

# Visual verification
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot price paths
for idx, (name, returns) in enumerate([('Random Walk', rw), ('Mean Reverting', mr), ('Trending', tr)]):
    prices = 100 * np.exp(np.cumsum(returns))
    axes[0, idx].plot(prices, linewidth=0.5)
    axes[0, idx].set_title(f'{name}\nVR(10) = {manual_variance_ratio(returns, 10):.4f}')
    axes[0, idx].set_xlabel('Time')
    axes[0, idx].set_ylabel('Price')
    
    # Plot ACF
    from scipy import signal
    acf = signal.correlate(returns - np.mean(returns), returns - np.mean(returns), mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    axes[1, idx].bar(range(21), acf[:21], alpha=0.7)
    axes[1, idx].set_title(f'ACF(1) = {acf[1]:.4f}')
    axes[1, idx].set_xlabel('Lag')
    axes[1, idx].set_ylabel('ACF')
    axes[1, idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.suptitle('Diagnostic: Variance Ratios and Autocorrelations')
plt.tight_layout()
plt.savefig('diagnostic_vr_acf.png')
print("\n✓ Diagnostic plot saved to diagnostic_vr_acf.png")

# Test your CUDA implementation
print("\n" + "="*50)
print("Testing CUDA Implementation:")

import ctypes
import os

# Load libraries
cuda_path = '/usr/local/cuda/lib64'
ctypes.CDLL(os.path.join(cuda_path, 'libcudart.so'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(cuda_path, 'libcublas.so'), mode=ctypes.RTLD_GLOBAL)

build_dir = Path(__file__).parent.parent / 'build'
cuda_lib = ctypes.CDLL(str(build_dir / 'libinfo_efficiency_cuda.so'))

cuda_lib.create_vr_calculator.restype = ctypes.c_void_p
cuda_lib.compute_variance_ratios.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

calc = cuda_lib.create_vr_calculator()

# Test with mean-reverting data
horizons = np.array([2, 5, 10, 20], dtype=np.int32)
results = np.zeros(4, dtype=np.float64)

cuda_lib.compute_variance_ratios(
    calc,
    mr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    len(mr),
    horizons.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    len(horizons),
    results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

print("CUDA VR results for mean-reverting:")
for h, vr in zip(horizons, results):
    manual_vr = manual_variance_ratio(mr, h)
    print(f"  VR({h:2d}): CUDA={vr:.4f}, Manual={manual_vr:.4f}, Diff={abs(vr-manual_vr):.4f}")

cuda_lib.destroy_vr_calculator(calc)
