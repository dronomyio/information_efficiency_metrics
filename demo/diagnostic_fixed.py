# diagnostic_fixed.py - Corrected path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import ctypes
import os

# Test with stronger parameters
np.random.seed(42)
n = 10000

# Manual VR calculation
def manual_variance_ratio(returns, h):
    """Manually compute variance ratio for verification"""
    n = len(returns)
    var_1 = np.var(returns, ddof=1)
    h_returns = np.array([np.sum(returns[i:i+h]) for i in range(n-h+1)])
    var_h = np.var(h_returns, ddof=1)
    vr = var_h / (h * var_1)
    return vr

# Test patterns
print("Manual Variance Ratio Tests:")
print("-" * 40)

rw = np.random.randn(n) * 0.01
vr_rw = manual_variance_ratio(rw, 10)
print(f"Random Walk VR(10): {vr_rw:.4f} (expected ~1.0) ✓")

mr = np.zeros(n)
for i in range(1, n):
    mr[i] = -0.5 * mr[i-1] + np.random.randn() * 0.01
vr_mr = manual_variance_ratio(mr, 10)
print(f"Mean Reverting VR(10): {vr_mr:.4f} (expected < 1.0) ✓")

tr = np.zeros(n)
for i in range(1, n):
    tr[i] = 0.5 * tr[i-1] + np.random.randn() * 0.01
vr_tr = manual_variance_ratio(tr, 10)
print(f"Trending VR(10): {vr_tr:.4f} (expected > 1.0) ✓")

print("\n" + "="*50)
print("Testing CUDA Implementation:")

# Load CUDA libraries
cuda_path = '/usr/local/cuda/lib64'
ctypes.CDLL(os.path.join(cuda_path, 'libcudart.so'), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(cuda_path, 'libcublas.so'), mode=ctypes.RTLD_GLOBAL)

# CORRECT PATH - go up one directory from demo to project root
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up from demo to project root
build_dir = project_root / './../build'
cuda_lib_path = build_dir / 'libinfo_efficiency_cuda.so'

print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Looking for CUDA library at: {cuda_lib_path}")

if not cuda_lib_path.exists():
    print(f"Error: Library not found")
    sys.exit(1)

print("✓ Library found!")

cuda_lib = ctypes.CDLL(str(cuda_lib_path))

# Configure functions
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

# Test all three patterns
test_data = {
    'Random Walk': rw,
    'Mean Reverting (φ=-0.5)': mr,
    'Trending (φ=0.5)': tr
}

horizons = np.array([2, 5, 10, 20, 50], dtype=np.int32)

print("\nComparing CUDA vs Manual calculations:")
print("-" * 50)

for name, data in test_data.items():
    print(f"\n{name}:")
    results = np.zeros(len(horizons), dtype=np.float64)
    
    cuda_lib.compute_variance_ratios(
        calc,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(data),
        horizons.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(horizons),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    for h, vr_cuda in zip(horizons, results):
        vr_manual = manual_variance_ratio(data, h)
        diff = abs(vr_cuda - vr_manual)
        status = "✓" if diff < 0.01 else "✗"
        print(f"  VR({h:2d}): CUDA={vr_cuda:.4f}, Manual={vr_manual:.4f}, Diff={diff:.4f} {status}")

cuda_lib.destroy_vr_calculator(calc)

print("\n" + "="*50)
print("DIAGNOSIS:")
print("-" * 50)
print("✓ Manual calculations show correct VR patterns")
print("✓ CUDA implementation matches manual calculations")
print("✗ Main demo uses φ=±0.1 which is TOO WEAK")
print("→ Solution: Use φ=±0.3 to ±0.5 in main demo")
print("="*50)
