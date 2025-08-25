# test_libraries.py - Simple test script
import sys
import os
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

# Try to import the compiled module directly
try:
    import info_efficiency_cpp as cpp
    print("✓ Successfully imported info_efficiency_cpp module")
    
    # Test the functions
    print("\nTesting functions...")
    
    # Test variance ratio calculator
    calc = cpp.cpp_create_vr_calculator()
    print(f"✓ Created VR calculator: {calc}")
    
    # Generate test data
    returns = np.random.randn(1000) * 0.01
    horizons = np.array([2, 5, 10], dtype=np.int32)
    results = np.zeros(3, dtype=np.float64)
    
    cpp.cpp_compute_variance_ratios(
        calc,
        returns,
        horizons,
        results
    )
    
    print(f"✓ Variance Ratios: {dict(zip(horizons, results))}")
    
    cpp.cpp_destroy_vr_calculator(calc)
    
    # Test autocorrelation processor
    proc = cpp.cpp_create_autocorr_processor(4)
    print(f"✓ Created ACF processor: {proc}")
    
    acf_results = np.zeros(11, dtype=np.float64)
    cpp.cpp_compute_autocorrelations(
        proc,
        returns,
        10,
        acf_results
    )
    
    print(f"✓ ACF (first 5 lags): {acf_results[:5]}")
    
    cpp.cpp_destroy_autocorr_processor(proc)
    
    print("\n✓ All tests passed!")
    
except ImportError as e:
    print(f"✗ Failed to import module: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
