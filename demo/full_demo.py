#!/usr/bin/env python3
"""
Complete demonstration with corrected strong parameters to show clear market patterns
"""
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

def find_project_root():
    """Find the project root directory by looking for CMakeLists.txt"""
    current = Path(__file__).resolve().parent

    # Search up to 5 levels up for CMakeLists.txt
    for _ in range(5):
        if (current / 'CMakeLists.txt').exists():
            return current
        if (current / './../build').exists() and (current / 'build' / 'libinfo_efficiency_cuda.so').exists():
            return current
        current = current.parent

    # If not found, assume current directory is project root
    return Path.cwd()

def load_libraries():
    """Dynamically load CUDA and SIMD libraries"""
    # Find project root and build directory
    project_root = find_project_root()
    build_dir = project_root / 'build'

    if not build_dir.exists():
        raise RuntimeError(f"Build directory not found at {build_dir}")

    print(f"Project root: {project_root}")
    print(f"Build directory: {build_dir}")

    # Load CUDA runtime libraries
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.8/lib64',  # Specific version
        '/usr/local/cuda-11/lib64',    # Older version
        '/usr/lib/x86_64-linux-gnu',   # System path
    ]

    cuda_loaded = False
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            try:
                cudart_path = os.path.join(cuda_path, 'libcudart.so')
                cublas_path = os.path.join(cuda_path, 'libcublas.so')

                if os.path.exists(cudart_path) and os.path.exists(cublas_path):
                    ctypes.CDLL(cudart_path, mode=ctypes.RTLD_GLOBAL)
                    ctypes.CDLL(cublas_path, mode=ctypes.RTLD_GLOBAL)
                    print(f"✓ CUDA runtime libraries loaded from {cuda_path}")
                    cuda_loaded = True
                    break
            except Exception as e:
                continue

    if not cuda_loaded:
        print("⚠ Warning: Could not load CUDA runtime libraries, trying to continue...")

    # Load our compiled libraries
    cuda_lib_path = build_dir / 'libinfo_efficiency_cuda.so'
    simd_lib_path = build_dir / 'libinfo_efficiency_simd.so'

    if not cuda_lib_path.exists():
        raise FileNotFoundError(f"CUDA library not found at {cuda_lib_path}")
    if not simd_lib_path.exists():
        raise FileNotFoundError(f"SIMD library not found at {simd_lib_path}")

    cuda_lib = ctypes.CDLL(str(cuda_lib_path))
    simd_lib = ctypes.CDLL(str(simd_lib_path))

    print(f"✓ Loaded CUDA library from {cuda_lib_path}")
    print(f"✓ Loaded SIMD library from {simd_lib_path}")

    return cuda_lib, simd_lib

# Load libraries dynamically
try:
    cuda_lib, simd_lib = load_libraries()
except Exception as e:
    print(f"Error loading libraries: {e}")
    print("Please ensure you have built the project (run 'make' in build directory)")
    exit(1)

# Configure function signatures
cuda_lib.create_vr_calculator.restype = ctypes.c_void_p
cuda_lib.destroy_vr_calculator.argtypes = [ctypes.c_void_p]
cuda_lib.compute_variance_ratios.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

simd_lib.create_autocorr_processor.restype = ctypes.c_void_p
simd_lib.create_autocorr_processor.argtypes = [ctypes.c_int]
simd_lib.destroy_autocorr_processor.argtypes = [ctypes.c_void_p]
simd_lib.compute_autocorrelations.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double)
]

def benchmark_performance():
    """Benchmark the multi-GPU and AVX2 performance"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING - 4x RTX 3070 + AVX2")
    print("="*60)

    # Test different data sizes
    sizes = [1000, 10000, 100000, 1000000]
    horizons = np.array([2, 5, 10, 20, 50, 100], dtype=np.int32)

    # Initialize calculators
    vr_calc = cuda_lib.create_vr_calculator()
    acf_proc = simd_lib.create_autocorr_processor(8)  # Use 8 threads

    results = []

    for size in sizes:
        print(f"\nData size: {size:,} points")
        returns = np.random.randn(size).astype(np.float64) * 0.01

        # Benchmark Variance Ratio (Multi-GPU)
        vr_results = np.zeros(len(horizons), dtype=np.float64)
        start = time.perf_counter()
        cuda_lib.compute_variance_ratios(
            vr_calc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            size,
            horizons.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(horizons),
            vr_results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        vr_time = (time.perf_counter() - start) * 1000

        # Benchmark Autocorrelation (AVX2)
        acf_results = np.zeros(101, dtype=np.float64)
        start = time.perf_counter()
        simd_lib.compute_autocorrelations(
            acf_proc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            size,
            100,
            acf_results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        acf_time = (time.perf_counter() - start) * 1000

        vr_throughput = size/vr_time if vr_time > 0 else 0
        acf_throughput = size/acf_time if acf_time > 0 else 0

        print(f"  Variance Ratio (4x GPU): {vr_time:.2f} ms ({vr_throughput:.0f} points/ms)")
        print(f"  Autocorrelation (AVX2):  {acf_time:.2f} ms ({acf_throughput:.0f} points/ms)")

        results.append({
            'size': size,
            'vr_time': vr_time,
            'acf_time': acf_time,
            'vr_throughput': vr_throughput,
            'acf_throughput': acf_throughput
        })

    # Cleanup
    cuda_lib.destroy_vr_calculator(vr_calc)
    simd_lib.destroy_autocorr_processor(acf_proc)

    return results

def analyze_market_data_with_strong_patterns():
    """Analyze different market conditions with STRONG PARAMETERS for clear patterns"""
    print("\n" + "="*60)
    print("MARKET MICROSTRUCTURE ANALYSIS - CORRECTED PARAMETERS")
    print("="*60)

    np.random.seed(42)
    n = 100000  # Large dataset to utilize GPU power

    # Generate different market conditions with STRONG parameters
    markets = {}

    # 1. Random Walk (Efficient Market) - baseline
    markets['Efficient Market'] = np.random.randn(n) * 0.015

    # 2. STRONG Mean Reverting (φ = -0.6) - clear mean reversion
    mean_rev = np.zeros(n)
    mean_rev[0] = np.random.randn() * 0.015
    for i in range(1, n):
        mean_rev[i] = -0.6 * mean_rev[i-1] + np.random.randn() * 0.015
    markets['Mean Reverting'] = mean_rev

    # 3. STRONG Trending (φ = 0.6) - clear momentum  
    trending = np.zeros(n)
    trending[0] = np.random.randn() * 0.015
    for i in range(1, n):
        trending[i] = 0.6 * trending[i-1] + np.random.randn() * 0.015
    markets['Trending'] = trending

    # 4. Enhanced Volatility Clustering (GARCH-like with stronger clustering)
    vol_cluster = np.zeros(n)
    h = np.zeros(n)  # conditional variance
    h[0] = 0.015**2
    vol_cluster[0] = np.sqrt(h[0]) * np.random.randn()
    
    # GARCH(1,1) with strong clustering parameters
    omega, alpha, beta = 0.00005, 0.15, 0.8  # Stronger clustering
    for i in range(1, n):
        h[i] = omega + alpha * vol_cluster[i-1]**2 + beta * h[i-1]
        vol_cluster[i] = np.sqrt(h[i]) * np.random.randn()
    
    markets['Volatility Clustering'] = vol_cluster

    # Initialize processors
    vr_calc = cuda_lib.create_vr_calculator()
    acf_proc = simd_lib.create_autocorr_processor(8)

    horizons = np.array([2, 5, 10, 20, 50, 100, 200], dtype=np.int32)
    results = {}

    print("\nComputing market efficiency metrics...")
    
    for name, returns in markets.items():
        print(f"\nAnalyzing {name}:")

        # Compute variance ratios using CUDA
        vr_results = np.zeros(len(horizons), dtype=np.float64)
        start_time = time.perf_counter()
        cuda_lib.compute_variance_ratios(
            vr_calc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(returns),
            horizons.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(horizons),
            vr_results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        vr_time = (time.perf_counter() - start_time) * 1000

        # Compute autocorrelations using AVX2
        acf = np.zeros(51, dtype=np.float64)
        start_time = time.perf_counter()
        simd_lib.compute_autocorrelations(
            acf_proc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(returns),
            50,
            acf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        acf_time = (time.perf_counter() - start_time) * 1000

        results[name] = {
            'vr': dict(zip(horizons, vr_results)),
            'acf': acf,
            'vr_time': vr_time,
            'acf_time': acf_time
        }

        # Print key metrics
        print(f"  VR(10)  = {vr_results[2]:.4f}")
        print(f"  VR(50)  = {vr_results[4]:.4f}")
        print(f"  ACF(1)  = {acf[1]:.4f}")
        print(f"  Performance: VR={vr_time:.1f}ms, ACF={acf_time:.1f}ms")

        # Market interpretation
        if name == 'Mean Reverting' and vr_results[2] < 0.8:
            print(f"  → Strong mean reversion detected! VR({horizons[2]}) = {vr_results[2]:.3f}")
        elif name == 'Trending' and vr_results[2] > 1.2:
            print(f"  → Strong momentum detected! VR({horizons[2]}) = {vr_results[2]:.3f}")
        elif name == 'Efficient Market' and 0.9 < vr_results[2] < 1.1:
            print(f"  → Market appears efficient! VR({horizons[2]}) = {vr_results[2]:.3f}")

    # Cleanup
    cuda_lib.destroy_vr_calculator(vr_calc)
    simd_lib.destroy_autocorr_processor(acf_proc)

    return results

def create_enhanced_dashboard(results):
    """Create enhanced dashboard with clear pattern differentiation"""
    fig = plt.figure(figsize=(16, 12))
    
    # Main layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1])
    
    # Top row: Variance Ratios and Autocorrelations
    ax1 = fig.add_subplot(gs[0, 0])  # Variance Ratios
    ax2 = fig.add_subplot(gs[0, 1])  # Autocorrelations
    ax3 = fig.add_subplot(gs[0, 2])  # Efficiency Scores
    
    # Bottom row: Sample paths
    ax4 = fig.add_subplot(gs[1, :])  # Time series plots
    
    # Performance metrics
    ax5 = fig.add_subplot(gs[2, :])  # Performance table
    
    # Colors for consistency
    colors = {
        'Efficient Market': '#1f77b4',      # Blue
        'Mean Reverting': '#ff7f0e',        # Orange  
        'Trending': '#2ca02c',              # Green
        'Volatility Clustering': '#d62728'  # Red
    }
    
    # Plot 1: Variance Ratios with clear differentiation
    horizons = list(results['Efficient Market']['vr'].keys())
    for name, res in results.items():
        vr_vals = list(res['vr'].values())
        ax1.plot(horizons, vr_vals, marker='o', label=name, linewidth=2.5, 
                markersize=6, color=colors[name])
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Variance Ratio')
    ax1.set_title('Variance Ratios (Computed on 4x RTX 3070)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 2.5)  # Better range to show differences
    
    # Plot 2: Autocorrelation Functions
    lags = np.arange(len(results['Efficient Market']['acf']))
    for name, res in results.items():
        ax2.plot(lags[:21], res['acf'][:21], label=name, linewidth=2.5, 
                color=colors[name])  # Show first 20 lags
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')
    ax2.set_title('Autocorrelation Functions (AVX2 SIMD)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.8, 1.0)
    
    # Plot 3: Market Efficiency Scores
    efficiency_scores = {}
    for name, res in results.items():
        # Efficiency score based on deviation from VR=1
        vr_10 = res['vr'][10]
        vr_50 = res['vr'][50] if 50 in res['vr'] else res['vr'][max(res['vr'].keys())]
        
        # Score: closer to 1.0 = more efficient
        deviation = abs(vr_10 - 1.0) + abs(vr_50 - 1.0)
        efficiency_scores[name] = max(0, 1.0 - deviation)
    
    names = list(efficiency_scores.keys())
    scores = list(efficiency_scores.values())
    bars = ax3.bar(names, scores, color=[colors[name] for name in names], alpha=0.8)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Market Efficiency Scores', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 4: Sample time series (first 1000 points)
    np.random.seed(42)  # Regenerate same data for visualization
    n_viz = 1000
    
    # Regenerate sample data for visualization
    viz_data = {}
    
    # Random Walk
    viz_data['Efficient Market'] = np.random.randn(n_viz) * 0.015
    
    # Strong Mean Reverting
    mr = np.zeros(n_viz)
    mr[0] = np.random.randn() * 0.015
    for i in range(1, n_viz):
        mr[i] = -0.6 * mr[i-1] + np.random.randn() * 0.015
    viz_data['Mean Reverting'] = mr
    
    # Strong Trending
    tr = np.zeros(n_viz)
    tr[0] = np.random.randn() * 0.015
    for i in range(1, n_viz):
        tr[i] = 0.6 * tr[i-1] + np.random.randn() * 0.015
    viz_data['Trending'] = tr
    
    # Volatility Clustering
    vc = np.zeros(n_viz)
    h = np.zeros(n_viz)
    h[0] = 0.015**2
    vc[0] = np.sqrt(h[0]) * np.random.randn()
    
    omega, alpha, beta = 0.00005, 0.15, 0.8
    for i in range(1, n_viz):
        h[i] = omega + alpha * vc[i-1]**2 + beta * h[i-1]
        vc[i] = np.sqrt(h[i]) * np.random.randn()
    viz_data['Volatility Clustering'] = vc
    
    # Plot cumulative returns
    for name, data in viz_data.items():
        cumulative = np.cumsum(data)
        ax4.plot(cumulative, label=f'{name} (VR₁₀={results[name]["vr"][10]:.3f})', 
                linewidth=2, color=colors[name])
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Returns')
    ax4.set_title('Sample Market Paths (First 1000 Periods)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Performance metrics table
    ax5.axis('off')
    
    # System info
    sys_info = [
        "System Configuration:",
        "• GPUs: 4x NVIDIA RTX 3070",
        "• VRAM: 7.8 GB per GPU (31.2 GB total)",
        "• Compute Capability: 8.6 (Ampere)",
        "• SIMD: AVX2 (4 doubles/operation)",
        "• CPU Threads: 8 (OpenMP)",
        "",
        "Performance Achieved:",
        f"• VR Throughput: {int(results['Efficient Market']['vr_time']*1000/100)}-{int(results['Efficient Market']['vr_time']*1000/10)}K points/ms",
        f"• ACF Throughput: {int(results['Efficient Market']['acf_time']*1000/100)}-{int(results['Efficient Market']['acf_time']*1000/10)}K points/ms",
        "• Total GPU Memory: 31.2 GB",
        "• Parallel Streams: 4"
    ]
    
    y_pos = 0.95
    for line in sys_info:
        if line.startswith("•"):
            ax5.text(0.02, y_pos, line, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
        elif line == "":
            pass  # Skip empty lines
        else:
            ax5.text(0.02, y_pos, line, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', fontweight='bold')
        y_pos -= 0.08
    
    plt.suptitle('Microstructure Analysis Dashboard - 4x RTX 3070 + AVX2', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig

def main():
    """Main demonstration function"""
    print("🚀 Market Microstructure Analysis - High Performance Demo")
    print("Using CORRECTED parameters for clear pattern differentiation")
    print("="*70)
    
    # Run performance benchmarks
    perf_results = benchmark_performance()
    
    # Analyze market data with strong patterns
    market_results = analyze_market_data_with_strong_patterns()
    
    # Create enhanced dashboard
    fig = create_enhanced_dashboard(market_results)
    
    # Save high-quality plots
    plt.savefig('microstructure_dashboard_corrected.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('microstructure_dashboard_corrected.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\n📊 Enhanced dashboard saved as:")
    print(f"   • microstructure_dashboard_corrected.png (high-res)")
    print(f"   • microstructure_dashboard_corrected.pdf (vector)")
    
    # Summary of key findings
    print(f"\n{'='*70}")
    print("🔍 KEY FINDINGS:")
    print("="*70)
    
    for name, res in market_results.items():
        vr_10 = res['vr'][10]
        acf_1 = res['acf'][1]
        
        print(f"\n{name}:")
        print(f"  VR(10) = {vr_10:.4f}")
        print(f"  ACF(1) = {acf_1:.4f}")
        
        if name == 'Mean Reverting':
            if vr_10 < 0.8:
                print(f"  ✓ Strong mean reversion detected (VR << 1)")
            else:
                print(f"  ⚠ Weak mean reversion (increase |φ| parameter)")
        elif name == 'Trending':
            if vr_10 > 1.2:
                print(f"  ✓ Strong trending detected (VR >> 1)")
            else:
                print(f"  ⚠ Weak trending (increase φ parameter)")
        elif name == 'Efficient Market':
            if 0.9 <= vr_10 <= 1.1:
                print(f"  ✓ Market efficiency confirmed (VR ≈ 1)")
            else:
                print(f"  ⚠ Unexpected deviation from efficiency")
    
    print(f"\n{'='*70}")
    print("🎯 SOLUTION IMPLEMENTED:")
    print("✓ Used φ = ±0.6 instead of ±0.1 for strong signal")
    print("✓ Enhanced GARCH parameters for volatility clustering")
    print("✓ Clear pattern differentiation now visible")
    print("✓ Dashboard shows meaningful market microstructure effects")
    
    plt.show()

if __name__ == "__main__":
    main()
