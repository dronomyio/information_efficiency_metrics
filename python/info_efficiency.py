# demo_microstructure_analysis.py
"""
Demonstration script for Information Efficiency Microstructure Analysis
Shows how to use both the simple ctypes wrapper and the full API
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
try:
    from python.info_efficiency import (
        MicrostructureAnalyzer,
        VarianceRatioCalculator,
        AutocorrelationProcessor,
        PolygonDataReader
    )
    print("✓ Successfully imported info_efficiency module")
except ImportError as e:
    print(f"✗ Failed to import info_efficiency: {e}")
    print("Please ensure the C++ libraries are built (run 'make' in build directory)")
    sys.exit(1)

# Set up plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)

def generate_synthetic_data(n_points=10000, model='random_walk'):
    """
    Generate synthetic return data for testing
    
    Args:
        n_points: Number of data points
        model: Type of model ('random_walk', 'mean_reverting', 'trending')
    
    Returns:
        Array of returns
    """
    np.random.seed(42)
    
    if model == 'random_walk':
        # Pure random walk (VR should be close to 1)
        returns = np.random.randn(n_points) * 0.01
        
    elif model == 'mean_reverting':
        # AR(1) process with negative autocorrelation (VR < 1)
        returns = np.zeros(n_points)
        phi = -0.1  # Mean reversion parameter
        for i in range(1, n_points):
            returns[i] = phi * returns[i-1] + np.random.randn() * 0.01
            
    elif model == 'trending':
        # AR(1) process with positive autocorrelation (VR > 1)
        returns = np.zeros(n_points)
        phi = 0.1  # Momentum parameter
        for i in range(1, n_points):
            returns[i] = phi * returns[i-1] + np.random.randn() * 0.01
            
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return returns

def demo_variance_ratio():
    """Demonstrate variance ratio calculation"""
    print("\n" + "="*60)
    print("VARIANCE RATIO DEMONSTRATION")
    print("="*60)
    
    # Initialize calculator
    vr_calc = VarianceRatioCalculator()
    
    # Test different market conditions
    models = ['random_walk', 'mean_reverting', 'trending']
    horizons = [2, 5, 10, 20, 50, 100]
    
    results = {}
    
    for model in models:
        print(f"\n{model.upper().replace('_', ' ')} Model:")
        returns = generate_synthetic_data(10000, model)
        
        # Time the computation
        start_time = time.time()
        vr_results = vr_calc.compute(returns, horizons)
        elapsed_time = time.time() - start_time
        
        results[model] = vr_results
        
        print(f"  Computation time: {elapsed_time*1000:.2f} ms")
        print("  Variance Ratios:")
        for h, vr in vr_results.items():
            deviation = (vr - 1) * 100
            print(f"    VR({h:3d}) = {vr:.4f} ({deviation:+.2f}% from 1)")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model, vr_dict in results.items():
        horizons_list = list(vr_dict.keys())
        vr_values = list(vr_dict.values())
        ax.plot(horizons_list, vr_values, marker='o', label=model.replace('_', ' ').title())
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Random Walk (VR=1)')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Ratios for Different Market Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_ratios_demo.png', dpi=150)
    print(f"\n✓ Plot saved as 'variance_ratios_demo.png'")

def demo_autocorrelation():
    """Demonstrate autocorrelation analysis"""
    print("\n" + "="*60)
    print("AUTOCORRELATION DEMONSTRATION")
    print("="*60)
    
    # Initialize processor
    acf_proc = AutocorrelationProcessor(num_threads=4)
    
    # Generate data with known autocorrelation structure
    n = 10000
    returns = np.zeros(n)
    
    # AR(2) process: r_t = 0.3*r_{t-1} - 0.2*r_{t-2} + epsilon
    for i in range(2, n):
        returns[i] = 0.3 * returns[i-1] - 0.2 * returns[i-2] + np.random.randn() * 0.01
    
    # Compute ACF
    max_lag = 50
    start_time = time.time()
    acf = acf_proc.compute(returns, max_lag)
    elapsed_time = time.time() - start_time
    
    print(f"  Computation time: {elapsed_time*1000:.2f} ms")
    
    # Compute decay parameters
    phi, half_life = acf_proc.compute_decay_parameters(acf)
    print(f"  Decay parameter (φ): {phi:.4f}")
    print(f"  Half-life: {half_life:.2f} periods")
    
    # Compute PACF
    pacf = acf_proc.compute_partial_acf(returns, 20)
    
    # Plot ACF and PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF plot
    axes[0].bar(range(len(acf)), acf, color='steelblue', alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('Autocorrelation Function')
    axes[0].grid(True, alpha=0.3)
    
    # PACF plot
    axes[1].bar(range(len(pacf)), pacf, color='coral', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].set_title('Partial Autocorrelation Function')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autocorrelation_demo.png', dpi=150)
    print(f"\n✓ Plots saved as 'autocorrelation_demo.png'")

def demo_full_analysis():
    """Demonstrate complete microstructure analysis"""
    print("\n" + "="*60)
    print("COMPLETE MICROSTRUCTURE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer (without Polygon API key for demo)
    analyzer = MicrostructureAnalyzer(num_threads=8)
    
    # Generate realistic intraday returns (5-minute returns for a trading day)
    n_periods = 78  # 6.5 hours * 12 periods per hour
    timestamps = pd.date_range(start='2024-01-15 09:30:00', periods=n_periods, freq='5min')
    
    # Simulate intraday pattern with volatility clustering
    returns = []
    volatility = 0.001
    
    for i in range(n_periods):
        # U-shaped intraday volatility
        time_of_day = i / n_periods
        intraday_factor = 1 + 2 * (time_of_day - 0.5) ** 2
        
        # GARCH-like volatility clustering
        if i > 0 and abs(returns[-1]) > 2 * volatility:
            volatility *= 1.2
        else:
            volatility *= 0.95
        volatility = max(0.0005, min(0.003, volatility))
        
        # Generate return
        ret = np.random.randn() * volatility * intraday_factor
        returns.append(ret)
    
    returns = np.array(returns)
    
    # Perform analysis
    print("\nAnalyzing intraday returns...")
    start_time = time.time()
    results = analyzer.analyze(returns, horizons=[2, 5, 10, 15, 20, 30], max_lag=30)
    elapsed_time = time.time() - start_time
    
    print(f"Total analysis time: {elapsed_time*1000:.2f} ms")
    
    # Display results
    print("\n📊 SUMMARY STATISTICS:")
    for key, value in results['summary'].items():
        print(f"  {key:20s}: {value:12.6f}")
    
    print("\n📈 VARIANCE RATIOS:")
    for h, vr in results['variance_ratios'].items():
        print(f"  VR({h:2d}) = {vr:.4f}")
    
    print(f"\n🔄 AUTOCORRELATION DECAY:")
    print(f"  Decay parameter (φ): {results['decay_phi']:.4f}")
    print(f"  Half-life: {results['decay_half_life']:.2f} periods")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Price path
    ax1 = plt.subplot(2, 3, 1)
    prices = 100 * np.exp(np.cumsum(returns))
    ax1.plot(timestamps, prices, color='navy', linewidth=0.8)
    ax1.set_title('Simulated Price Path')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(timestamps, returns * 100, color='darkgreen', linewidth=0.5)
    ax2.set_title('Returns (%)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Variance Ratios
    ax3 = plt.subplot(2, 3, 3)
    horizons = list(results['variance_ratios'].keys())
    vr_values = list(results['variance_ratios'].values())
    ax3.plot(horizons, vr_values, marker='o', color='crimson', markersize=8)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Variance Ratios')
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('VR')
    ax3.grid(True, alpha=0.3)
    
    # 4. ACF
    ax4 = plt.subplot(2, 3, 4)
    acf_values = results['autocorrelation'][:31]
    ax4.bar(range(len(acf_values)), acf_values, color='steelblue', alpha=0.7)
    ax4.set_title('Autocorrelation Function')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('ACF')
    ax4.grid(True, alpha=0.3)
    
    # 5. PACF
    ax5 = plt.subplot(2, 3, 5)
    pacf_values = results['partial_acf']
    ax5.bar(range(len(pacf_values)), pacf_values, color='coral', alpha=0.7)
    ax5.set_title('Partial Autocorrelation Function')
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('PACF')
    ax5.grid(True, alpha=0.3)
    
    # 6. Return distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(returns * 100, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax6.set_title('Return Distribution')
    ax6.set_xlabel('Return (%)')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Microstructure Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('microstructure_analysis_dashboard.png', dpi=150)
    print(f"\n✓ Dashboard saved as 'microstructure_analysis_dashboard.png'")
    
    # Save results to JSON
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'num_observations': len(returns),
        'variance_ratios': results['variance_ratios'],
        'decay_parameters': {
            'phi': results['decay_phi'],
            'half_life': results['decay_half_life']
        },
        'summary_stats': results['summary'],
        'first_10_acf': results['autocorrelation'][:10]
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print("✓ Results saved to 'analysis_results.json'")

def demo_performance_comparison():
    """Compare performance across different data sizes"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    vr_calc = VarianceRatioCalculator()
    acf_proc = AutocorrelationProcessor(num_threads=8)
    
    data_sizes = [100, 1000, 10000, 100000]
    horizons = [2, 5, 10, 20, 50]
    max_lag = 100
    
    results = []
    
    for size in data_sizes:
        print(f"\nData size: {size:,} points")
        returns = np.random.randn(size) * 0.01
        
        # Variance Ratio timing
        start = time.time()
        vr_calc.compute(returns, horizons)
        vr_time = (time.time() - start) * 1000
        
        # Autocorrelation timing
        start = time.time()
        acf_proc.compute(returns, min(max_lag, size-1))
        acf_time = (time.time() - start) * 1000
        
        results.append({
            'size': size,
            'vr_time_ms': vr_time,
            'acf_time_ms': acf_time,
            'vr_throughput': size / vr_time if vr_time > 0 else 0,
            'acf_throughput': size / acf_time if acf_time > 0 else 0
        })
        
        print(f"  VR computation: {vr_time:.2f} ms ({size/vr_time:.0f} points/ms)")
        print(f"  ACF computation: {acf_time:.2f} ms ({size/acf_time:.0f} points/ms)")
    
    # Plot performance scaling
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Computation time
    axes[0].loglog(df['size'], df['vr_time_ms'], marker='o', label='Variance Ratio')
    axes[0].loglog(df['size'], df['acf_time_ms'], marker='s', label='Autocorrelation')
    axes[0].set_xlabel('Data Size')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Computation Time Scaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Throughput
    axes[1].semilogx(df['size'], df['vr_throughput'], marker='o', label='Variance Ratio')
    axes[1].semilogx(df['size'], df['acf_throughput'], marker='s', label='Autocorrelation')
    axes[1].set_xlabel('Data Size')
    axes[1].set_ylabel('Throughput (points/ms)')
    axes[1].set_title('Processing Throughput')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=150)
    print(f"\n✓ Benchmark plot saved as 'performance_benchmark.png'")

def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("INFORMATION EFFICIENCY MICROSTRUCTURE ANALYSIS")
    print("Demonstrating CUDA and SIMD Accelerated Computations")
    print("="*60)
    
    try:
        # Run demonstrations
        demo_variance_ratio()
        demo_autocorrelation()
        demo_full_analysis()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\nGenerated files:")
        print("  - variance_ratios_demo.png")
        print("  - autocorrelation_demo.png")
        print("  - microstructure_analysis_dashboard.png")
        print("  - performance_benchmark.png")
        print("  - analysis_results.json")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
