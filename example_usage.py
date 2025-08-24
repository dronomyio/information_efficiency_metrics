# example_usage.py
"""
Example usage of the Information Efficiency Analysis System
Demonstrates how to use the API for analyzing market microstructure
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict
import os

# Configure plotting
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 8)


class InfoEfficiencyClient:
    """Client for interacting with the Information Efficiency API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def analyze_symbols(
        self,
        symbols: List[str],
        dates: List[str],
        metrics: List[str] = ['vr', 'acf']
    ) -> pd.DataFrame:
        """Run analysis for multiple symbols and dates"""
        
        payload = {
            "symbols": symbols,
            "dates": dates,
            "metrics": metrics
        }
        
        response = self.session.post(
            f"{self.base_url}/analyze",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return pd.DataFrame(data['results'])
        
        raise Exception(f"Analysis failed: {response.text}")
    
    def get_historical_results(
        self,
        symbol: str,
        date: str
    ) -> Dict:
        """Retrieve historical analysis results"""
        
        response = self.session.get(
            f"{self.base_url}/results/{symbol}/{date}"
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            raise Exception(f"Failed to retrieve results: {response.text}")


def plot_variance_ratios(df: pd.DataFrame, symbol: str):
    """Plot variance ratios across different horizons"""
    
    vr_columns = [col for col in df.columns if col.startswith('vr_')]
    horizons = [int(col.split('_')[1]) for col in vr_columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Variance Ratio Analysis - {symbol}', fontsize=16)
    
    # Plot 1: VR over time for different horizons
    ax1 = axes[0, 0]
    for col in vr_columns:
        ax1.plot(pd.to_datetime(df['date']), df[col], marker='o', label=col)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Variance Ratio')
    ax1.set_title('Variance Ratios Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average VR by horizon
    ax2 = axes[0, 1]
    avg_vr = [df[col].mean() for col in vr_columns]
    std_vr = [df[col].std() for col in vr_columns]
    ax2.errorbar(horizons, avg_vr, yerr=std_vr, marker='o', capsize=5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Horizon (h)')
    ax2.set_ylabel('Average Variance Ratio')
    ax2.set_title('Mean Variance Ratio by Horizon')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of VR values
    ax3 = axes[1, 0]
    vr_data = df[vr_columns].values.flatten()
    ax3.hist(vr_data, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Variance Ratio')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Variance Ratios')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of VR values
    ax4 = axes[1, 1]
    vr_matrix = df[vr_columns].values
    im = ax4.imshow(vr_matrix.T, aspect='auto', cmap='RdBu_r', vmin=0.5, vmax=1.5)
    ax4.set_yticks(range(len(vr_columns)))
    ax4.set_yticklabels(vr_columns)
    ax4.set_xlabel('Date Index')
    ax4.set_ylabel('Horizon')
    ax4.set_title('Variance Ratio Heatmap')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    return fig


def plot_autocorrelation_decay(df: pd.DataFrame, symbol: str):
    """Plot autocorrelation decay characteristics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Autocorrelation Analysis - {symbol}', fontsize=16)
    
    # Plot 1: ACF decay parameter (phi) over time
    ax1 = axes[0, 0]
    ax1.plot(pd.to_datetime(df['date']), df['acf_phi'], marker='o', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Decay Parameter (φ)')
    ax1.set_title('Autocorrelation Decay Parameter Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Half-life over time
    ax2 = axes[0, 1]
    ax2.plot(pd.to_datetime(df['date']), df['acf_half_life'], marker='o', color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Half-life (periods)')
    ax2.set_title('Autocorrelation Half-life Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: First-lag autocorrelation
    ax3 = axes[1, 0]
    ax3.plot(pd.to_datetime(df['date']), df['acf_lag1'], marker='o', color='orange')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('ACF(1)')
    ax3.set_title('First-Lag Autocorrelation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relationship between phi and half-life
    ax4 = axes[1, 1]
    ax4.scatter(df['acf_phi'], df['acf_half_life'], alpha=0.6)
    ax4.set_xlabel('Decay Parameter (φ)')
    ax4.set_ylabel('Half-life (periods)')
    ax4.set_title('Decay Parameter vs Half-life')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_efficiency_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate composite efficiency score based on VR and ACF metrics
    Score ranges from 0 (least efficient) to 100 (most efficient)
    """
    
    scores = pd.Series(index=df.index, dtype=float)
    
    # VR component: deviation from 1 (lower is better)
    vr_columns = [col for col in df.columns if col.startswith('vr_')]
    vr_deviations = df[vr_columns].apply(lambda x: np.abs(x - 1))
    vr_score = 100 * (1 - vr_deviations.mean(axis=1))
    vr_score = np.clip(vr_score, 0, 100)
    
    # ACF component: faster decay is better
    # Normalize half-life (lower is better)
    max_half_life = df['acf_half_life'].quantile(0.95)
    acf_score = 100 * (1 - df['acf_half_life'] / max_half_life)
    acf_score = np.clip(acf_score, 0, 100)
    
    # Combine scores (equal weighting)
    scores = 0.5 * vr_score + 0.5 * acf_score
    
    return scores


async def main():
    """Main example workflow"""
    
    # Initialize client
    client = InfoEfficiencyClient()
    
    # Define analysis parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Generate date range (last 30 trading days)
    end_date = datetime.now()
    dates = []
    current_date = end_date - timedelta(days=42)  # Account for weekends
    
    while len(dates) < 30 and current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    print(f"Analyzing {len(symbols)} symbols over {len(dates)} days...")
    
    # Run analysis
    results = client.analyze_symbols(symbols, dates)
    
    # Add efficiency scores
    for symbol in symbols:
        symbol_data = results[results['symbol'] == symbol]
        if not symbol_data.empty:
            results.loc[symbol_data.index, 'efficiency_score'] = \
                calculate_efficiency_score(symbol_data)
    
    # Save results
    results.to_csv('efficiency_analysis_results.csv', index=False)
    print(f"Results saved to efficiency_analysis_results.csv")
    
    # Generate visualizations for each symbol
    for symbol in symbols:
        symbol_data = results[results['symbol'] == symbol]
        
        if not symbol_data.empty:
            # Create variance ratio plots
            vr_fig = plot_variance_ratios(symbol_data, symbol)
            vr_fig.savefig(f'variance_ratios_{symbol}.png', dpi=300, bbox_inches='tight')
            
            # Create autocorrelation plots
            acf_fig = plot_autocorrelation_decay(symbol_data, symbol)
            acf_fig.savefig(f'autocorrelation_{symbol}.png', dpi=300, bbox_inches='tight')
            
            print(f"Plots saved for {symbol}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary = results.groupby('symbol').agg({
        'efficiency_score': ['mean', 'std', 'min', 'max'],
        'vr_10': ['mean', 'std'],
        'acf_half_life': ['mean', 'std'],
        'num_observations': 'mean'
    }).round(3)
    
    print(summary)
    
    # Rank symbols by average efficiency score
    print("\n" + "="*60)
    print("EFFICIENCY RANKINGS")
    print("="*60)
    
    rankings = results.groupby('symbol')['efficiency_score'].mean().sort_values(ascending=False)
    for rank, (symbol, score) in enumerate(rankings.items(), 1):
        print(f"{rank}. {symbol}: {score:.2f}")
    
    # Identify periods of low efficiency
    print("\n" + "="*60)
    print("LOW EFFICIENCY PERIODS (Score < 30)")
    print("="*60)
    
    low_efficiency = results[results['efficiency_score'] < 30][
        ['symbol', 'date', 'efficiency_score', 'vr_10', 'acf_half_life']
    ].sort_values('efficiency_score')
    
    if not low_efficiency.empty:
        print(low_efficiency.head(10))
    else:
        print("No periods with efficiency score < 30")
    
    # Time series correlation analysis
    print("\n" + "="*60)
    print("CROSS-SYMBOL EFFICIENCY CORRELATION")
    print("="*60)
    
    efficiency_pivot = results.pivot(
        index='date', 
        columns='symbol', 
        values='efficiency_score'
    )
    
    correlation_matrix = efficiency_pivot.corr()
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True,
        ax=ax
    )
    ax.set_title('Cross-Symbol Efficiency Correlation')
    plt.savefig('efficiency_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    
    print("\nCorrelation matrix saved to efficiency_correlation_heatmap.png")
    print("\nTop correlated pairs:")
    
    # Find top correlated pairs
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            symbol1 = correlation_matrix.columns[i]
            symbol2 = correlation_matrix.columns[j]
            corr = correlation_matrix.iloc[i, j]
            corr_pairs.append((symbol1, symbol2, corr))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for symbol1, symbol2, corr in corr_pairs[:5]:
        print(f"{symbol1} - {symbol2}: {corr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
