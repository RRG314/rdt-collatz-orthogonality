"""
RDT vs Collatz Stopping Time Comparison
========================================
Compare the Recursive Division Tree depth with Collatz stopping time
to see if these two recursion-based complexity measures are correlated.

Author: Steven Reid
ORCID: 0009-0003-9132-3410
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import math
import time
import os

# ==========================
# RDT Implementation
# ==========================

def rdt_depth(n, alpha=1.5):
    """
    Compute RDT depth for integer n.
    
    Args:
        n: positive integer
        alpha: exponent parameter (default 1.5)
    
    Returns:
        Number of steps to reach 1
    """
    if n <= 1:
        return 0
    k = 0
    x = n
    while x > 1:
        d = max(2, int((math.log(x)) ** alpha))
        x = x // d
        k += 1
        if k > 10000:  # Safety limit
            break
    return k


# ==========================
# Collatz Implementation
# ==========================

def collatz_stopping_time(n):
    """
    Compute Collatz stopping time (total stopping time).
    This counts steps until reaching 1.
    
    Collatz rule:
    - If n is even: n -> n/2
    - If n is odd: n -> 3n+1
    
    Args:
        n: positive integer
    
    Returns:
        Number of steps to reach 1, or -1 if exceeds safety limit
    """
    if n <= 0:
        return 0
    if n == 1:
        return 0
    
    steps = 0
    current = n
    
    while current != 1:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1
        steps += 1
        
        # Safety limit (Collatz is unproven, might not terminate)
        if steps > 100000:
            return -1  # Mark as non-terminating or too long
    
    return steps


# ==========================
# Data Collection
# ==========================

def collect_comparison_data(n_max=100000, sample_size=None):
    """
    Collect RDT and Collatz data for integers up to n_max.
    
    Args:
        n_max: maximum integer to compute (default 100000)
        sample_size: if specified, randomly sample this many integers
                    otherwise compute for all integers 2 to n_max
    
    Returns:
        Tuple of (n_values, rdt_depths, collatz_times) as numpy arrays
    """
    print(f"Collecting data for n = 2 to {n_max}...")
    
    if sample_size:
        print(f"Using random sample of {sample_size} integers")
        integers = np.random.choice(range(2, n_max+1), size=sample_size, replace=False)
        integers = sorted(integers)
    else:
        integers = range(2, n_max+1)
    
    rdt_depths = []
    collatz_times = []
    valid_n = []
    
    start_time = time.time()
    
    for i, n in enumerate(integers):
        rdt = rdt_depth(n)
        collatz = collatz_stopping_time(n)
        
        # Only include if Collatz terminated
        if collatz >= 0:
            rdt_depths.append(rdt)
            collatz_times.append(collatz)
            valid_n.append(n)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(list(integers))} ({elapsed:.1f}s)")
    
    print(f"Collected {len(valid_n)} valid data points in {time.time()-start_time:.2f}s\n")
    
    return np.array(valid_n), np.array(rdt_depths), np.array(collatz_times)


# ==========================
# Statistical Analysis
# ==========================

def analyze_correlation(n_values, rdt_depths, collatz_times):
    """
    Perform statistical analysis of the correlation.
    
    Returns:
        Tuple of (pearson_r, spearman_r)
    """
    
    print("="*70)
    print("CORRELATION ANALYSIS: RDT vs COLLATZ")
    print("="*70)
    
    # Basic statistics
    print(f"\nSample size: {len(n_values)}")
    print(f"Range: n ∈ [{n_values.min()}, {n_values.max()}]")
    
    print(f"\nRDT Depth Statistics:")
    print(f"  Min: {rdt_depths.min()}")
    print(f"  Max: {rdt_depths.max()}")
    print(f"  Mean: {rdt_depths.mean():.2f}")
    print(f"  Std: {rdt_depths.std():.2f}")
    
    print(f"\nCollatz Stopping Time Statistics:")
    print(f"  Min: {collatz_times.min()}")
    print(f"  Max: {collatz_times.max()}")
    print(f"  Mean: {collatz_times.mean():.2f}")
    print(f"  Std: {collatz_times.std():.2f}")
    
    # Correlation coefficients
    pearson_r, pearson_p = pearsonr(rdt_depths, collatz_times)
    spearman_r, spearman_p = spearmanr(rdt_depths, collatz_times)
    
    print(f"\n" + "="*70)
    print("CORRELATION COEFFICIENTS")
    print("="*70)
    print(f"\nPearson correlation (linear):  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"Spearman correlation (rank):   ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
    
    # Interpretation
    print(f"\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if abs(pearson_r) > 0.7:
        strength = "STRONG"
    elif abs(pearson_r) > 0.4:
        strength = "MODERATE"
    elif abs(pearson_r) > 0.2:
        strength = "WEAK"
    else:
        strength = "NEGLIGIBLE"
    
    print(f"\nCorrelation strength: {strength}")
    
    if pearson_r > 0:
        print("Direction: POSITIVE (higher RDT → higher Collatz)")
    else:
        print("Direction: NEGATIVE (higher RDT → lower Collatz)")
    
    if pearson_p < 0.001:
        print("Statistical significance: p < 0.001 (highly significant)")
    elif pearson_p < 0.05:
        print("Statistical significance: p < 0.05 (significant)")
    else:
        print("Statistical significance: p ≥ 0.05 (not significant)")
    
    # R-squared
    r_squared = pearson_r ** 2
    print(f"\nR² = {r_squared:.4f}")
    print(f"Interpretation: {r_squared*100:.2f}% of variance in Collatz explained by RDT")
    
    print(f"\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if abs(pearson_r) > 0.5:
        print("\n✓ RDT and Collatz ARE correlated")
        print("  → They may measure related aspects of integer complexity")
        print("  → RDT could potentially inform Collatz research")
    else:
        print("\n✗ RDT and Collatz are NOT strongly correlated")
        print("  → They measure different aspects of integer complexity")
        print("  → RDT provides orthogonal information to Collatz")
    
    return pearson_r, spearman_r


# ==========================
# Visualization
# ==========================

def create_visualizations(n_values, rdt_depths, collatz_times, pearson_r, output_dir='.'):
    """
    Create comprehensive visualization plots.
    
    Args:
        n_values: array of integers
        rdt_depths: array of RDT depths
        collatz_times: array of Collatz stopping times
        pearson_r: Pearson correlation coefficient
        output_dir: directory to save plots (default: current directory)
    
    Returns:
        matplotlib figure object
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(rdt_depths, collatz_times, alpha=0.3, s=1, c='blue')
    ax1.set_xlabel('RDT Depth', fontsize=11)
    ax1.set_ylabel('Collatz Stopping Time', fontsize=11)
    ax1.set_title(f'RDT vs Collatz (Pearson r = {pearson_r:.3f})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(rdt_depths, collatz_times, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rdt_depths.min(), rdt_depths.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, 
             label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()
    
    # Plot 2: Hexbin (density plot)
    ax2 = axes[0, 1]
    hb = ax2.hexbin(rdt_depths, collatz_times, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('RDT Depth', fontsize=11)
    ax2.set_ylabel('Collatz Stopping Time', fontsize=11)
    ax2.set_title('Density Plot (Hexbin)', fontsize=12)
    plt.colorbar(hb, ax=ax2, label='Count')
    
    # Plot 3: RDT distribution
    ax3 = axes[1, 0]
    ax3.hist(rdt_depths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('RDT Depth', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('RDT Depth Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Collatz distribution
    ax4 = axes[1, 1]
    ax4.hist(collatz_times, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Collatz Stopping Time', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Collatz Stopping Time Distribution', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save with error handling
    try:
        output_path = os.path.join(output_dir, 'rdt_vs_collatz.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    except Exception as e:
        print(f"\nWarning: Could not save figure to {output_dir}")
        print(f"Error: {e}")
        print("Figure will still be displayed.")
    
    return fig


# ==========================
# Deeper Analysis
# ==========================

def analyze_by_rdt_depth(n_values, rdt_depths, collatz_times):
    """
    Analyze Collatz statistics for each RDT depth value.
    Shows mean, std, and range of Collatz times grouped by RDT depth.
    """
    
    print(f"\n" + "="*70)
    print("COLLATZ STATISTICS BY RDT DEPTH")
    print("="*70)
    
    unique_depths = sorted(set(rdt_depths))
    
    print(f"\n{'RDT':<6} {'Count':<8} {'Collatz Mean':<14} {'Collatz Std':<14} {'Collatz Range'}")
    print("-" * 70)
    
    for depth in unique_depths:
        mask = rdt_depths == depth
        collatz_subset = collatz_times[mask]
        
        if len(collatz_subset) > 0:
            mean_c = collatz_subset.mean()
            std_c = collatz_subset.std()
            min_c = collatz_subset.min()
            max_c = collatz_subset.max()
            count = len(collatz_subset)
            
            print(f"{depth:<6} {count:<8} {mean_c:<14.2f} {std_c:<14.2f} [{min_c}, {max_c}]")


def find_extreme_cases(n_values, rdt_depths, collatz_times):
    """
    Find interesting extreme cases where RDT and Collatz diverge.
    Identifies numbers that are simple in one measure but complex in another.
    """
    
    print(f"\n" + "="*70)
    print("EXTREME CASES")
    print("="*70)
    
    # Calculate ratio: Collatz / RDT (higher = more Collatz-complex relative to RDT)
    ratio = collatz_times / (rdt_depths + 1)  # Add 1 to avoid division by zero
    
    # Low ratio: efficient in both
    low_ratio_idx = np.argsort(ratio)[:10]
    
    print("\nTop 10: Low Collatz/RDT Ratio (efficient in both measures):")
    print(f"{'n':<12} {'RDT':<8} {'Collatz':<10} {'Ratio':<10}")
    print("-" * 42)
    for idx in low_ratio_idx:
        n = n_values[idx]
        rdt = rdt_depths[idx]
        col = collatz_times[idx]
        r = ratio[idx]
        print(f"{n:<12} {rdt:<8} {col:<10} {r:<10.2f}")
    
    # High ratio: Collatz anomalies
    high_ratio_idx = np.argsort(ratio)[-10:][::-1]
    
    print("\nTop 10: High Collatz/RDT Ratio (Collatz complexity anomalies):")
    print(f"{'n':<12} {'RDT':<8} {'Collatz':<10} {'Ratio':<10}")
    print("-" * 42)
    for idx in high_ratio_idx:
        n = n_values[idx]
        rdt = rdt_depths[idx]
        col = collatz_times[idx]
        r = ratio[idx]
        print(f"{n:<12} {rdt:<8} {col:<10} {r:<10.2f}")


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    
    # Configuration
    N_MAX = 100000  # Maximum integer to analyze
    SAMPLE_SIZE = None  # Use None for all integers, or specify sample size
    
    print("="*70)
    print("RDT vs COLLATZ STOPPING TIME ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  N_max: {N_MAX}")
    print(f"  Sample: {'All integers' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE} random'}")
    print()
    
    # Collect data
    n_values, rdt_depths, collatz_times = collect_comparison_data(
        n_max=N_MAX,
        sample_size=SAMPLE_SIZE
    )
    
    # Analyze correlation
    pearson_r, spearman_r = analyze_correlation(n_values, rdt_depths, collatz_times)
    
    # Deeper analysis
    analyze_by_rdt_depth(n_values, rdt_depths, collatz_times)
    find_extreme_cases(n_values, rdt_depths, collatz_times)
    
    # Visualize
    create_visualizations(n_values, rdt_depths, collatz_times, pearson_r)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nClose the plot window to exit.")
    
    plt.show()
