"""
Validation Tests: Is RDT-Collatz Independence Real?
====================================================
Multiple independent tests to verify the finding isn't an artifact.
"""

import numpy as np
import math
import random
from scipy.stats import pearsonr, spearmanr

# ==========================
# Core Functions
# ==========================

def rdt_depth(n, alpha=1.5):
    """Compute RDT depth."""
    if n <= 1:
        return 0
    k = 0
    x = n
    while x > 1:
        d = max(2, int((math.log(x)) ** alpha))
        x = x // d
        k += 1
        if k > 10000:
            break
    return k


def collatz_stopping_time(n):
    """Compute Collatz stopping time."""
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
        if steps > 100000:
            return -1
    return steps


# ==========================
# TEST 1: Different Sample Sizes
# ==========================

def test_different_ranges():
    """
    Test correlation across different ranges.
    If real, should be consistently low regardless of range.
    """
    print("="*70)
    print("TEST 1: CORRELATION ACROSS DIFFERENT RANGES")
    print("="*70)
    print("\nIf the independence is real, correlation should stay low")
    print("across all ranges. If it's an artifact, it might change.\n")
    
    ranges = [
        (2, 1000),
        (2, 10000),
        (2, 50000),
        (2, 100000),
        (50000, 60000),  # Different slice
        (1000, 11000),   # Small window
    ]
    
    results = []
    
    for start, end in ranges:
        rdt_vals = []
        collatz_vals = []
        
        for n in range(start, min(end, 100001)):
            r = rdt_depth(n)
            c = collatz_stopping_time(n)
            if c >= 0:
                rdt_vals.append(r)
                collatz_vals.append(c)
        
        if len(rdt_vals) > 100:
            corr, _ = pearsonr(rdt_vals, collatz_vals)
            results.append((start, end, len(rdt_vals), corr))
            print(f"  n ∈ [{start:6d}, {end:6d}]: r = {corr:+.4f}  (n={len(rdt_vals)})")
    
    # Check consistency
    correlations = [r[3] for r in results]
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print(f"\n  Mean correlation: {mean_corr:.4f}")
    print(f"  Std deviation:    {std_corr:.4f}")
    
    if std_corr < 0.05:
        print("\n  ✓ CONSISTENT: Correlation is stable across ranges")
        print("    → Independence appears to be real")
    else:
        print("\n  ✗ INCONSISTENT: Correlation varies significantly")
        print("    → Result may be range-dependent artifact")
    
    return results


# ==========================
# TEST 2: Subsampling Stability
# ==========================

def test_subsampling_stability():
    """
    Take multiple random samples and check if correlation stays consistent.
    Real properties don't depend on which specific numbers you sample.
    """
    print("\n" + "="*70)
    print("TEST 2: SUBSAMPLING STABILITY")
    print("="*70)
    print("\nDrawing 10 random samples of 5000 integers from [2, 100000]")
    print("If real, all samples should show similar low correlation.\n")
    
    correlations = []
    
    for trial in range(10):
        # Random sample
        sample = random.sample(range(2, 100001), 5000)
        
        rdt_vals = []
        collatz_vals = []
        
        for n in sample:
            r = rdt_depth(n)
            c = collatz_stopping_time(n)
            if c >= 0:
                rdt_vals.append(r)
                collatz_vals.append(c)
        
        corr, _ = pearsonr(rdt_vals, collatz_vals)
        correlations.append(corr)
        print(f"  Trial {trial+1}: r = {corr:+.4f}")
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print(f"\n  Mean: {mean_corr:.4f}")
    print(f"  Std:  {std_corr:.4f}")
    print(f"  Range: [{min(correlations):.4f}, {max(correlations):.4f}]")
    
    if std_corr < 0.02:
        print("\n  ✓ STABLE: All samples give similar results")
        print("    → Independence is robust to sampling")
    else:
        print("\n  ✗ UNSTABLE: High variance across samples")
        print("    → Result may be sample-dependent")
    
    return correlations


# ==========================
# TEST 3: Control - Compare to Random
# ==========================

def test_control_random():
    """
    Compare RDT-Collatz correlation to correlation with random numbers.
    If RDT-Collatz correlation is similar to random, it's meaningless.
    """
    print("\n" + "="*70)
    print("TEST 3: CONTROL COMPARISON (RDT vs Random)")
    print("="*70)
    print("\nComparing correlation of Collatz with:")
    print("  1. RDT depth (our finding)")
    print("  2. Random values (null hypothesis)")
    print("\nIf RDT-Collatz ≈ Random-Collatz, independence is trivial.\n")
    
    # Get real data
    n_vals = list(range(2, 10001))
    rdt_vals = []
    collatz_vals = []
    
    for n in n_vals:
        r = rdt_depth(n)
        c = collatz_stopping_time(n)
        if c >= 0:
            rdt_vals.append(r)
            collatz_vals.append(c)
    
    # Real correlation
    real_corr, _ = pearsonr(rdt_vals, collatz_vals)
    
    # Random correlations (10 trials)
    random_corrs = []
    for trial in range(10):
        random_vals = [random.randint(1, 6) for _ in rdt_vals]  # Same range as RDT
        rand_corr, _ = pearsonr(random_vals, collatz_vals)
        random_corrs.append(rand_corr)
    
    mean_random = np.mean(random_corrs)
    
    print(f"  RDT-Collatz correlation:    {real_corr:+.4f}")
    print(f"  Random-Collatz correlation: {mean_random:+.4f} (mean of 10 trials)")
    print(f"  Difference:                 {abs(real_corr - mean_random):.4f}")
    
    if abs(real_corr - mean_random) < 0.05:
        print("\n  ⚠ WARNING: RDT ≈ Random")
        print("    → RDT provides no more information than random guessing")
    else:
        print("\n  ✓ RDT ≠ Random")
        print("    → RDT captures real structure (just uncorrelated with Collatz)")
    
    return real_corr, mean_random


# ==========================
# TEST 4: Correlation with log(n)
# ==========================

def test_correlation_with_logn():
    """
    Since RDT depends on log(n), check if the weak correlation
    is just because both RDT and Collatz grow with n.
    """
    print("\n" + "="*70)
    print("TEST 4: CORRELATION WITH log(n)")
    print("="*70)
    print("\nBoth RDT and Collatz might correlate with log(n).")
    print("If so, their correlation might be spurious (both track size).\n")
    
    n_vals = list(range(2, 20001))
    rdt_vals = []
    collatz_vals = []
    log_vals = []
    
    for n in n_vals:
        r = rdt_depth(n)
        c = collatz_stopping_time(n)
        if c >= 0:
            rdt_vals.append(r)
            collatz_vals.append(c)
            log_vals.append(math.log(n))
    
    # Correlations
    rdt_collatz_corr, _ = pearsonr(rdt_vals, collatz_vals)
    rdt_log_corr, _ = pearsonr(rdt_vals, log_vals)
    collatz_log_corr, _ = pearsonr(collatz_vals, log_vals)
    
    print(f"  RDT ↔ Collatz:  r = {rdt_collatz_corr:+.4f}")
    print(f"  RDT ↔ log(n):   r = {rdt_log_corr:+.4f}")
    print(f"  Collatz ↔ log(n): r = {collatz_log_corr:+.4f}")
    
    # Partial correlation: RDT-Collatz controlling for log(n)
    # This removes the shared dependence on n
    # Simple approach: residuals after regressing out log(n)
    rdt_resid = np.array(rdt_vals) - np.polyval(np.polyfit(log_vals, rdt_vals, 1), log_vals)
    collatz_resid = np.array(collatz_vals) - np.polyval(np.polyfit(log_vals, collatz_vals, 1), log_vals)
    
    partial_corr, _ = pearsonr(rdt_resid, collatz_resid)
    
    print(f"\n  Partial correlation (controlling for log(n)): {partial_corr:+.4f}")
    
    if abs(partial_corr) < abs(rdt_collatz_corr) * 0.5:
        print("\n  ⚠ CAUTION: Correlation drops when controlling for size")
        print("    → The weak correlation may be spurious (both track n)")
    else:
        print("\n  ✓ Partial correlation similar to raw correlation")
        print("    → Independence holds even after controlling for size")
    
    return rdt_collatz_corr, partial_corr


# ==========================
# TEST 5: Monotonicity Check
# ==========================

def test_monotonicity():
    """
    Check if there's even a monotonic relationship (not just linear).
    Spearman vs Pearson comparison.
    """
    print("\n" + "="*70)
    print("TEST 5: MONOTONICITY CHECK")
    print("="*70)
    print("\nPearson measures linear correlation.")
    print("Spearman measures any monotonic relationship.")
    print("If Spearman >> Pearson, relationship exists but isn't linear.\n")
    
    n_vals = list(range(2, 20001))
    rdt_vals = []
    collatz_vals = []
    
    for n in n_vals:
        r = rdt_depth(n)
        c = collatz_stopping_time(n)
        if c >= 0:
            rdt_vals.append(r)
            collatz_vals.append(c)
    
    pearson_r, _ = pearsonr(rdt_vals, collatz_vals)
    spearman_r, _ = spearmanr(rdt_vals, collatz_vals)
    
    print(f"  Pearson (linear):     r = {pearson_r:+.4f}")
    print(f"  Spearman (monotonic): ρ = {spearman_r:+.4f}")
    print(f"  Difference:           {abs(spearman_r - pearson_r):.4f}")
    
    if abs(spearman_r - pearson_r) > 0.1:
        print("\n  ⚠ NONLINEAR: Spearman differs from Pearson")
        print("    → There may be a nonlinear relationship")
    else:
        print("\n  ✓ CONSISTENT: Spearman ≈ Pearson")
        print("    → No hidden monotonic relationship")
    
    return pearson_r, spearman_r


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VALIDATION SUITE: IS RDT-COLLATZ INDEPENDENCE REAL?")
    print("="*70)
    print("\nRunning 5 independent tests to verify the finding...\n")
    
    # Run all tests
    test1_results = test_different_ranges()
    test2_results = test_subsampling_stability()
    test3_results = test_control_random()
    test4_results = test_correlation_with_logn()
    test5_results = test_monotonicity()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    print("""
If ALL tests pass:
  ✓ Independence is REAL and ROBUST
  ✓ Safe to publish
  
If ANY test fails:
  ✗ Result may be artifact or conditional
  ✗ Need to investigate further before publishing
  
Check the output above for ✓ (pass) or ✗ (fail) marks.
""")
