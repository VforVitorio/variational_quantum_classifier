"""
Test: Verification of predict_batch vs predict_single_point

Checks that predict_batch returns exactly the same results
as calling predict_single_point individually for each point.

This is critical because the change to batch predictions could have
introduced a bug that explains the degradation from 82% → 53% accuracy.

Usage:
    python test_batch_vs_single.py
"""

import numpy as np
import sys
import os

# Add root directory to path (tests/ -> root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.quantum_circuit import predict_single_point, predict_batch
from data.dataset_generator import make_spiral_dataset


def test_batch_vs_single_predictions(X, params, shots, n_layers):
    """
    Compares batch vs single predictions for all points.

    Returns:
        tuple: (all_match, differences, batch_preds, single_preds)
    """
    print(f"Comparing predictions for {len(X)} points...")
    print(f"  Parameters: {len(params)} params")
    print(f"  Shots: {shots}")
    print(f"  Layers: {n_layers}\n")

    # Batch predictions
    print("Running predict_batch...")
    batch_preds = predict_batch(X, params, shots, n_layers)

    # Individual predictions
    print("Running predict_single_point for each point...")
    single_preds = np.array([
        predict_single_point(X[i, 0], X[i, 1], params, shots, n_layers)
        for i in range(len(X))
    ])

    # Compare
    matches = batch_preds == single_preds
    all_match = np.all(matches)

    differences = []
    for i in range(len(X)):
        if batch_preds[i] != single_preds[i]:
            differences.append({
                'index': i,
                'point': X[i],
                'batch_pred': batch_preds[i],
                'single_pred': single_preds[i]
            })

    return all_match, differences, batch_preds, single_preds


def test_with_multiple_trials(X, params, shots, n_layers, n_trials=5):
    """
    Runs multiple trials to detect stochastic inconsistencies.

    With quantum measurements, predictions can vary between executions
    due to shot noise. This verifies if batch and single have the same
    result distribution.
    """
    print(f"\nRunning {n_trials} trials for statistical analysis...")

    batch_results = []
    single_results = []

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='', flush=True)

        batch_preds = predict_batch(X, params, shots, n_layers)
        single_preds = np.array([
            predict_single_point(X[i, 0], X[i, 1], params, shots, n_layers)
            for i in range(len(X))
        ])

        batch_results.append(batch_preds)
        single_results.append(single_preds)

        print(" OK")

    # Analyze consistency
    batch_results = np.array(batch_results)  # (n_trials, n_points)
    single_results = np.array(single_results)

    # For each point, calculate proportion of times it predicts class 1
    batch_class1_rate = np.mean(batch_results, axis=0)  # (n_points,)
    single_class1_rate = np.mean(single_results, axis=0)

    # Calculate differences
    rate_differences = np.abs(batch_class1_rate - single_class1_rate)

    return batch_class1_rate, single_class1_rate, rate_differences


if __name__ == "__main__":
    print("="*60)
    print("TEST: Verification predict_batch vs predict_single_point")
    print("="*60)

    # Generate small test dataset
    print("\n[1/3] Generating test dataset...")
    np.random.seed(42)
    X_test, y_test = make_spiral_dataset(n_points=30, noise=0.1, normalize=True)
    print(f"Dataset: {len(X_test)} points")

    # Test parameters (random but fixed)
    n_params = 8
    n_layers = 2
    shots = 100

    params = np.random.rand(n_params) * 2 * np.pi
    print(f"Random parameters: {n_params} values")
    print(f"Configuration: {n_layers} layers, {shots} shots")

    # Test 1: Direct comparison
    print("\n" + "="*60)
    print("[2/3] TEST 1: Direct Comparison (1 execution)")
    print("="*60)

    all_match, differences, batch_preds, single_preds = test_batch_vs_single_predictions(
        X_test, params, shots, n_layers
    )

    if all_match:
        print("\n✅ ALL predictions match!")
        print(f"   {len(X_test)}/{len(X_test)} points predicted identically")
    else:
        print(f"\n❌ DISCREPANCIES DETECTED!")
        print(f"   {len(differences)}/{len(X_test)} points differ")
        print(f"\nDiscrepancy details:")
        for diff in differences[:10]:  # Show first 10
            print(f"  Point {diff['index']}: {diff['point']}")
            print(f"    batch_pred:  {diff['batch_pred']}")
            print(f"    single_pred: {diff['single_pred']}")

    # Show first predictions
    print("\nFirst 10 predictions:")
    print("  Index | X[0]   | X[1]   | Batch | Single | Match")
    print("  " + "-"*54)
    for i in range(min(10, len(X_test))):
        match_symbol = "✓" if batch_preds[i] == single_preds[i] else "✗"
        print(f"  {i:5d} | {X_test[i,0]:6.3f} | {X_test[i,1]:6.3f} | "
              f"{batch_preds[i]:5d} | {single_preds[i]:6d} | {match_symbol}")

    # Test 2: Statistical analysis with multiple trials
    print("\n" + "="*60)
    print("[3/3] TEST 2: Statistical Analysis (5 trials)")
    print("="*60)

    batch_rates, single_rates, rate_diffs = test_with_multiple_trials(
        X_test, params, shots, n_layers, n_trials=5
    )

    print("\nPrediction rate analysis (proportion of class 1):")
    print("  Maximum difference: {:.2%}".format(np.max(rate_diffs)))
    print("  Average differences: {:.2%}".format(np.mean(rate_diffs)))
    print("  Standard deviation: {:.2%}".format(np.std(rate_diffs)))

    # Determine if differences are significant
    # With shots=100, we expect ~10% variation due to quantum noise
    expected_shot_noise = 1.0 / np.sqrt(shots)  # ~10% for shots=100

    if np.max(rate_diffs) < expected_shot_noise * 2:
        print(f"\n✅ Differences WITHIN expected quantum noise (~{expected_shot_noise:.1%})")
        print("   batch and single are statistically equivalent")
    else:
        print(f"\n⚠️ Differences LARGER than expected noise (~{expected_shot_noise:.1%})")
        print("   Possible inconsistency between batch and single")

    # Points with highest discrepancy
    top_discrepancies = np.argsort(rate_diffs)[::-1][:5]
    print("\nPoints with highest discrepancy (top 5):")
    print("  Index | Batch Rate | Single Rate | Difference")
    print("  " + "-"*50)
    for idx in top_discrepancies:
        print(f"  {idx:5d} | {batch_rates[idx]:10.2%} | {single_rates[idx]:11.2%} | "
              f"{rate_diffs[idx]:10.2%}")

    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if all_match and np.max(rate_diffs) < expected_shot_noise * 2:
        print("✅ predict_batch and predict_single_point are EQUIVALENT")
        print("   The change to batch predictions is NOT the cause of the problem")
        print("\nThe problem must be elsewhere:")
        print("  - Changes in optimizer (COBYLA/SLSQP)")
        print("  - Early stopping parameters")
        print("  - Dataset too small (120 train vs 2000 typical)")
        print("  - Too few layers (2 vs 5-6 typical)")
    elif not all_match:
        print("❌ predict_batch and predict_single_point DO NOT match")
        print("   BUG DETECTED in predict_batch!")
        print("\nThis could explain the degradation 82% → 53%")
    else:
        print("⚠️ Statistical differences detected between batch and single")
        print("   Review predict_batch implementation")

    print("="*60)
