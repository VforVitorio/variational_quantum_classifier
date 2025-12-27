"""
Main Script - Variational Quantum Classifier Demo

Executes the complete pipeline:
    1. Load spiral dataset
    2. Train quantum classifier
    3. Evaluate performance
    4. Generate visualizations

Usage:
    python main.py
"""

from src.utils import (plot_dataset, plot_decision_boundary,
                       calculate_metrics, print_metrics,
                       plot_training_history, save_results)
from src.classifier import QuantumClassifier
from data.dataset_generator import make_spiral_dataset, load_dataset
import numpy as np
import sys
import os
import threading
from queue import Queue
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))


def train_attempt_thread(thread_id, X, y, seed, n_params, shots, n_layers, max_iter, results_queue):
    """
    Trains a classifier in a separate thread.
    Thread-safe following PyQuil multithreading guide.

    Args:
        thread_id: Thread ID
        X, y: Training data
        seed: Random seed
        n_params, shots, n_layers: Classifier configuration
        max_iter: Optimization iterations
        results_queue: Thread-safe queue for results
    """
    try:
        # Unique seed per thread
        np.random.seed(seed)

        # Create classifier (independent instance per thread)
        classifier = QuantumClassifier(
            n_qubits=2,
            n_params=n_params,
            shots=shots,
            n_layers=n_layers
        )

        # Train (verbose=False to avoid mixed outputs between threads)
        start_time = time.time()
        training_result = classifier.train(
            X, y,
            max_iter=max_iter,
            method='COBYLA',
            verbose=False
        )
        training_time = time.time() - start_time

        # Evaluate
        accuracy = classifier.evaluate(X, y)

        # Save result in thread-safe queue
        results_queue.put({
            'thread_id': thread_id,
            'accuracy': accuracy,
            'classifier': classifier,
            'training_result': training_result,
            'time': training_time
        })

    except Exception as e:
        # Catch errors to avoid blocking other threads
        results_queue.put({
            'thread_id': thread_id,
            'error': str(e),
            'accuracy': 0.0
        })


def train_parallel(X, y, n_attempts=3, n_params=8, shots=100, n_layers=2, max_iter=80):
    """
    Trains multiple attempts in parallel using threading.

    Returns:
        tuple: (best_classifier, best_training_result, best_accuracy, results)
    """
    print(f"Parallel training: {n_attempts} simultaneous threads\n")

    results_queue = Queue()
    threads = []
    seeds = [42, 123, 456, 789, 101112][:n_attempts]  # Different seeds

    # Create and launch threads
    start_time = time.time()
    for i, seed in enumerate(seeds):
        thread = threading.Thread(
            target=train_attempt_thread,
            args=(i, X, y, seed, n_params, shots,
                  n_layers, max_iter, results_queue),
            name=f"Attempt-{i+1}"
        )
        threads.append(thread)
        thread.start()
        print(f"→ Attempt {i+1}/{n_attempts} started (Thread-{i+1})")

    print(f"\nAll threads active. Waiting for results...\n")

    # Wait for all to finish
    for i, thread in enumerate(threads):
        thread.join()
        print(f"✓ Attempt {i+1} completed")

    total_time = time.time() - start_time
    print(
        f"\nTotal parallel time: {total_time:.1f}s ({total_time/60:.1f} min)\n")

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Filter errors and sort by accuracy
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        raise RuntimeError("All attempts failed")

    results_sorted = sorted(
        valid_results, key=lambda r: r['accuracy'], reverse=True)

    # Show summary
    print("Attempt summary:")
    for r in results_sorted:
        print(f"  Attempt {r['thread_id']+1}: Accuracy={r['accuracy']:.2%}, "
              f"Time={r['time']:.1f}s")

    # Return best
    best = results_sorted[0]
    return best['classifier'], best['training_result'], best['accuracy'], results_sorted


def main():
    """Main pipeline for the quantum classifier."""

    print("=" * 60)
    print("VARIATIONAL QUANTUM CLASSIFIER (VQC)")
    print("=" * 60)

    # =========================================================================
    # 1. DATA GENERATION/LOADING
    # =========================================================================
    print("\n[1/5] Generating spiral dataset...")

    # Dataset configuration
    # n_points: 50 (quick tests), 150 (standard), 400 (production)
    X, y = make_spiral_dataset(n_points=150, noise=0.1, normalize=True)

    print(f"Dataset generated: {X.shape[0]} points")
    print(f"Distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")

    # =========================================================================
    # SPLIT TRAIN/VALIDATION (80/20)
    # =========================================================================
    # Use fixed seed for reproducibility
    np.random.seed(42)

    # Create random indices
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"\nSplit train/validation (80/20):")
    print(
        f"  Train: {len(X_train)} points (Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)})")
    print(
        f"  Val:   {len(X_val)} points (Class 0={np.sum(y_val==0)}, Class 1={np.sum(y_val==1)})")

    # Visualize complete dataset
    plot_dataset(X, y,
                 title="Intertwined Spiral Dataset (150 points)",
                 save_path="results/dataset.png")

    # =========================================================================
    # 2. TRAINING WITH MULTIPLE ATTEMPTS
    # =========================================================================
    print("\n[2/5] Training classifier (multiple attempts)...")

    # Configuration
    n_attempts = 1  # REDUCED: 3 → 1 (compensate with better optimizer)
    USE_THREADING = False  # Disabled (GIL doesn't allow real parallelism)

    # Classifier parameters (OPTION A: COBYLA + more shots)
    n_params = 8      # 2 layers × 2 qubits × 2 rotations
    shots = 500       # Increased 150→300→500 (reduces shot noise: 8% → 4.5%)
    n_layers = 2      # Keep architecture that worked
    max_iter = 120    # Sufficient for convergence

    if USE_THREADING:
        # Parallel version (not recommended - GIL issue)
        best_classifier, best_training_result, best_accuracy, all_results = train_parallel(
            X_train, y_train,
            n_attempts=n_attempts,
            n_params=n_params,
            shots=shots,
            n_layers=n_layers,
            max_iter=max_iter
        )
    else:
        # Sequential version (recommended)
        print(f"Training attempts: {n_attempts} (sequential)\n")
        best_accuracy = 0
        best_val_accuracy = 0
        best_classifier = None
        best_training_result = None

        for attempt in range(n_attempts):
            print(f"--- Attempt {attempt + 1}/{n_attempts} ---")

            classifier = QuantumClassifier(
                n_qubits=2,
                n_params=n_params,
                shots=shots,
                n_layers=n_layers
            )

            # Train only with training data
            training_result = classifier.train(
                X_train, y_train,
                max_iter=max_iter,
                method='COBYLA',    # Back to COBYLA (showed better convergence)
                verbose=True,
                patience=40,        # More permissive than before (was 30)
                min_delta=0.002     # More permissive than before (was 0.003)
            )

            # Evaluate on train and validation
            train_accuracy = classifier.evaluate(X_train, y_train)
            val_accuracy = classifier.evaluate(X_val, y_val)

            print(f"\nTrain Accuracy: {train_accuracy:.2%}")
            print(f"Val Accuracy:   {val_accuracy:.2%}")
            print(f"Final cost: {training_result['final_cost']:.4f}")
            print(f"Time: {training_result['time']:.1f}s")

            # Calculate overfitting gap
            overfit_gap = train_accuracy - val_accuracy
            if overfit_gap > 0.1:
                print(f"⚠️  Possible overfitting (gap: {overfit_gap:.2%})")
            print()

            # Select best model based on VALIDATION accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_accuracy = train_accuracy
                best_classifier = classifier
                best_training_result = training_result
                print("★ New best model saved (based on val accuracy)!\n")

    print(f"\n★ Best Train Accuracy: {best_accuracy:.2%}")
    print(f"★ Best Val Accuracy:   {best_val_accuracy:.2%}")

    # =========================================================================
    # 3. DETAILED EVALUATION
    # =========================================================================
    print("\n[3/5] Evaluating best model...")

    # Evaluate on both sets
    print("\nEvaluation on Train:")
    y_pred_train = best_classifier.predict(X_train)
    metrics_train = calculate_metrics(y_train, y_pred_train)
    print_metrics(metrics_train)

    print("\n" + "-" * 40)
    print("Evaluation on Validation:")
    y_pred_val = best_classifier.predict(X_val)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    print_metrics(metrics_val)

    # Final metrics (use validation for report)
    final_metrics = metrics_val

    # =========================================================================
    # 4. RESULTS VISUALIZATION
    # =========================================================================
    print("\n[4/5] Generating visualizations...")

    # Decision boundary visualization (with all data)
    # resolution: 30 (fast), 40 (standard), 100 (high quality)
    plot_decision_boundary(
        best_classifier, X, y,
        resolution=40,
        title="Learned Decision Boundary (Complete Dataset)",
        save_path="results/decision_boundary.png"
    )

    # Convergence of best model
    if best_classifier.training_history['cost']:
        plot_training_history(
            best_classifier.training_history,
            save_path="results/training_convergence.png"
        )

    # =========================================================================
    # 5. SAVE RESULTS
    # =========================================================================
    print("\n[5/5] Saving results...")

    # Save metrics (validation)
    save_results(final_metrics, best_training_result, "results/metrics.txt")

    # Save best model parameters
    best_classifier.save_params("results/best_model_params.pkl")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Dataset: {len(X)} points (Train: {len(X_train)}, Val: {len(X_val)})")
    print(f"Training attempts: {n_attempts}")
    print(f"Best Train Accuracy: {best_accuracy:.2%}")
    print(f"Best Val Accuracy:   {best_val_accuracy:.2%}")
    print(f"Overfitting gap:     {(best_accuracy - best_val_accuracy):.2%}")
    print(f"Best model time: {best_training_result['time']:.2f}s")
    print(f"Iterations: {best_training_result['iterations']}")
    if best_training_result.get('stopped_early'):
        print(f"🛑 Early stopping activated")
    print("\nGenerated files:")
    print("  - results/dataset.png")
    print("  - results/decision_boundary.png")
    print("  - results/training_convergence.png")
    print("  - results/metrics.txt (validation metrics)")
    print("  - results/best_model_params.pkl")
    print("\nPipeline completed successfully!")
    print("\nNote: The model was selected using validation accuracy")
    print("      to avoid overfitting.")


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Execute pipeline
    main()
