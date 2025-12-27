"""
Visualization and Metrics Utilities

Helper functions for evaluating and visualizing quantum classifier results.

Functions:
    plot_dataset: Visualize dataset points
    plot_decision_boundary: Classifier decision boundary
    calculate_metrics: Accuracy, precision, recall
    plot_training_history: Cost evolution during training
    save_results: Save metrics to text file
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Optional, Dict, Tuple
import os


# =============================================================================
# DATA VISUALIZATION
# =============================================================================

def plot_dataset(X: np.ndarray, y: np.ndarray,
                 title: str = "Dataset",
                 save_path: Optional[str] = None) -> None:
    """
    Visualizes dataset with color-coded classes.

    Args:
        X: Coordinates (n_samples, 2)
        y: Labels (n_samples,)
        title: Plot title
        save_path: Path to save (optional)
    """
    plt.figure(figsize=(8, 8))

    # Class 0
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=50, alpha=0.7,
                label='Class 0', edgecolors='darkred', linewidths=1.5)

    # Class 1
    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=50, alpha=0.7,
                label='Class 1', edgecolors='darkblue', linewidths=1.5)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()


def plot_decision_boundary(classifier, X: np.ndarray, y: np.ndarray,
                           resolution: int = 100,
                           title: str = "Decision Boundary",
                           save_path: Optional[str] = None) -> None:
    """
    Visualizes classifier decision boundary.

    Args:
        classifier: Trained QuantumClassifier instance
        X: Coordinates (n_samples, 2)
        y: Labels (n_samples,)
        resolution: Grid resolution (default: 100)
        title: Plot title
        save_path: Path to save
    """
    # Create point grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict class for each grid point
    print(
        f"Generating decision boundary ({resolution}x{resolution} points)...")
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Visualize
    plt.figure(figsize=(10, 8))

    # Boundary contour
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=1)
    plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=1)

    # Training data
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=50, alpha=0.8,
                label='Class 0', edgecolors='darkred', linewidths=1.5)

    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=50, alpha=0.8,
                label='Class 1', edgecolors='darkblue', linewidths=1.5)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates classification metrics.

    Args:
        y_true: True labels
        y_pred: Predictions

    Returns:
        dict: Metrics (accuracy, precision, recall)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Prints formatted metrics.

    Args:
        metrics: Metrics dictionary
    """
    print("\n=== Classification Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])


# =============================================================================
# TRAINING
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualizes cost evolution during training.

    Args:
        history: Dictionary with 'cost' and 'iteration'
        save_path: Path to save
    """
    if not history['cost']:
        print("No training history available.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['iteration'], history['cost'],
             'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (Error)', fontsize=12)
    plt.title('Training Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    plt.show()


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_results(metrics: Dict, training_info: Dict, filepath: str) -> None:
    """
    Saves metrics and results to text file.

    Args:
        metrics: Evaluation metrics
        training_info: Training information
        filepath: File path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("VARIATIONAL QUANTUM CLASSIFIER RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("--- Training Information ---\n")
        f.write(f"Converged: {training_info.get('success', 'N/A')}\n")
        f.write(f"Final cost: {training_info.get('final_cost', 'N/A'):.4f}\n")
        f.write(f"Iterations: {training_info.get('iterations', 'N/A')}\n")
        f.write(f"Time: {training_info.get('time', 'N/A'):.2f}s\n\n")

        f.write("--- Evaluation Metrics ---\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.2%}\n")
        f.write(f"Precision: {metrics['precision']:.2%}\n")
        f.write(f"Recall:    {metrics['recall']:.2%}\n\n")

        f.write("--- Confusion Matrix ---\n")
        cm = metrics['confusion_matrix']
        f.write(f"TN: {cm[0,0]:<4} FP: {cm[0,1]:<4}\n")
        f.write(f"FN: {cm[1,0]:<4} TP: {cm[1,1]:<4}\n")

    print(f"Results saved: {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test script for visualization functions.
    """
    print("=== Utilities Test ===\n")

    # Generate test data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Test 1: Visualize dataset
    print("1. Dataset visualization")
    plot_dataset(X, y, title="Test Dataset")

    # Test 2: Metrics
    print("\n2. Metrics calculation")
    y_pred = np.random.randint(0, 2, size=100)
    metrics = calculate_metrics(y, y_pred)
    print_metrics(metrics)

    # Test 3: Training history
    print("\n3. Convergence visualization")
    history = {
        'iteration': list(range(1, 21)),
        'cost': [0.5 - i*0.02 for i in range(20)]
    }
    plot_training_history(history)

    # Test 4: Save results
    print("\n4. Save results")
    os.makedirs('results', exist_ok=True)
    training_info = {
        'success': True,
        'final_cost': 0.15,
        'iterations': 20,
        'time': 45.2
    }
    save_results(metrics, training_info, 'results/test_results.txt')

    print("\n✅ Tests completed!")
