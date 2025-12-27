"""
Script to regenerate visualizations with better quality
without needing to retrain the model.

Uses the saved model and increases shots + resolution
to obtain smoother decision boundaries.
"""

import numpy as np
import pickle
from src.classifier import QuantumClassifier
from src.utils import plot_decision_boundary
from data.dataset_generator import make_spiral_dataset


def main():
    print("=" * 70)
    print("HIGH QUALITY VISUALIZATION REGENERATION")
    print("=" * 70)

    # 1. Load data (same configuration as training)
    print("\n[1/3] Loading dataset...")
    X, y = make_spiral_dataset(n_points=100, noise=0.1, normalize=True)
    print(f"Dataset loaded: {X.shape[0]} points")

    # 2. Load saved model
    print("\n[2/3] Loading trained model...")
    try:
        with open('results/best_model_params.pkl', 'rb') as f:
            data = pickle.load(f)

        # The pkl saves a complete dictionary, not just the parameters
        best_params = data['params']
        n_params = data['n_params']
        n_layers = data['n_layers']

        print(f"Model loaded:")
        print(f"  - Parameters: {n_params} values")
        print(f"  - Layers: {n_layers}")
        print(f"  - Array shape: {best_params.shape}")

        # Recreate classifier with more shots for smooth visualization
        classifier = QuantumClassifier(
            n_qubits=2,
            n_params=n_params,
            shots=300,  # Increased from 100 to reduce noise
            n_layers=n_layers
        )
        classifier.params = best_params
        print(
            f"Classifier ready: shots={classifier.shots}, layers={n_layers}")

    except FileNotFoundError:
        print("Error: results/best_model_params.pkl not found")
        print("You must train the model first (run main.py)")
        return

    # 3. Regenerate visualization with high quality
    print("\n[3/3] Regenerating decision boundary...")
    print("High quality configuration:")
    print(f"  - Resolution: 60×60 = 3600 points (vs 1600 previous)")
    print(f"  - Shots: 300 per point (vs 100 previous)")
    print(f"  - Estimated: ~8-12 minutes")
    print("\nGenerating...")

    plot_decision_boundary(
        classifier, X, y,
        resolution=60,  # Higher resolution for better detail
        title="Decision Boundary (High Quality: 300 shots, 60×60 res)",
        save_path="results/decision_boundary_high_quality.png"
    )

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print("\nGenerated file:")
    print("  - results/decision_boundary_high_quality.png")
    print("\nCompare with:")
    print("  - results/decision_boundary.png (original)")
    print("\nImprovements with 300 shots and resolution 60:")
    print("  - Smoother boundary")
    print("  - Less pixelation")
    print("  - Reduced quantum noise")
    print("=" * 70)


if __name__ == "__main__":
    main()
