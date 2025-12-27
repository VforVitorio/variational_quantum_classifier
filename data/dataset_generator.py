"""
Spiral Dataset Generator for Variational Quantum Classifier

This module generates an intertwined spiral dataset for binary classification.
Consists of two spirals rotating in opposite directions, creating a non-linearly
separable problem ideal for demonstrating quantum classifier capabilities.

Mathematical formulation:
    - Spiral A (class 0): r = 2θ + π, θ ∈ [0, 2π]
    - Spiral B (class 1): r = -2θ - π, θ ∈ [0, 2π]
    - Coordinates: (x, y) = (r·cos(θ), r·sin(θ))

Functions:
    generate_spiral_points: Spiral generation using polar coordinates
    normalize_data: Scale coordinates to range [0, 1]
    make_spiral_dataset: Main pipeline (generate + normalize)
    plot_spiral_dataset: Visualization utility
    save_dataset: Persist dataset to disk
    load_dataset: Load dataset from disk
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def generate_spiral_points(n_points: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates intertwined spiral dataset through polar coordinate transformation.

    Creates two spirals with opposite rotation patterns. Points are distributed
    uniformly along the angular parameter, with optional Gaussian noise.

    Args:
        n_points: Total number of points to generate (divided equally between classes)
        noise: Standard deviation of added Gaussian noise (default: 0.1)

    Returns:
        X: Array of shape (n_points, 2) containing coordinates [x, y]
        y: Array of shape (n_points,) containing binary labels [0, 1]

    Example:
        >>> X, y = generate_spiral_points(200, noise=0.05)
        >>> X.shape, y.shape
        ((200, 2), (200,))
    """
    n_per_class = n_points // 2

    # Generate uniformly distributed angles
    theta = np.linspace(0, 2 * np.pi, n_per_class)

    # Spiral A (class 0): clockwise rotation
    r_a = 2 * theta + np.pi
    x_a = r_a * np.cos(theta)
    y_a = r_a * np.sin(theta)

    # Spiral B (class 1): counterclockwise rotation
    r_b = -2 * theta - np.pi
    x_b = r_b * np.cos(theta)
    y_b = r_b * np.sin(theta)

    # Combine both spirals
    X = np.vstack([
        np.column_stack([x_a, y_a]),
        np.column_stack([x_b, y_b])
    ])

    # Add Gaussian noise
    X += np.random.randn(*X.shape) * noise

    # Create labels
    y = np.hstack([
        np.zeros(n_per_class, dtype=int),
        np.ones(n_per_class, dtype=int)
    ])

    # Shuffle data
    indices = np.random.permutation(n_points)
    X = X[indices]
    y = y[indices]

    return X, y


# =============================================================================
# DATA PROCESSING
# =============================================================================

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalizes coordinates to range [0, 1] using min-max scaling.

    Essential for quantum encoding since rotation angles must be bounded.
    Applies independent normalization to x and y coordinates.

    Args:
        X: Array of shape (n_samples, 2) containing raw coordinates

    Returns:
        X_norm: Array of shape (n_samples, 2) with normalized coordinates in [0, 1]

    Formula:
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    Example:
        >>> X = np.array([[-1, -1], [0, 0], [1, 1]])
        >>> normalize_data(X)
        array([[0. , 0. ],
               [0.5, 0.5],
               [1. , 1. ]])
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X_norm = (X - X_min) / X_range

    return X_norm


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """
    Saves dataset to disk in CSV format.

    Args:
        X: Feature matrix of shape (n_samples, 2)
        y: Label vector of shape (n_samples,)
        filepath: Path where to save the file (must end in .csv)

    Example:
        >>> save_dataset(X, y, 'data/spiral_dataset.csv')
    """
    # Combine X and y in a single matrix
    data = np.column_stack([X, y])

    # Save with headers
    np.savetxt(filepath, data, delimiter=',', header='x,y,label',
               comments='', fmt='%.6f,%.6f,%d')
    print(f"Dataset saved to: {filepath}")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset from disk.

    Args:
        filepath: Path to .csv file

    Returns:
        X: Feature matrix of shape (n_samples, 2)
        y: Label vector of shape (n_samples,)

    Example:
        >>> X, y = load_dataset('data/spiral_dataset.csv')
    """
    # Load data skipping header
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    # Separate features and labels
    X = data[:, :2]
    y = data[:, 2].astype(int)

    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: X={X.shape}, y={y.shape}")
    return X, y


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_spiral_dataset(X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Visualizes spiral dataset with classes color-coded.

    Creates a scatter plot with:
        - Class 0 (red circles): Outer spiral
        - Class 1 (blue circles): Inner spiral

    Args:
        X: Coordinates of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
        save_path: Optional path to save the figure (e.g.: 'results/spiral.png')

    Example:
        >>> plot_spiral_dataset(X, y, save_path='results/dataset.png')
    """
    plt.figure(figsize=(8, 8))

    # Plot class 0
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=30, alpha=0.6,
                label='Class 0', edgecolors='darkred')

    # Plot class 1
    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=30, alpha=0.6,
                label='Class 1', edgecolors='darkblue')

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Intertwined Spiral Dataset',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def make_spiral_dataset(n_points: int = 200,
                        noise: float = 0.1,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline to generate normalized spiral dataset.

    This is the main function for external use. Combines generation
    and normalization in a single call.

    Args:
        n_points: Total number of points (default: 200)
        noise: Noise level for spiral generation (default: 0.1)
        normalize: Whether to apply normalization or not (default: True)

    Returns:
        X: Normalized coordinates of shape (n_points, 2) in [0, 1]
        y: Binary labels of shape (n_points,)

    Example:
        >>> X_train, y_train = make_spiral_dataset(n_points=200, noise=0.1)
        >>> X_train.min(), X_train.max()
        (0.0, 1.0)
    """
    # Generate spirals
    X, y = generate_spiral_points(n_points, noise)

    # Normalize if requested
    if normalize:
        X = normalize_data(X)

    print(f"Dataset generated: {n_points} points, {len(np.unique(y))} classes")
    print(f"X range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"Y range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")

    return X, y


# =============================================================================
# TEST SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Test script to verify dataset generation.
    Run: python data/dataset_generator.py
    """
    print("=== Generating Spiral Dataset ===\n")

    # Generate dataset
    X, y = make_spiral_dataset(n_points=400, noise=0.1, normalize=True)

    # Visualize
    print("\n=== Visualizing Dataset ===")
    plot_spiral_dataset(X, y, save_path='results/spiral_dataset.png')

    # Save dataset
    print("\n=== Saving Dataset ===")
    save_dataset(X, y, 'data/spiral_dataset.csv')

    # Verify loading
    print("\n=== Verifying Loading ===")
    X_loaded, y_loaded = load_dataset('data/spiral_dataset.csv')

    print("\n✅ All tests completed successfully!")
