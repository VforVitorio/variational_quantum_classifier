"""
Variational Quantum Classifier (VQC)

This module implements the main quantum classifier class, combining
the parameterized quantum circuit with classical optimization for hybrid
machine learning.

Components:
    - QuantumClassifier: Main class that encapsulates the VQC model
    - Cost function: Calculates classification error
    - Optimization: Integration with scipy.optimize (COBYLA, Nelder-Mead)
    - Evaluation: Accuracy metrics and parameter persistence

Training algorithm:
    1. Initialize parameters randomly
    2. For each iteration:
        a. Predict classes with current parameters
        b. Calculate error (cost function)
        c. Optimizer adjusts parameters
    3. Converge when error stops decreasing

Functions:
    QuantumClassifier: Complete classifier class
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import time
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.quantum_circuit import predict_single_point, predict_batch
except ModuleNotFoundError:
    from quantum_circuit import predict_single_point, predict_batch


# =============================================================================
# QUANTUM CLASSIFIER CLASS
# =============================================================================

class QuantumClassifier:
    """
    Variational Quantum Classifier for binary classification.

    Implements a hybrid quantum-classical model that:
    - Encodes data in quantum states
    - Applies parameterized transformations
    - Optimizes parameters using classical algorithms

    Attributes:
        n_qubits: Number of qubits in the circuit
        n_params: Number of trainable parameters
        shots: Repetitions per measurement
        params: Current model parameters
        training_history: Training history
    """

    def __init__(self, n_qubits: int = 2, n_params: int = 4, shots: int = 100, n_layers: int = 1):
        """
        Initializes the quantum classifier.

        Args:
            n_qubits: Number of qubits (default: 2)
            n_params: Number of parameters (default: 4 for 1 layer, 8 for 2 layers)
            shots: Shots per measurement (default: 100)
            n_layers: Number of variational layers (default: 1)
        """
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.shots = shots
        # Support for multiple variational layers
        self.n_layers = n_layers

        # Initialize parameters randomly in [0, 2π]
        self.params = np.random.rand(n_params) * 2 * np.pi

        # Training history
        self.training_history = {
            'cost': [],
            'iteration': [],
            'time': []
        }

    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the cost function (classification error).

        The cost function is the percentage of incorrect predictions:
            Cost = (1/N) * Σ |y_pred - y_true|

        Uses batch predictions for better efficiency.

        Args:
            params: Current circuit parameters
            X: Features of shape (n_samples, 2)
            y: Labels of shape (n_samples,)

        Returns:
            float: Cost in range [0, 1] where 0 = perfect classification
        """
        # Use batch prediction for better efficiency
        predictions = predict_batch(X, params, self.shots, self.n_layers)

        # Calculate error percentage
        errors = np.sum(predictions != y)
        cost = errors / len(y)

        return cost

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              max_iter: int = 200,
              method: str = 'COBYLA',
              verbose: bool = True,
              patience: int = 20,
              min_delta: float = 1e-4) -> Dict:
        """
        Trains the classifier by optimizing the parameters.

        Uses scipy.optimize.minimize to find parameters that
        minimize the cost function with early stopping.

        Args:
            X: Training features (n_samples, 2)
            y: Training labels (n_samples,)
            max_iter: Maximum iterations (default: 200)
            method: Optimization algorithm (default: 'COBYLA')
                   Options: 'COBYLA', 'Nelder-Mead', 'Powell'
            verbose: Whether to show progress (default: True)
            patience: Iterations without improvement before stopping (default: 20)
            min_delta: Minimum improvement considered significant (default: 1e-4)

        Returns:
            dict: Training information
                - 'success': Whether it converged
                - 'final_cost': Final cost
                - 'iterations': Iterations performed
                - 'time': Total time
                - 'stopped_early': Whether early stopping was activated
        """
        if verbose:
            print(f"=== Quantum Classifier Training ===")
            print(f"Dataset: {X.shape[0]} points")
            print(f"Method: {method}")
            print(
                f"Variational layers: {self.n_layers} (Parameters: {self.n_params})")
            print(f"Max iterations: {max_iter}")
            print(
                f"Early stopping: patience={patience}, min_delta={min_delta}\n")

        start_time = time.time()

        # Variables for early stopping
        best_cost = float('inf')
        no_improvement_count = 0
        early_stopped = False

        # Callback to show progress and handle early stopping
        def callback(params):
            nonlocal best_cost, no_improvement_count, early_stopped

            cost = self._cost_function(params, X, y)
            self.training_history['cost'].append(cost)
            current_iter = len(self.training_history['cost'])
            self.training_history['iteration'].append(current_iter)
            elapsed_time = time.time() - start_time
            self.training_history['time'].append(elapsed_time)

            # Early stopping logic
            if cost < best_cost - min_delta:
                best_cost = cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # If no improvement for 'patience' iterations, stop
            if no_improvement_count >= patience:
                early_stopped = True
                if verbose:
                    print(
                        f"\n\n⚠️  Early stopping activated at iteration {current_iter}")
                    print(
                        f"No improvement in {patience} consecutive iterations")
                # Force scipy to stop
                raise StopIteration

            if verbose:
                # Calculate progress bar
                progress = current_iter / max_iter
                bar_length = 20
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                # Calculate estimated time
                avg_time_per_iter = elapsed_time / current_iter if current_iter > 0 else 0
                eta = avg_time_per_iter * (max_iter - current_iter)

                # Show bar (overwrite same line with \r)
                print(f"\rEpoch {current_iter}/{max_iter} [{bar}] "
                      f"loss: {cost:.4f} (best: {best_cost:.4f}) - "
                      f"ETA: {eta:.1f}s - no improvement: {no_improvement_count}/{patience}", end='', flush=True)

        # Optimization with early stopping handling
        try:
            result = minimize(
                fun=self._cost_function,
                x0=self.params,
                args=(X, y),
                method=method,
                options={'maxiter': max_iter},
                callback=callback
            )
        except StopIteration:
            # Early stopping activated - use last parameters
            result = type('Result', (), {
                'x': self.params,  # Keep current parameters
                'fun': best_cost,
                'success': True,
                'nit': len(self.training_history['cost']),
                'message': 'Early stopping'
            })()

        # Update optimal parameters
        self.params = result.x

        training_time = time.time() - start_time

        # Get number of iterations (not all optimizers return nit)
        iterations = getattr(result, 'nit', len(self.training_history['cost']))

        if verbose:
            # Line break after progress bar
            print("\n")
            print(f"=== Training Completed ===")
            if early_stopped:
                print(f"🛑 Stopped by early stopping")
            print(f"Converged: {result.success}")
            print(f"Final cost: {result.fun:.4f}")
            print(f"Iterations: {iterations}")
            print(f"Time: {training_time:.2f}s")

        return {
            'success': result.success,
            'final_cost': result.fun,
            'iterations': iterations,
            'time': training_time,
            'message': result.message,
            'stopped_early': early_stopped
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts classes for one or multiple points.

        Args:
            X: Features (n_samples, 2) or (2,) for a single point

        Returns:
            np.ndarray: Predictions (n_samples,) or int for a single point
        """
        # Handle single point (now with n_layers)
        if X.ndim == 1:
            return predict_single_point(X[0], X[1], self.params, self.shots, self.n_layers)

        # Handle batch (now with n_layers)
        return predict_batch(X, self.params, self.shots, self.n_layers)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates accuracy on a dataset.

        Args:
            X: Features (n_samples, 2)
            y: True labels (n_samples,)

        Returns:
            float: Accuracy in range [0, 1]
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def save_params(self, filepath: str) -> None:
        """
        Saves trained parameters to disk.

        Args:
            filepath: File path (e.g.: 'models/vqc_params.pkl')
        """
        data = {
            'params': self.params,
            'n_qubits': self.n_qubits,
            'n_params': self.n_params,
            'shots': self.shots,
            'n_layers': self.n_layers,  # Save n_layers for compatibility
            'training_history': self.training_history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Parameters saved to: {filepath}")

    def load_params(self, filepath: str) -> None:
        """
        Loads parameters from disk.

        Args:
            filepath: File path
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.params = data['params']
        self.n_qubits = data['n_qubits']
        self.n_params = data['n_params']
        self.shots = data['shots']
        # For compatibility with old models
        self.n_layers = data.get('n_layers', 1)
        self.training_history = data['training_history']

        print(f"Parameters loaded from: {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Classifier test script.
    Run: python src/classifier.py
    """
    print("=== Quantum Classifier Test ===\n")

    # Generate simple test data
    np.random.seed(42)
    X_train = np.random.rand(20, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

    print(f"Test dataset: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}\n")

    # Create and train classifier
    classifier = QuantumClassifier(n_qubits=2, n_params=4, shots=50)

    print("Initial accuracy:", classifier.evaluate(X_train, y_train))

    # Train
    result = classifier.train(X_train, y_train, max_iter=30, verbose=True)

    # Evaluate
    accuracy = classifier.evaluate(X_train, y_train)
    print(f"\nFinal accuracy: {accuracy:.2%}")

    # Test individual prediction
    test_point = np.array([0.3, 0.8])
    prediction = classifier.predict(test_point)
    print(f"\nPrediction for {test_point}: Class {prediction}")

    print("\n✅ Test completed!")
