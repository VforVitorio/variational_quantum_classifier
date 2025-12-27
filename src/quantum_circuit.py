"""
Parameterized Quantum Circuit for Variational Classification

This module implements the quantum circuit architecture used in the
Variational Quantum Classifier (VQC). The circuit combines classical data encoding
through parameterized rotations with trainable variational layers.

Circuit Architecture:
    1. Data Encoding: Maps coordinates (x,y) → quantum state |ψ⟩
       - Angle encoding: RX(π*x) on qubit 0, RY(π*y) on qubit 1

    2. Variational Layer: Trainable parameterized transformation
       - First layer: RY(θ₁), RY(θ₂) on each qubit
       - Entanglement: CNOT between qubits
       - Second layer: RX(θ₃), RX(θ₄) on each qubit

    3. Measurement: State collapse → binary class {0, 1}
       - Multiple shots to reduce statistical noise
       - Majority voting to determine final class

Functions:
    encode_data_point: Encodes a 2D point into quantum state
    variational_layer: Applies parameterized transformation
    build_circuit: Combines encoding + variational in complete circuit
    measure_circuit: Executes circuit and measures result
    predict_single_point: Predicts class for a point
    predict_batch: Predicts classes for multiple points

Usage example:
    >>> params = np.random.rand(4) * 2 * np.pi
    >>> prediction = predict_single_point(0.5, 0.7, params, shots=100)
    >>> print(f"Predicted class: {prediction}")
"""

import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import RX, RY, RZ, CNOT, MEASURE
from pyquil.quilbase import Declare
from typing import List, Tuple

# =============================================================================
# DATA ENCODING
# =============================================================================


def encode_data_point(x: float, y: float) -> Program:
    """
    Encodes a 2D point (x,y) into quantum state using angle encoding.

    Encoding strategy:
        - Applies RX(π * x) to qubit 0 → encodes x coordinate
        - Applies RY(π * y) to qubit 1 → encodes y coordinate

    The multiplication by π ensures that normalized coordinates [0,1]
    map to rotations in the range [0, π], allowing exploration of
    the entire qubit state space.

    Args:
        x: Normalized x coordinate in range [0, 1]
        y: Normalized y coordinate in range [0, 1]

    Returns:
        Program: PyQuil circuit with encoding operations applied

    Example:
        circuit = encode_data_point(0.5, 0.7)
        print(circuit)
            RX(1.5707963267948966) 0
            RY(2.199114857512855) 1

    Note:
        Coordinates must be normalized before calling this function.
        Values outside [0,1] will result in unexpected rotations.
    """
    program = Program()

    # Encode x on qubit 0 through rotation on X axis
    program += RX(2 * np.pi * x, 0)

    # Encode y on qubit 1 through rotation on Y axis
    program += RY(2 * np.pi * y, 1)

    return program


# =============================================================================
# VARIATIONAL LAYER
# =============================================================================

def variational_layer(params: np.ndarray, n_qubits: int = 2, n_layers: int = 1) -> Program:
    """
    Applies variational layers with trainable parameterized gates.

    Structure of each layer:
        1. First round of rotations: RY(θᵢ) on each qubit
        2. Entanglement: CNOT between adjacent qubits
        3. Second round of rotations: RX(θᵢ) on each qubit

    This architecture allows:
        - Individual Hilbert space exploration (rotations)
        - Quantum correlations between qubits (CNOT)
        - Greater model expressiveness

    Args:
        params: Parameter array [θ₁, θ₂, θ₃, θ₄, ...]
                For n_qubits=2, n_layers=1: 4 parameters
                For n_qubits=2, n_layers=2: 8 parameters
        n_qubits: Number of qubits in the circuit (default: 2)
        n_layers: Number of variational layers (default: 1)

    Returns:
        Program: Circuit with variational layers applied

    Example (1 layer):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        layer = variational_layer(params, n_qubits=2, n_layers=1)

    Example (2 layers):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        layer = variational_layer(params, n_qubits=2, n_layers=2)

    Note:
        The number of parameters must be 2 * n_qubits * n_layers.
    """
    program = Program()

    # Supports multiple variational layers for greater expressiveness
    # n_layers layers (e.g.: 2 layers = 8 parameters for 2 qubits)

    # Verify we have enough parameters
    expected_params = 2 * n_qubits * n_layers
    if len(params) < expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters (2 × {n_qubits} qubits × {n_layers} layers), "
            f"received {len(params)}")

    param_idx = 0

    # Apply each layer sequentially
    for layer in range(n_layers):
        # First layer: Individual RY rotations
        for i in range(n_qubits):
            program += RY(params[param_idx], i)
            param_idx += 1

        # Entanglement layer: CNOT between adjacent qubits
        for i in range(n_qubits - 1):
            program += CNOT(i, i + 1)

        # Second layer: Individual RX rotations
        for i in range(n_qubits):
            program += RX(params[param_idx], i)
            param_idx += 1
    # ========== END IMPROVEMENT ==========

    return program


# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

def build_circuit(x: float, y: float, params: np.ndarray, n_layers: int = 1) -> Program:
    """
    Builds complete quantum circuit combining encoding and variational layers.

    Circuit pipeline:
        1. Initialize qubits in state |00⟩
        2. Apply data encoding (x,y) → |ψ₀⟩
        3. Apply variational layers with parameters θ → |ψ(θ)⟩

    The resulting circuit implements the function:
        f(x, y; θ) = Measure[U_var(θ) · U_enc(x,y) · |00⟩]

    Args:
        x: x coordinate of point to classify [0, 1]
        y: y coordinate of point to classify [0, 1]
        params: Trainable parameters of variational layers
                Array of shape (4,) for 1 layer with 2 qubits
                Array of shape (8,) for 2 layers with 2 qubits
        n_layers: Number of variational layers (default: 1)

    Returns:
        Program: Complete quantum circuit without measurements

    Example (1 layer):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        circuit = build_circuit(0.5, 0.7, params, n_layers=1)

    Example (2 layers):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        circuit = build_circuit(0.5, 0.7, params, n_layers=2)

    Note:
        Measurements are not included here to allow flexibility
        in measurement type (simulation vs real hardware).
    """
    program = Program()

    # Step 1: Classical data encoding (improved with 2π)
    program += encode_data_point(x, y)

    # Step 2: Trainable variational layers (now supports multiple layers)
    program += variational_layer(params, n_qubits=2, n_layers=n_layers)

    return program


# =============================================================================
# MEASUREMENT
# =============================================================================

def measure_circuit(circuit: Program, n_qubits: int = 2, shots: int = 100) -> int:
    """
    Executes quantum circuit and returns predicted class via majority voting.

    Process:
        1. Adds classical memory declaration (readout)
        2. Adds MEASURE instructions to each qubit
        3. Executes the circuit 'shots' times
        4. Counts frequency of each result
        5. Returns most frequent class

    Majority voting reduces statistical noise inherent in
    quantum measurements, improving classifier stability.

    Args:
        circuit: Quantum circuit without measurements
        n_qubits: Number of qubits to measure (default: 2)
        shots: Number of circuit executions (default: 100)
               More shots → greater precision but more time

    Returns:
        int: Predicted class (0 or 1)
             - 0 if qubit 0 collapses mostly to |0⟩
             - 1 if qubit 0 collapses mostly to |1⟩

    Example:
        >>> circuit = build_circuit(0.5, 0.7, params)
        >>> prediction = measure_circuit(circuit, n_qubits=2, shots=100)
        >>> print(f"Class: {prediction}")
        Class: 1

    Note:
        This function uses simulator. For real hardware would require
        additional configuration with get_qc().
    """
    # Create program with measurements
    measurement_program = Program()

    # Declare classical memory to store results
    ro = measurement_program.declare('ro', 'BIT', n_qubits)

    # Add original circuit
    measurement_program += circuit

    # Add measurements
    for i in range(n_qubits):
        measurement_program += MEASURE(i, ro[i])

    # Wrap program for execution
    measurement_program.wrap_in_numshots_loop(shots)

    # Execute on simulator
    qc = get_qc(f'{n_qubits}q-qvm')
    executable = qc.compile(measurement_program)
    result = qc.run(executable)

    # Access data correctly in PyQuil 3.x
    measurements = result.readout_data.get('ro')

    # Combine both qubits for classification
    # Mapping: combined state = qubit_0 * 2 + qubit_1
    #   |00⟩ = 0  →  Class 0
    #   |01⟩ = 1  →  Class 0
    #   |10⟩ = 2  →  Class 1
    #   |11⟩ = 3  →  Class 1
    measurements_combined = measurements[:, 0] * 2 + measurements[:, 1]

    # Vote: states 0,1 → Class 0;  states 2,3 → Class 1
    votes_class_0 = np.sum((measurements_combined == 0)
                           | (measurements_combined == 1))
    votes_class_1 = np.sum((measurements_combined == 2)
                           | (measurements_combined == 3))

    # Return class with more votes
    predicted_class = 0 if votes_class_0 > votes_class_1 else 1
    # ========== END IMPROVEMENT ==========

    return predicted_class


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_point(x: float, y: float, params: np.ndarray, shots: int = 100, n_layers: int = 1) -> int:
    """
    Predicts class for a single point using the complete quantum circuit.

    Complete pipeline:
        1. Builds circuit with encoding + variational
        2. Executes and measures with majority voting
        3. Returns predicted class

    Args:
        x: Normalized x coordinate [0, 1]
        y: Normalized y coordinate [0, 1]
        params: Classifier parameters
                (4 values for 1 layer, 8 for 2 layers with 2 qubits)
        shots: Number of measurements for voting (default: 100)
        n_layers: Number of variational layers (default: 1)

    Returns:
        int: Predicted class (0 or 1)

    Example (1 layer):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        class_pred = predict_single_point(0.3, 0.7, params, shots=100, n_layers=1)

    Example (2 layers):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        class_pred = predict_single_point(0.3, 0.7, params, shots=100, n_layers=2)
    """
    # Build complete circuit with n layers
    circuit = build_circuit(x, y, params, n_layers=n_layers)

    # Measure and return class (now using both qubits)
    prediction = measure_circuit(circuit, n_qubits=2, shots=shots)

    return prediction


def predict_batch(X: np.ndarray, params: np.ndarray, shots: int = 100, n_layers: int = 1) -> np.ndarray:
    """
    Predicts classes for multiple points reusing a single QVM instance.

    Optimization: Instead of creating a new QVM for each point,
    creates a single instance and reuses it for all predictions.
    This eliminates compilation inconsistencies and improves performance.

    Args:
        X: Array of shape (n_samples, 2) with point coordinates
        params: Classifier parameters
        shots: Shots per prediction (default: 100)
        n_layers: Number of variational layers (default: 1)

    Returns:
        np.ndarray: Array of shape (n_samples,) with predictions {0, 1}

    Example (1 layer):
        X = np.array([[0.1, 0.2], [0.5, 0.7], [0.9, 0.3]])
        params = np.array([0.5, 1.2, 0.8, 2.1])
        predictions = predict_batch(X, params, shots=100, n_layers=1)

    Example (2 layers):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        predictions = predict_batch(X, params, shots=100, n_layers=2)

    Note:
        This refactored version reuses the QVM to avoid
        compilation inconsistencies between individual calls.
    """
    n_samples = X.shape[0]
    n_qubits = 2
    predictions = np.zeros(n_samples, dtype=int)

    # Create SINGLE QVM instance for all predictions
    qc = get_qc(f'{n_qubits}q-qvm')

    # Predict each point using the SAME QVM
    for i in range(n_samples):
        # Build circuit for this point
        circuit = build_circuit(X[i, 0], X[i, 1], params, n_layers=n_layers)

        # Create program with measurements
        measurement_program = Program()
        ro = measurement_program.declare('ro', 'BIT', n_qubits)
        measurement_program += circuit

        # Add measurements
        for q in range(n_qubits):
            measurement_program += MEASURE(q, ro[q])

        measurement_program.wrap_in_numshots_loop(shots)

        # Execute with the SAME QVM (reused)
        executable = qc.compile(measurement_program)
        result = qc.run(executable)

        # Process measurements
        measurements = result.readout_data.get('ro')

        # Combine both qubits for classification
        measurements_combined = measurements[:, 0] * 2 + measurements[:, 1]

        # Vote: states 0,1 → Class 0;  states 2,3 → Class 1
        votes_class_0 = np.sum((measurements_combined == 0) | (measurements_combined == 1))
        votes_class_1 = np.sum((measurements_combined == 2) | (measurements_combined == 3))

        # Return class with more votes
        predictions[i] = 0 if votes_class_0 > votes_class_1 else 1

    return predictions

# =============================================================================
# TESTING
# =============================================================================


if __name__ == "__main__":
    """
    Test script to verify quantum circuit functionality.
    Run: python src/quantum_circuit.py
    """
    print("=== Quantum Circuit Test ===\n")

    # Random parameters for testing
    params = np.random.rand(4) * 2 * np.pi
    print(f"Test parameters: {params}\n")

    # Test 1: Encoding
    print("1. Encoding Test:")
    circuit_enc = encode_data_point(0.5, 0.7)
    print(circuit_enc)

    # Test 2: Variational Layer
    print("\n2. Variational Layer Test:")
    circuit_var = variational_layer(params, n_qubits=2)
    print(circuit_var)

    # Test 3: Complete Circuit
    print("\n3. Complete Circuit Test:")
    circuit_full = build_circuit(0.5, 0.7, params)
    print(circuit_full)

    # Test 4: Single point prediction
    print("\n4. Prediction Test (single point):")
    x_test, y_test = 0.3, 0.8
    prediction = predict_single_point(x_test, y_test, params, shots=50)
    print(f"Point ({x_test}, {y_test}) → Class: {prediction}")

    # Test 5: Batch prediction
    print("\n5. Prediction Test (batch):")
    X_test = np.array([
        [0.1, 0.2],
        [0.5, 0.5],
        [0.9, 0.8]
    ])
    predictions = predict_batch(X_test, params, shots=50)
    print(f"Predictions: {predictions}")

    print("\n✅ All tests completed!")
