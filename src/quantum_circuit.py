"""
Circuito Cuántico Parametrizado para Clasificación Variacional

Este módulo implementa la arquitectura del circuito cuántico utilizado en el
Clasificador Cuántico Variacional (VQC). El circuito combina encoding de datos
clásicos mediante rotaciones parametrizadas con capas variacionales entrenables.

Arquitectura del Circuito:
    1. Data Encoding: Mapea coordenadas (x,y) → estado cuántico |ψ⟩
       - Angle encoding: RX(π*x) en qubit 0, RY(π*y) en qubit 1
    
    2. Variational Layer: Transformación parametrizada entrenable
       - Primera capa: RY(θ₁), RY(θ₂) en cada qubit
       - Entrelazamiento: CNOT entre qubits
       - Segunda capa: RX(θ₃), RX(θ₄) en cada qubit
    
    3. Measurement: Colapso del estado → clase binaria {0, 1}
       - Múltiples shots para reducir ruido estadístico
       - Votación mayoritaria para determinar clase final

Funciones:
    encode_data_point: Codifica un punto 2D en estado cuántico
    variational_layer: Aplica transformación parametrizada
    build_circuit: Combina encoding + variational en circuito completo
    measure_circuit: Ejecuta circuito y mide resultado
    predict_single_point: Predice clase para un punto
    predict_batch: Predice clases para múltiples puntos

Ejemplo de uso:
    >>> params = np.random.rand(4) * 2 * np.pi
    >>> prediction = predict_single_point(0.5, 0.7, params, shots=100)
    >>> print(f"Clase predicha: {prediction}")
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
    Codifica un punto 2D (x,y) en estado cuántico mediante angle encoding.

    Estrategia de codificación:
        - Aplica RX(π * x) al qubit 0 → codifica coordenada x
        - Aplica RY(π * y) al qubit 1 → codifica coordenada y

    La multiplicación por π garantiza que coordenadas normalizadas [0,1]
    se mapeen a rotaciones en el rango [0, π], permitiendo explorar
    todo el espacio de estados del qubit.

    Args:
        x: Coordenada x normalizada en rango [0, 1]
        y: Coordenada y normalizada en rango [0, 1]

    Returns:
        Program: Circuito PyQuil con operaciones de encoding aplicadas

    Ejemplo:
        circuit = encode_data_point(0.5, 0.7)
        print(circuit)
            RX(1.5707963267948966) 0
            RY(2.199114857512855) 1

    Nota:
        Las coordenadas deben estar normalizadas antes de llamar esta función.
        Valores fuera de [0,1] resultarán en rotaciones inesperadas.
    """
    program = Program()

    # Se codifica x en qbit 0 mediante rotación en el eje X
    program += RX(np.pi * x, 0)

    # Se codifica y en qbit 1 mediante rotación en el eje Y
    program += RY(np.pi * y, 1)

    return program


# =============================================================================
# VARIATIONAL LAYER
# =============================================================================

# =============================================================================
# VARIATIONAL LAYER
# =============================================================================

def variational_layer(params: np.ndarray, n_qubits: int = 2) -> Program:
    """
    Aplica capa variacional con puertas parametrizadas entrenables.

    Estructura de la capa:
        1. Primera ronda de rotaciones: RY(θᵢ) en cada qubit
        2. Entrelazamiento: CNOT entre qubits adyacentes
        3. Segunda ronda de rotaciones: RX(θᵢ) en cada qubit

    Esta arquitectura permite:
        - Exploración individual del espacio de Hilbert (rotaciones)
        - Correlaciones cuánticas entre qubits (CNOT)
        - Mayor expresividad del modelo

    Args:
        params: Array de parámetros [θ₁, θ₂, θ₃, θ₄, ...]
                Para n_qubits=2 se esperan 4 parámetros:
                - params[0], params[1]: rotaciones RY
                - params[2], params[3]: rotaciones RX
        n_qubits: Número de qubits en el circuito (default: 2)

    Returns:
        Program: Circuito con capa variacional aplicada

    Ejemplo:
        params = np.array([0.5, 1.2, 0.8, 2.1])
        layer = variational_layer(params, n_qubits=2)
        print(layer)
            RY(0.5) 0
            RY(1.2) 1
            CNOT 0 1
            RX(0.8) 0
            RX(2.1) 1

    Nota:
        El número de parámetros debe ser 2 * n_qubits.
        Para circuitos más profundos, esta función puede llamarse
        múltiples veces con diferentes conjuntos de parámetros.
    """
    program = Program()

    # Verificar que tenemos suficientes parámetros
    expected_params = 2 * n_qubits
    if len(params) < expected_params:
        raise ValueError(
            f"Se esperan {expected_params} parámetros, se recibieron {len(params)}")

    # Primera capa: Rotaciones RY individuales
    for i in range(n_qubits):
        program += RY(params[i], i)

    # Capa de entrelazamiento: CNOT entre qubits adyacentes
    for i in range(n_qubits - 1):
        program += CNOT(i, i + 1)

    # Segunda capa: Rotaciones RX individuales
    for i in range(n_qubits):
        program += RX(params[n_qubits + i], i)

    return program


# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

def build_circuit(x: float, y: float, params: np.ndarray) -> Program:
    """
    Construye circuito cuántico completo combinando encoding y capa variacional.

    Pipeline del circuito:
        1. Inicializa qubits en estado |00⟩
        2. Aplica encoding de datos (x,y) → |ψ₀⟩
        3. Aplica capa variacional con parámetros θ → |ψ(θ)⟩

    El circuito resultante implementa la función:
        f(x, y; θ) = Measure[U_var(θ) · U_enc(x,y) · |00⟩]

    Args:
        x: Coordenada x del punto a clasificar [0, 1]
        y: Coordenada y del punto a clasificar [0, 1]
        params: Parámetros entrenables de la capa variacional
                Array de forma (4,) para 2 qubits

    Returns:
        Program: Circuito cuántico completo sin mediciones

    Ejemplo:
        x, y = 0.5, 0.7
        params = np.array([0.5, 1.2, 0.8, 2.1])
        circuit = build_circuit(x, y, params)
        print(circuit)
            RX(1.5708) 0
            RY(2.1991) 1
            RY(0.5) 0
            RY(1.2) 1
            CNOT 0 1
            RX(0.8) 0
            RX(2.1) 1

    Nota:
        Las mediciones no se incluyen aquí para permitir flexibilidad
        en el tipo de medición (simulación vs hardware real).
    """
    program = Program()

    # Paso 1: Encoding de datos clásicos
    program += encode_data_point(x, y)

    # Paso 2: Capa variacional entrenable
    program += variational_layer(params, n_qubits=2)

    return program


# =============================================================================
# MEASUREMENT
# =============================================================================

def measure_circuit(circuit: Program, n_qubits: int = 2, shots: int = 100) -> int:
    """
    Ejecuta circuito cuántico y retorna clase predicha mediante votación mayoritaria.

    Proceso:
        1. Añade declaración de memoria clásica (readout)
        2. Añade instrucciones MEASURE a cada qubit
        3. Ejecuta el circuito 'shots' veces
        4. Cuenta frecuencia de cada resultado
        5. Retorna clase más frecuente

    La votación mayoritaria reduce el ruido estadístico inherente a las
    mediciones cuánticas, mejorando la estabilidad del clasificador.

    Args:
        circuit: Circuito cuántico sin mediciones
        n_qubits: Número de qubits a medir (default: 2)
        shots: Número de ejecuciones del circuito (default: 100)
               Más shots → mayor precisión pero mayor tiempo

    Returns:
        int: Clase predicha (0 o 1)
             - 0 si el qubit 0 colapsa mayoritariamente a |0⟩
             - 1 si el qubit 0 colapsa mayoritariamente a |1⟩

    Ejemplo:
        >>> circuit = build_circuit(0.5, 0.7, params)
        >>> prediction = measure_circuit(circuit, n_qubits=2, shots=100)
        >>> print(f"Clase: {prediction}")
        Clase: 1

    Nota:
        Esta función usa simulador. Para hardware real se requeriría
        configuración adicional con get_qc().
    """
    # Crear programa con mediciones
    measurement_program = Program()

    # Declarar memoria clásica para almacenar resultados
    ro = measurement_program.declare('ro', 'BIT', n_qubits)

    # Añadir el circuito original
    measurement_program += circuit

    # Añadir mediciones
    for i in range(n_qubits):
        measurement_program += MEASURE(i, ro[i])

    # Wrap del programa para ejecución
    measurement_program.wrap_in_numshots_loop(shots)

    # Ejecutar en simulador
    qc = get_qc(f'{n_qubits}q-qvm')
    executable = qc.compile(measurement_program)
    result = qc.run(executable)

    # Acceder a los datos correctamente en PyQuil 3.x
    measurements = result.readout_data.get('ro')

    # Contar frecuencias del qubit 0 (qubit clasificador)
    measurements_q0 = measurements[:, 0]
    count_0 = np.sum(measurements_q0 == 0)
    count_1 = np.sum(measurements_q0 == 1)

    # Retornar clase mayoritaria
    predicted_class = 0 if count_0 > count_1 else 1

    return predicted_class


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_point(x: float, y: float, params: np.ndarray, shots: int = 100) -> int:
    """
    Predice clase para un único punto usando el circuito cuántico completo.

    Pipeline completo:
        1. Construye circuito con encoding + variational
        2. Ejecuta y mide con votación mayoritaria
        3. Retorna clase predicha

    Args:
        x: Coordenada x normalizada [0, 1]
        y: Coordenada y normalizada [0, 1]
        params: Parámetros del clasificador (4 valores para 2 qubits)
        shots: Número de mediciones para votación (default: 100)

    Returns:
        int: Clase predicha (0 o 1)

    Ejemplo:
        params = np.array([0.5, 1.2, 0.8, 2.1])
        clase = predict_single_point(0.3, 0.7, params, shots=100)
        print(f"Predicción: {clase}")
            Predicción: 1
    """
    # Construir circuito completo
    circuit = build_circuit(x, y, params)

    # Medir y retornar clase
    prediction = measure_circuit(circuit, n_qubits=2, shots=shots)

    return prediction


def predict_batch(X: np.ndarray, params: np.ndarray, shots: int = 100) -> np.ndarray:
    """
    Predice clases para múltiples puntos de forma vectorizada.

    Itera sobre cada punto del dataset y aplica predict_single_point.
    Útil para evaluación del modelo en conjunto de entrenamiento/test.

    Args:
        X: Array de forma (n_samples, 2) con coordenadas de puntos
        params: Parámetros del clasificador
        shots: Shots por predicción

    Returns:
        np.ndarray: Array de forma (n_samples,) con predicciones {0, 1}

    Ejemplo:
        X = np.array([[0.1, 0.2], [0.5, 0.7], [0.9, 0.3]])
        params = np.array([0.5, 1.2, 0.8, 2.1])
        predictions = predict_batch(X, params, shots=100)
        print(predictions)
            [0 1 1]

    Nota:
        Para datasets grandes, esta función puede ser lenta debido a
        la ejecución individual de cada circuito. Optimizaciones futuras
        podrían paralelizar las ejecuciones.
    """
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)

    # Predecir cada punto individualmente
    for i in range(n_samples):
        predictions[i] = predict_single_point(X[i, 0], X[i, 1], params, shots)

    return predictions

# =============================================================================
# TESTING
# =============================================================================


if __name__ == "__main__":
    """
    Script de prueba para verificar funcionamiento del circuito cuántico.
    Ejecutar: python src/quantum_circuit.py
    """
    print("=== Prueba de Circuito Cuántico ===\n")

    # Parámetros aleatorios para prueba
    params = np.random.rand(4) * 2 * np.pi
    print(f"Parámetros de prueba: {params}\n")

    # Prueba 1: Encoding
    print("1. Test de Encoding:")
    circuit_enc = encode_data_point(0.5, 0.7)
    print(circuit_enc)

    # Prueba 2: Capa Variacional
    print("\n2. Test de Capa Variacional:")
    circuit_var = variational_layer(params, n_qubits=2)
    print(circuit_var)

    # Prueba 3: Circuito Completo
    print("\n3. Test de Circuito Completo:")
    circuit_full = build_circuit(0.5, 0.7, params)
    print(circuit_full)

    # Prueba 4: Predicción de un punto
    print("\n4. Test de Predicción (single point):")
    x_test, y_test = 0.3, 0.8
    prediction = predict_single_point(x_test, y_test, params, shots=50)
    print(f"Punto ({x_test}, {y_test}) → Clase: {prediction}")

    # Prueba 5: Predicción batch
    print("\n5. Test de Predicción (batch):")
    X_test = np.array([
        [0.1, 0.2],
        [0.5, 0.5],
        [0.9, 0.8]
    ])
    predictions = predict_batch(X_test, params, shots=50)
    print(f"Predicciones: {predictions}")

    print("\n✅ Todas las pruebas completadas!")
