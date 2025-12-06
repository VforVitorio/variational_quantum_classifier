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
    program += RX(2 * np.pi * x, 0)

    # Se codifica y en qbit 1 mediante rotación en el eje Y
    program += RY(2 * np.pi * y, 1)

    return program


# =============================================================================
# VARIATIONAL LAYER
# =============================================================================

def variational_layer(params: np.ndarray, n_qubits: int = 2, n_layers: int = 1) -> Program:
    """
    Aplica capas variacionales con puertas parametrizadas entrenables.

    Estructura de cada capa:
        1. Primera ronda de rotaciones: RY(θᵢ) en cada qubit
        2. Entrelazamiento: CNOT entre qubits adyacentes
        3. Segunda ronda de rotaciones: RX(θᵢ) en cada qubit

    Esta arquitectura permite:
        - Exploración individual del espacio de Hilbert (rotaciones)
        - Correlaciones cuánticas entre qubits (CNOT)
        - Mayor expresividad del modelo

    Args:
        params: Array de parámetros [θ₁, θ₂, θ₃, θ₄, ...]
                Para n_qubits=2, n_layers=1: 4 parámetros
                Para n_qubits=2, n_layers=2: 8 parámetros
        n_qubits: Número de qubits en el circuito (default: 2)
        n_layers: Número de capas variacionales (default: 1)

    Returns:
        Program: Circuito con capas variacionales aplicadas

    Ejemplo (1 capa):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        layer = variational_layer(params, n_qubits=2, n_layers=1)

    Ejemplo (2 capas):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        layer = variational_layer(params, n_qubits=2, n_layers=2)

    Nota:
        El número de parámetros debe ser 2 * n_qubits * n_layers.
    """
    program = Program()

    # Soporta múltiples capas variacionales para mayor expresividad
    # n_layers capas (ej: 2 capas = 8 parámetros para 2 qubits)

    # Verificar que tenemos suficientes parámetros
    expected_params = 2 * n_qubits * n_layers
    if len(params) < expected_params:
        raise ValueError(
            f"Se esperan {expected_params} parámetros (2 × {n_qubits} qubits × {n_layers} layers), "
            f"se recibieron {len(params)}")

    param_idx = 0

    # Aplicar cada capa secuencialmente
    for layer in range(n_layers):
        # Primera capa: Rotaciones RY individuales
        for i in range(n_qubits):
            program += RY(params[param_idx], i)
            param_idx += 1

        # Capa de entrelazamiento: CNOT entre qubits adyacentes
        for i in range(n_qubits - 1):
            program += CNOT(i, i + 1)

        # Segunda capa: Rotaciones RX individuales
        for i in range(n_qubits):
            program += RX(params[param_idx], i)
            param_idx += 1
    # ========== FIN MEJORA ==========

    return program


# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

# =============================================================================
# CIRCUIT BUILDER
# =============================================================================

def build_circuit(x: float, y: float, params: np.ndarray, n_layers: int = 1) -> Program:
    """
    Construye circuito cuántico completo combinando encoding y capas variacionales.

    Pipeline del circuito:
        1. Inicializa qubits en estado |00⟩
        2. Aplica encoding de datos (x,y) → |ψ₀⟩
        3. Aplica capas variacionales con parámetros θ → |ψ(θ)⟩

    El circuito resultante implementa la función:
        f(x, y; θ) = Measure[U_var(θ) · U_enc(x,y) · |00⟩]

    Args:
        x: Coordenada x del punto a clasificar [0, 1]
        y: Coordenada y del punto a clasificar [0, 1]
        params: Parámetros entrenables de las capas variacionales
                Array de forma (4,) para 1 capa con 2 qubits
                Array de forma (8,) para 2 capas con 2 qubits
        n_layers: Número de capas variacionales (default: 1)

    Returns:
        Program: Circuito cuántico completo sin mediciones

    Ejemplo (1 capa):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        circuit = build_circuit(0.5, 0.7, params, n_layers=1)

    Ejemplo (2 capas):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        circuit = build_circuit(0.5, 0.7, params, n_layers=2)

    Nota:
        Las mediciones no se incluyen aquí para permitir flexibilidad
        en el tipo de medición (simulación vs hardware real).
    """
    program = Program()

    # Paso 1: Encoding de datos clásicos (mejorado con 2π)
    program += encode_data_point(x, y)

    # Paso 2: Capas variacionales entrenables (ahora soporta múltiples capas)
    program += variational_layer(params, n_qubits=2, n_layers=n_layers)

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

    # Combinar ambos qubits para clasificación
    # Mapeo: estado combinado = qubit_0 * 2 + qubit_1
    #   |00⟩ = 0  →  Clase 0
    #   |01⟩ = 1  →  Clase 0
    #   |10⟩ = 2  →  Clase 1
    #   |11⟩ = 3  →  Clase 1
    measurements_combined = measurements[:, 0] * 2 + measurements[:, 1]

    # Votar: estados 0,1 → Clase 0;  estados 2,3 → Clase 1
    votes_class_0 = np.sum((measurements_combined == 0)
                           | (measurements_combined == 1))
    votes_class_1 = np.sum((measurements_combined == 2)
                           | (measurements_combined == 3))

    # Retornar clase con más votos
    predicted_class = 0 if votes_class_0 > votes_class_1 else 1
    # ========== FIN MEJORA ==========

    return predicted_class


# =============================================================================
# PREDICTION
# =============================================================================

def predict_single_point(x: float, y: float, params: np.ndarray, shots: int = 100, n_layers: int = 1) -> int:
    """
    Predice clase para un único punto usando el circuito cuántico completo.

    Pipeline completo:
        1. Construye circuito con encoding + variational
        2. Ejecuta y mide con votación mayoritaria
        3. Retorna clase predicha

    Args:
        x: Coordenada x normalizada [0, 1]
        y: Coordenada y normalizada [0, 1]
        params: Parámetros del clasificador
                (4 valores para 1 capa, 8 para 2 capas con 2 qubits)
        shots: Número de mediciones para votación (default: 100)
        n_layers: Número de capas variacionales (default: 1)

    Returns:
        int: Clase predicha (0 o 1)

    Ejemplo (1 capa):
        params = np.array([0.5, 1.2, 0.8, 2.1])
        clase = predict_single_point(0.3, 0.7, params, shots=100, n_layers=1)

    Ejemplo (2 capas):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        clase = predict_single_point(0.3, 0.7, params, shots=100, n_layers=2)
    """
    # Construir circuito completo con n capas
    circuit = build_circuit(x, y, params, n_layers=n_layers)

    # Medir y retornar clase (ahora usando ambos qubits)
    prediction = measure_circuit(circuit, n_qubits=2, shots=shots)

    return prediction


def predict_batch(X: np.ndarray, params: np.ndarray, shots: int = 100, n_layers: int = 1) -> np.ndarray:
    """
    Predice clases para múltiples puntos reutilizando una única instancia QVM.

    Optimización: En lugar de crear una QVM nueva para cada punto,
    crea una sola instancia y la reutiliza para todas las predicciones.
    Esto elimina inconsistencias de compilación y mejora el rendimiento.

    Args:
        X: Array de forma (n_samples, 2) con coordenadas de puntos
        params: Parámetros del clasificador
        shots: Shots por predicción (default: 100)
        n_layers: Número de capas variacionales (default: 1)

    Returns:
        np.ndarray: Array de forma (n_samples,) con predicciones {0, 1}

    Ejemplo (1 capa):
        X = np.array([[0.1, 0.2], [0.5, 0.7], [0.9, 0.3]])
        params = np.array([0.5, 1.2, 0.8, 2.1])
        predictions = predict_batch(X, params, shots=100, n_layers=1)

    Ejemplo (2 capas):
        params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 2.3])
        predictions = predict_batch(X, params, shots=100, n_layers=2)

    Nota:
        Esta versión refactorizada reutiliza la QVM para evitar
        inconsistencias de compilación entre llamadas individuales.
    """
    n_samples = X.shape[0]
    n_qubits = 2
    predictions = np.zeros(n_samples, dtype=int)

    # Crear UNA SOLA instancia de QVM para todas las predicciones
    qc = get_qc(f'{n_qubits}q-qvm')

    # Predecir cada punto usando la MISMA QVM
    for i in range(n_samples):
        # Construir circuito para este punto
        circuit = build_circuit(X[i, 0], X[i, 1], params, n_layers=n_layers)

        # Crear programa con mediciones
        measurement_program = Program()
        ro = measurement_program.declare('ro', 'BIT', n_qubits)
        measurement_program += circuit

        # Añadir mediciones
        for q in range(n_qubits):
            measurement_program += MEASURE(q, ro[q])

        measurement_program.wrap_in_numshots_loop(shots)

        # Ejecutar con la MISMA QVM (reutilizada)
        executable = qc.compile(measurement_program)
        result = qc.run(executable)

        # Procesar mediciones
        measurements = result.readout_data.get('ro')

        # Combinar ambos qubits para clasificación
        measurements_combined = measurements[:, 0] * 2 + measurements[:, 1]

        # Votar: estados 0,1 → Clase 0;  estados 2,3 → Clase 1
        votes_class_0 = np.sum((measurements_combined == 0) | (measurements_combined == 1))
        votes_class_1 = np.sum((measurements_combined == 2) | (measurements_combined == 3))

        # Retornar clase con más votos
        predictions[i] = 0 if votes_class_0 > votes_class_1 else 1

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
