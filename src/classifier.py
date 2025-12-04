"""
Clasificador Cuántico Variacional (VQC)

Este módulo implementa la clase principal del clasificador cuántico, combinando
el circuito cuántico parametrizado con optimización clásica para aprendizaje
automático híbrido.

Componentes:
    - QuantumClassifier: Clase principal que encapsula el modelo VQC
    - Función de costo: Calcula error de clasificación
    - Optimización: Integración con scipy.optimize (COBYLA, Nelder-Mead)
    - Evaluación: Métricas de accuracy y persistencia de parámetros

Algoritmo de entrenamiento:
    1. Inicializar parámetros aleatoriamente
    2. Para cada iteración:
        a. Predecir clases con parámetros actuales
        b. Calcular error (función de costo)
        c. Optimizador ajusta parámetros
    3. Converger cuando error deja de disminuir

Funciones:
    QuantumClassifier: Clase del clasificador completo
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict
import time
import pickle
import sys
import os

# Añadir directorio padre al path para imports
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
    Clasificador Cuántico Variacional para clasificación binaria.

    Implementa un modelo híbrido cuántico-clásico que:
    - Codifica datos en estados cuánticos
    - Aplica transformaciones parametrizadas
    - Optimiza parámetros mediante algoritmos clásicos

    Attributes:
        n_qubits: Número de qubits en el circuito
        n_params: Número de parámetros entrenables
        shots: Repeticiones por medición
        params: Parámetros actuales del modelo
        training_history: Historial de entrenamiento
    """

    def __init__(self, n_qubits: int = 2, n_params: int = 4, shots: int = 100, n_layers: int = 1):
        """
        Inicializa el clasificador cuántico.

        Args:
            n_qubits: Número de qubits (default: 2)
            n_params: Número de parámetros (default: 4 para 1 capa, 8 para 2 capas)
            shots: Shots por medición (default: 100)
            n_layers: Número de capas variacionales (default: 1)
        """
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.shots = shots
        # Soporte para múltiples capas variacionales
        self.n_layers = n_layers

        # Inicializar parámetros aleatoriamente en [0, 2π]
        self.params = np.random.rand(n_params) * 2 * np.pi

        # Historial de entrenamiento
        self.training_history = {
            'cost': [],
            'iteration': [],
            'time': []
        }

    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la función de costo (error de clasificación).

        La función de costo es el porcentaje de predicciones incorrectas:
            Cost = (1/N) * Σ |y_pred - y_true|

        Args:
            params: Parámetros actuales del circuito
            X: Features de forma (n_samples, 2)
            y: Labels de forma (n_samples,)

        Returns:
            float: Costo en rango [0, 1] donde 0 = clasificación perfecta
        """
        n_samples = X.shape[0]
        errors = 0

        # Predecir cada punto y contar errores
        for i in range(n_samples):
            prediction = predict_single_point(
                X[i, 0], X[i, 1], params, self.shots, self.n_layers)
            if prediction != y[i]:
                errors += 1

        # Calcular porcentaje de error
        cost = errors / n_samples

        return cost

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              max_iter: int = 200,
              method: str = 'COBYLA',
              verbose: bool = True) -> Dict:
        """
        Entrena el clasificador optimizando los parámetros.

        Utiliza scipy.optimize.minimize para encontrar parámetros que
        minimizan la función de costo.

        Args:
            X: Features de entrenamiento (n_samples, 2)
            y: Labels de entrenamiento (n_samples,)
            max_iter: Máximo de iteraciones (default: 200)
            method: Algoritmo de optimización (default: 'COBYLA')
                   Opciones: 'COBYLA', 'Nelder-Mead', 'Powell'
            verbose: Si mostrar progreso (default: True)

        Returns:
            dict: Información del entrenamiento
                - 'success': Si convergió
                - 'final_cost': Costo final
                - 'iterations': Iteraciones realizadas
                - 'time': Tiempo total
        """
        if verbose:
            print(f"=== Entrenamiento del Clasificador Cuántico ===")
            print(f"Dataset: {X.shape[0]} puntos")
            print(f"Método: {method}")
            print(f"Capas variacionales: {self.n_layers} (Parámetros: {self.n_params})")
            print(f"Máx iteraciones: {max_iter}\n")

        start_time = time.time()

        # Callback para mostrar progreso durante entrenamiento
        def callback(params):
            cost = self._cost_function(params, X, y)
            self.training_history['cost'].append(cost)
            current_iter = len(self.training_history['cost'])
            self.training_history['iteration'].append(current_iter)
            elapsed_time = time.time() - start_time
            self.training_history['time'].append(elapsed_time)

            if verbose:
                # Calcular barra de progreso
                progress = current_iter / max_iter
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                # Calcular tiempo estimado
                avg_time_per_iter = elapsed_time / current_iter if current_iter > 0 else 0
                eta = avg_time_per_iter * (max_iter - current_iter)

                # Mostrar barra (sobreescribir misma línea con \r)
                print(f"\rÉpoca {current_iter}/{max_iter} [{bar}] "
                      f"- loss: {cost:.4f} - ETA: {eta:.1f}s", end='', flush=True)

        # Optimización
        result = minimize(
            fun=self._cost_function,
            x0=self.params,
            args=(X, y),
            method=method,
            options={'maxiter': max_iter},
            callback=callback
        )

        # Actualizar parámetros óptimos
        self.params = result.x

        training_time = time.time() - start_time

        # Obtener número de iteraciones (no todos los optimizadores devuelven nit)
        iterations = getattr(result, 'nit', len(self.training_history['cost']))

        if verbose:
            # Salto de línea después de la barra de progreso
            print("\n")
            print(f"=== Entrenamiento Completado ===")
            print(f"Convergió: {result.success}")
            print(f"Costo final: {result.fun:.4f}")
            print(f"Iteraciones: {iterations}")
            print(f"Tiempo: {training_time:.2f}s")

        return {
            'success': result.success,
            'final_cost': result.fun,
            'iterations': iterations,
            'time': training_time,
            'message': result.message
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases para uno o múltiples puntos.

        Args:
            X: Features (n_samples, 2) o (2,) para un punto

        Returns:
            np.ndarray: Predicciones (n_samples,) o int para un punto
        """
        # Manejar single point (ahora con n_layers)
        if X.ndim == 1:
            return predict_single_point(X[0], X[1], self.params, self.shots, self.n_layers)

        # Manejar batch (ahora con n_layers)
        return predict_batch(X, self.params, self.shots, self.n_layers)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula accuracy en un conjunto de datos.

        Args:
            X: Features (n_samples, 2)
            y: Labels verdaderos (n_samples,)

        Returns:
            float: Accuracy en rango [0, 1]
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def save_params(self, filepath: str) -> None:
        """
        Guarda parámetros entrenados a disco.

        Args:
            filepath: Ruta del archivo (ej: 'models/vqc_params.pkl')
        """
        data = {
            'params': self.params,
            'n_qubits': self.n_qubits,
            'n_params': self.n_params,
            'shots': self.shots,
            'n_layers': self.n_layers,  # Guardar n_layers para compatibilidad
            'training_history': self.training_history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Parámetros guardados en: {filepath}")

    def load_params(self, filepath: str) -> None:
        """
        Carga parámetros desde disco.

        Args:
            filepath: Ruta del archivo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.params = data['params']
        self.n_qubits = data['n_qubits']
        self.n_params = data['n_params']
        self.shots = data['shots']
        self.n_layers = data.get('n_layers', 1)  # Por compatibilidad con modelos antiguos
        self.training_history = data['training_history']

        print(f"Parámetros cargados desde: {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Script de prueba del clasificador.
    Ejecutar: python src/classifier.py
    """
    print("=== Prueba del Clasificador Cuántico ===\n")

    # Generar datos de prueba simple
    np.random.seed(42)
    X_train = np.random.rand(20, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

    print(f"Dataset de prueba: {X_train.shape}")
    print(f"Distribución clases: {np.bincount(y_train)}\n")

    # Crear y entrenar clasificador
    classifier = QuantumClassifier(n_qubits=2, n_params=4, shots=50)

    print("Accuracy inicial:", classifier.evaluate(X_train, y_train))

    # Entrenar
    result = classifier.train(X_train, y_train, max_iter=30, verbose=True)

    # Evaluar
    accuracy = classifier.evaluate(X_train, y_train)
    print(f"\nAccuracy final: {accuracy:.2%}")

    # Probar predicción individual
    test_point = np.array([0.3, 0.8])
    prediction = classifier.predict(test_point)
    print(f"\nPredicción para {test_point}: Clase {prediction}")

    print("\n✅ Prueba completada!")
