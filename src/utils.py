"""
Utilidades de Visualización y Métricas

Funciones auxiliares para evaluar y visualizar resultados del clasificador cuántico.

Funciones:
    plot_dataset: Visualizar puntos del dataset
    plot_decision_boundary: Frontera de decisión del clasificador
    calculate_metrics: Accuracy, precision, recall
    plot_training_history: Evolución del costo durante entrenamiento
    save_results: Guardar métricas en archivo de texto
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Optional, Dict, Tuple
import os


# =============================================================================
# VISUALIZACIÓN DE DATOS
# =============================================================================

def plot_dataset(X: np.ndarray, y: np.ndarray,
                 title: str = "Dataset",
                 save_path: Optional[str] = None) -> None:
    """
    Visualiza dataset con clases codificadas por color.

    Args:
        X: Coordenadas (n_samples, 2)
        y: Labels (n_samples,)
        title: Título del gráfico
        save_path: Ruta para guardar (opcional)
    """
    plt.figure(figsize=(8, 8))

    # Clase 0
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=50, alpha=0.7,
                label='Clase 0', edgecolors='darkred', linewidths=1.5)

    # Clase 1
    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=50, alpha=0.7,
                label='Clase 1', edgecolors='darkblue', linewidths=1.5)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")

    plt.show()


def plot_decision_boundary(classifier, X: np.ndarray, y: np.ndarray,
                           resolution: int = 100,
                           title: str = "Frontera de Decisión",
                           save_path: Optional[str] = None) -> None:
    """
    Visualiza frontera de decisión del clasificador.

    Args:
        classifier: Instancia de QuantumClassifier entrenado
        X: Coordenadas (n_samples, 2)
        y: Labels (n_samples,)
        resolution: Resolución de la malla (default: 100)
        title: Título del gráfico
        save_path: Ruta para guardar
    """
    # Crear malla de puntos
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predecir clase para cada punto de la malla
    print(
        f"Generando frontera de decisión ({resolution}x{resolution} puntos)...")
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Visualizar
    plt.figure(figsize=(10, 8))

    # Contorno de la frontera
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu', levels=1)
    plt.contour(xx, yy, Z, colors='black', linewidths=2, levels=1)

    # Datos de entrenamiento
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=50, alpha=0.8,
                label='Clase 0', edgecolors='darkred', linewidths=1.5)

    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=50, alpha=0.8,
                label='Clase 1', edgecolors='darkblue', linewidths=1.5)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")

    plt.show()


# =============================================================================
# MÉTRICAS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de clasificación.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones

    Returns:
        dict: Métricas (accuracy, precision, recall)
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
    Imprime métricas formateadas.

    Args:
        metrics: Diccionario de métricas
    """
    print("\n=== Métricas de Clasificación ===")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print("\nMatriz de Confusión:")
    print(metrics['confusion_matrix'])


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualiza evolución del costo durante entrenamiento.

    Args:
        history: Diccionario con 'cost' e 'iteration'
        save_path: Ruta para guardar
    """
    if not history['cost']:
        print("No hay historial de entrenamiento.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['iteration'], history['cost'],
             'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Costo (Error)', fontsize=12)
    plt.title('Convergencia del Entrenamiento', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")

    plt.show()


# =============================================================================
# PERSISTENCIA
# =============================================================================

def save_results(metrics: Dict, training_info: Dict, filepath: str) -> None:
    """
    Guarda métricas y resultados en archivo de texto.

    Args:
        metrics: Métricas de evaluación
        training_info: Información del entrenamiento
        filepath: Ruta del archivo
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("RESULTADOS DEL CLASIFICADOR CUÁNTICO VARIACIONAL\n")
        f.write("=" * 50 + "\n\n")

        f.write("--- Información de Entrenamiento ---\n")
        f.write(f"Convergió: {training_info.get('success', 'N/A')}\n")
        f.write(f"Costo final: {training_info.get('final_cost', 'N/A'):.4f}\n")
        f.write(f"Iteraciones: {training_info.get('iterations', 'N/A')}\n")
        f.write(f"Tiempo: {training_info.get('time', 'N/A'):.2f}s\n\n")

        f.write("--- Métricas de Evaluación ---\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.2%}\n")
        f.write(f"Precision: {metrics['precision']:.2%}\n")
        f.write(f"Recall:    {metrics['recall']:.2%}\n\n")

        f.write("--- Matriz de Confusión ---\n")
        cm = metrics['confusion_matrix']
        f.write(f"TN: {cm[0,0]:<4} FP: {cm[0,1]:<4}\n")
        f.write(f"FN: {cm[1,0]:<4} TP: {cm[1,1]:<4}\n")

    print(f"Resultados guardados: {filepath}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Script de prueba de funciones de visualización.
    """
    print("=== Prueba de Utilidades ===\n")

    # Generar datos de prueba
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Test 1: Visualizar dataset
    print("1. Visualización de dataset")
    plot_dataset(X, y, title="Dataset de Prueba")

    # Test 2: Métricas
    print("\n2. Cálculo de métricas")
    y_pred = np.random.randint(0, 2, size=100)
    metrics = calculate_metrics(y, y_pred)
    print_metrics(metrics)

    # Test 3: Historial de entrenamiento
    print("\n3. Visualización de convergencia")
    history = {
        'iteration': list(range(1, 21)),
        'cost': [0.5 - i*0.02 for i in range(20)]
    }
    plot_training_history(history)

    # Test 4: Guardar resultados
    print("\n4. Guardar resultados")
    os.makedirs('results', exist_ok=True)
    training_info = {
        'success': True,
        'final_cost': 0.15,
        'iterations': 20,
        'time': 45.2
    }
    save_results(metrics, training_info, 'results/test_results.txt')

    print("\n✅ Pruebas completadas!")
