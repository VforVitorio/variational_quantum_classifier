"""
Generador de Dataset de Espirales para Clasificador Cuántico Variacional

Este módulo genera un dataset de espirales entrelazadas para clasificación binaria.
Consiste en dos espirales que rotan en direcciones opuestas, creando un problema
no linealmente separable ideal para demostrar las capacidades de clasificadores cuánticos.

Formulación matemática:
    - Espiral A (clase 0): r = 2θ + π, θ ∈ [0, 2π]
    - Espiral B (clase 1): r = -2θ - π, θ ∈ [0, 2π]
    - Coordenadas: (x, y) = (r·cos(θ), r·sin(θ))

Funciones:
    generate_spiral_points: Generación de espirales usando coordenadas polares
    normalize_data: Escalar coordenadas al rango [0, 1]
    make_spiral_dataset: Pipeline principal (generar + normalizar)
    plot_spiral_dataset: Utilidad de visualización
    save_dataset: Persistir dataset a disco
    load_dataset: Cargar dataset desde disco
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def generate_spiral_points(n_points: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera dataset de espirales entrelazadas mediante transformación de coordenadas polares.

    Crea dos espirales con patrones de rotación opuestos. Los puntos se distribuyen
    uniformemente a lo largo del parámetro angular, con ruido gaussiano opcional.

    Args:
        n_points: Número total de puntos a generar (divididos equitativamente entre clases)
        noise: Desviación estándar del ruido gaussiano añadido (default: 0.1)

    Returns:
        X: Array de forma (n_points, 2) conteniendo coordenadas [x, y]
        y: Array de forma (n_points,) conteniendo etiquetas binarias [0, 1]

    Ejemplo:
        >>> X, y = generate_spiral_points(200, noise=0.05)
        >>> X.shape, y.shape
        ((200, 2), (200,))
    """
    n_per_class = n_points // 2

    # Generar ángulos uniformemente distribuidos
    theta = np.linspace(0, 2 * np.pi, n_per_class)

    # Espiral A (clase 0): rotación horaria
    r_a = 2 * theta + np.pi
    x_a = r_a * np.cos(theta)
    y_a = r_a * np.sin(theta)

    # Espiral B (clase 1): rotación antihoraria
    r_b = -2 * theta - np.pi
    x_b = r_b * np.cos(theta)
    y_b = r_b * np.sin(theta)

    # Combinar ambas espirales
    X = np.vstack([
        np.column_stack([x_a, y_a]),
        np.column_stack([x_b, y_b])
    ])

    # Añadir ruido gaussiano
    X += np.random.randn(*X.shape) * noise

    # Crear etiquetas
    y = np.hstack([
        np.zeros(n_per_class, dtype=int),
        np.ones(n_per_class, dtype=int)
    ])

    # Mezclar datos
    indices = np.random.permutation(n_points)
    X = X[indices]
    y = y[indices]

    return X, y


# =============================================================================
# PROCESAMIENTO DE DATOS
# =============================================================================

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normaliza coordenadas al rango [0, 1] usando escalado min-max.

    Esencial para encoding cuántico ya que los ángulos de rotación deben estar acotados.
    Aplica normalización independiente a las coordenadas x e y.

    Args:
        X: Array de forma (n_samples, 2) conteniendo coordenadas crudas

    Returns:
        X_norm: Array de forma (n_samples, 2) con coordenadas normalizadas en [0, 1]

    Fórmula:
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    Ejemplo:
        >>> X = np.array([[-1, -1], [0, 0], [1, 1]])
        >>> normalize_data(X)
        array([[0. , 0. ],
               [0.5, 0.5],
               [1. , 1. ]])
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Evitar división por cero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0

    X_norm = (X - X_min) / X_range

    return X_norm


# =============================================================================
# PERSISTENCIA
# =============================================================================

def save_dataset(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """
    Guarda dataset a disco en formato CSV.

    Args:
        X: Matriz de características de forma (n_samples, 2)
        y: Vector de etiquetas de forma (n_samples,)
        filepath: Ruta donde guardar el archivo (debe terminar en .csv)

    Ejemplo:
        >>> save_dataset(X, y, 'data/spiral_dataset.csv')
    """
    # Combinar X e y en una sola matriz
    data = np.column_stack([X, y])

    # Guardar con encabezados
    np.savetxt(filepath, data, delimiter=',', header='x,y,label',
               comments='', fmt='%.6f,%.6f,%d')
    print(f"Dataset guardado en: {filepath}")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga dataset desde disco.

    Args:
        filepath: Ruta al archivo .csv

    Returns:
        X: Matriz de características de forma (n_samples, 2)
        y: Vector de etiquetas de forma (n_samples,)

    Ejemplo:
        >>> X, y = load_dataset('data/spiral_dataset.csv')
    """
    # Cargar datos saltando la cabecera
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    # Separar características y etiquetas
    X = data[:, :2]
    y = data[:, 2].astype(int)

    print(f"Dataset cargado desde: {filepath}")
    print(f"Forma: X={X.shape}, y={y.shape}")
    return X, y


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def plot_spiral_dataset(X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Visualiza dataset de espirales con clases codificadas por color.

    Crea un scatter plot con:
        - Clase 0 (círculos rojos): Espiral exterior
        - Clase 1 (círculos azules): Espiral interior

    Args:
        X: Coordenadas de forma (n_samples, 2)
        y: Etiquetas de forma (n_samples,)
        save_path: Ruta opcional para guardar la figura (ej: 'results/spiral.png')

    Ejemplo:
        >>> plot_spiral_dataset(X, y, save_path='results/dataset.png')
    """
    plt.figure(figsize=(8, 8))

    # Plotear clase 0
    mask_0 = y == 0
    plt.scatter(X[mask_0, 0], X[mask_0, 1],
                c='red', marker='o', s=30, alpha=0.6,
                label='Clase 0', edgecolors='darkred')

    # Plotear clase 1
    mask_1 = y == 1
    plt.scatter(X[mask_1, 0], X[mask_1, 1],
                c='blue', marker='o', s=30, alpha=0.6,
                label='Clase 1', edgecolors='darkblue')

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Dataset de Espirales Entrelazadas',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")

    plt.show()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def make_spiral_dataset(n_points: int = 200,
                        noise: float = 0.1,
                        normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline completo para generar dataset de espirales normalizado.

    Esta es la función principal para usar externamente. Combina generación
    y normalización en una única llamada.

    Args:
        n_points: Número total de puntos (default: 200)
        noise: Nivel de ruido para generación de espirales (default: 0.1)
        normalize: Si aplicar normalización o no (default: True)

    Returns:
        X: Coordenadas normalizadas de forma (n_points, 2) en [0, 1]
        y: Etiquetas binarias de forma (n_points,)

    Ejemplo:
        >>> X_train, y_train = make_spiral_dataset(n_points=200, noise=0.1)
        >>> X_train.min(), X_train.max()
        (0.0, 1.0)
    """
    # Generar espirales
    X, y = generate_spiral_points(n_points, noise)

    # Normalizar si se solicita
    if normalize:
        X = normalize_data(X)

    print(f"Dataset generado: {n_points} puntos, {len(np.unique(y))} clases")
    print(f"Rango X: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
    print(f"Rango Y: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")

    return X, y


# =============================================================================
# SCRIPT DE PRUEBA
# =============================================================================

if __name__ == "__main__":
    """
    Script de prueba para verificar la generación del dataset.
    Ejecutar: python data/dataset_generator.py
    """
    print("=== Generando Dataset de Espirales ===\n")

    # Generar dataset
    X, y = make_spiral_dataset(n_points=400, noise=0.1, normalize=True)

    # Visualizar
    print("\n=== Visualizando Dataset ===")
    plot_spiral_dataset(X, y, save_path='results/spiral_dataset.png')

    # Guardar dataset
    print("\n=== Guardando Dataset ===")
    save_dataset(X, y, 'data/spiral_dataset.csv')

    # Verificar carga
    print("\n=== Verificando Carga ===")
    X_loaded, y_loaded = load_dataset('data/spiral_dataset.csv')

    print("\n✅ Todas las pruebas completadas exitosamente!")
