"""
Script Principal - Demo del Clasificador Cuántico Variacional

Ejecuta el pipeline completo:
    1. Carga dataset de espirales
    2. Entrena clasificador cuántico
    3. Evalúa performance
    4. Genera visualizaciones

Uso:
    python main.py
"""

from src.utils import (plot_dataset, plot_decision_boundary,
                       calculate_metrics, print_metrics,
                       plot_training_history, save_results)
from src.classifier import QuantumClassifier
from data.dataset_generator import make_spiral_dataset, load_dataset
import numpy as np
import sys
import os

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))


def main():
    """Pipeline principal del clasificador cuántico."""

    print("=" * 60)
    print("CLASIFICADOR CUÁNTICO VARIACIONAL (VQC)")
    print("=" * 60)

    # =========================================================================
    # 1. GENERACIÓN/CARGA DE DATOS
    # =========================================================================
    print("\n[1/5] Generando dataset de espirales...")

    # Generar dataset
    X, y = make_spiral_dataset(n_points=200, noise=0.1, normalize=True)

    print(f"✓ Dataset: {X.shape[0]} puntos")
    print(f"✓ Distribución: Clase 0={np.sum(y==0)}, Clase 1={np.sum(y==1)}")

    # Visualizar dataset
    plot_dataset(X, y,
                 title="Dataset de Espirales Entrelazadas",
                 save_path="results/dataset.png")

    # =========================================================================
    # 2. INICIALIZACIÓN DEL CLASIFICADOR
    # =========================================================================
    print("\n[2/5] Inicializando clasificador cuántico...")

    classifier = QuantumClassifier(
        n_qubits=2,
        n_params=4,
        shots=50
    )

    print("✓ Circuito: 2 qubits")
    print("✓ Parámetros: 4 (θ₁, θ₂, θ₃, θ₄)")
    print("✓ Shots: 50")

    # =========================================================================
    # 3. ENTRENAMIENTO
    # =========================================================================
    print("\n[3/5] Entrenando clasificador...")

    training_result = classifier.train(
        X, y,
        max_iter=50,
        method='COBYLA',
        verbose=True
    )

    # =========================================================================
    # 4. EVALUACIÓN
    # =========================================================================
    print("\n[4/5] Evaluando modelo...")

    # Predicciones
    y_pred = classifier.predict(X)

    # Calcular métricas
    metrics = calculate_metrics(y, y_pred)
    print_metrics(metrics)

    # =========================================================================
    # 5. VISUALIZACIÓN DE RESULTADOS
    # =========================================================================
    print("\n[5/5] Generando visualizaciones...")

    # Frontera de decisión
    plot_decision_boundary(
        classifier, X, y,
        resolution=50,  # Reducido para velocidad
        title="Frontera de Decisión Aprendida",
        save_path="results/decision_boundary.png"
    )

    # Convergencia
    if classifier.training_history['cost']:
        plot_training_history(
            classifier.training_history,
            save_path="results/training_convergence.png"
        )

    # Guardar resultados
    save_results(metrics, training_result, "results/metrics.txt")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Accuracy Final: {metrics['accuracy']:.2%}")
    print(f"Tiempo de Entrenamiento: {training_result['time']:.2f}s")
    print(f"Iteraciones: {training_result['iterations']}")
    print("\nArchivos generados:")
    print("  - results/dataset.png")
    print("  - results/decision_boundary.png")
    print("  - results/training_convergence.png")
    print("  - results/metrics.txt")
    print("\n✅ Pipeline completado exitosamente!")


if __name__ == "__main__":
    # Crear directorio de resultados
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Ejecutar pipeline
    main()
