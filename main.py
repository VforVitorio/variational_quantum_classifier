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

    # Configuración del dataset (ajustar según necesidad)
    # n_points: 50 (pruebas rápidas), 100 (estándar), 400 (producción)
    X, y = make_spiral_dataset(n_points=100, noise=0.1, normalize=True)

    print(f"Dataset generado: {X.shape[0]} puntos")
    print(f"Distribución: Clase 0={np.sum(y==0)}, Clase 1={np.sum(y==1)}")

    # Visualizar dataset
    plot_dataset(X, y,
                 title="Dataset de Espirales Entrelazadas",
                 save_path="results/dataset.png")

    # =========================================================================
    # 2. ENTRENAMIENTO CON MÚLTIPLES INTENTOS
    # =========================================================================
    print("\n[2/5] Entrenando clasificador (múltiples intentos)...")

    # Número de intentos de entrenamiento (selecciona el mejor)
    # 1: rápido, 3: estándar, 5-10: producción
    n_attempts = 3

    print(f"Intentos de entrenamiento: {n_attempts}\n")

    best_accuracy = 0
    best_classifier = None
    best_training_result = None

    for attempt in range(n_attempts):
        print(f"--- Intento {attempt + 1}/{n_attempts} ---")

        # Configuración del clasificador cuántico
        # 2 capas variacionales para mejor expresividad
        # 100 shots por medición para balance ruido/velocidad
        classifier = QuantumClassifier(
            n_qubits=2,
            n_params=8,      # 2 capas × 2 qubits × 2 rotaciones
            shots=100,
            n_layers=2
        )

        # Entrenar con COBYLA (optimizador libre de gradientes)
        # max_iter: ~10× número de parámetros es una buena regla
        training_result = classifier.train(
            X, y,
            max_iter=80,
            method='COBYLA',
            verbose=True
        )

        # Evaluar
        accuracy = classifier.evaluate(X, y)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Costo final: {training_result['final_cost']:.4f}")
        print(f"Tiempo: {training_result['time']:.1f}s\n")

        # Guardar si es el mejor
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = classifier
            best_training_result = training_result
            print("★ Nuevo mejor modelo guardado!\n")

    print(f"Mejor accuracy obtenida: {best_accuracy:.2%}")

    # =========================================================================
    # 3. EVALUACIÓN DETALLADA
    # =========================================================================
    print("\n[3/5] Evaluando mejor modelo...")

    # Predicciones
    y_pred = best_classifier.predict(X)

    # Calcular métricas
    metrics = calculate_metrics(y, y_pred)
    print_metrics(metrics)

    # =========================================================================
    # 4. VISUALIZACIÓN DE RESULTADOS
    # =========================================================================
    print("\n[4/5] Generando visualizaciones...")

    # Visualización de frontera de decisión
    # resolution: 30 (rápido), 40 (estándar), 100 (alta calidad)
    plot_decision_boundary(
        best_classifier, X, y,
        resolution=40,
        title="Frontera de Decisión Aprendida",
        save_path="results/decision_boundary.png"
    )

    # Convergencia del mejor modelo
    if best_classifier.training_history['cost']:
        plot_training_history(
            best_classifier.training_history,
            save_path="results/training_convergence.png"
        )

    # =========================================================================
    # 5. GUARDAR RESULTADOS
    # =========================================================================
    print("\n[5/5] Guardando resultados...")

    # Guardar métricas
    save_results(metrics, best_training_result, "results/metrics.txt")

    # Guardar parámetros del mejor modelo
    best_classifier.save_params("results/best_model_params.pkl")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Intentos de entrenamiento: {n_attempts}")
    print(f"Mejor Accuracy: {best_accuracy:.2%}")
    print(f"Tiempo mejor modelo: {best_training_result['time']:.2f}s")
    print(f"Iteraciones: {best_training_result['iterations']}")
    print("\nArchivos generados:")
    print("  - results/dataset.png")
    print("  - results/decision_boundary.png")
    print("  - results/training_convergence.png")
    print("  - results/metrics.txt")
    print("  - results/best_model_params.pkl")
    print("\nPipeline completado exitosamente!")
    print("\nNota: Para datasets más grandes (400 puntos), considerar:")
    print("  - Aumentar max_iter a 100-200")
    print("  - Usar más attempts (5-10)")
    print("  - Optimizar shots según precisión deseada")


if __name__ == "__main__":
    # Crear directorio de resultados
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Ejecutar pipeline
    main()
