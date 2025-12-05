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
import threading
from queue import Queue
import time

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))


def train_attempt_thread(thread_id, X, y, seed, n_params, shots, n_layers, max_iter, results_queue):
    """
    Entrena un clasificador en un thread separado.
    Thread-safe siguiendo guía de PyQuil multithreading.

    Args:
        thread_id: ID del thread
        X, y: Datos de entrenamiento
        seed: Semilla aleatoria
        n_params, shots, n_layers: Configuración del clasificador
        max_iter: Iteraciones de optimización
        results_queue: Queue thread-safe para resultados
    """
    try:
        # Seed único por thread
        np.random.seed(seed)

        # Crear clasificador (instancia independiente por thread)
        classifier = QuantumClassifier(
            n_qubits=2,
            n_params=n_params,
            shots=shots,
            n_layers=n_layers
        )

        # Entrenar (verbose=False para no mezclar outputs entre threads)
        start_time = time.time()
        training_result = classifier.train(
            X, y,
            max_iter=max_iter,
            method='COBYLA',
            verbose=False
        )
        training_time = time.time() - start_time

        # Evaluar
        accuracy = classifier.evaluate(X, y)

        # Guardar resultado en queue thread-safe
        results_queue.put({
            'thread_id': thread_id,
            'accuracy': accuracy,
            'classifier': classifier,
            'training_result': training_result,
            'time': training_time
        })

    except Exception as e:
        # Capturar errores para no bloquear otros threads
        results_queue.put({
            'thread_id': thread_id,
            'error': str(e),
            'accuracy': 0.0
        })


def train_parallel(X, y, n_attempts=3, n_params=8, shots=100, n_layers=2, max_iter=80):
    """
    Entrena múltiples intentos en paralelo usando threading.

    Returns:
        tuple: (mejor_classifier, mejor_training_result, mejor_accuracy, resultados)
    """
    print(f"Entrenamiento paralelo: {n_attempts} threads simultáneos\n")

    results_queue = Queue()
    threads = []
    seeds = [42, 123, 456, 789, 101112][:n_attempts]  # Semillas distintas

    # Crear y lanzar threads
    start_time = time.time()
    for i, seed in enumerate(seeds):
        thread = threading.Thread(
            target=train_attempt_thread,
            args=(i, X, y, seed, n_params, shots,
                  n_layers, max_iter, results_queue),
            name=f"Attempt-{i+1}"
        )
        threads.append(thread)
        thread.start()
        print(f"→ Intento {i+1}/{n_attempts} iniciado (Thread-{i+1})")

    print(f"\nTodos los threads activos. Esperando resultados...\n")

    # Esperar a que todos terminen
    for i, thread in enumerate(threads):
        thread.join()
        print(f"✓ Intento {i+1} completado")

    total_time = time.time() - start_time
    print(
        f"\nTiempo total paralelo: {total_time:.1f}s ({total_time/60:.1f} min)\n")

    # Recolectar resultados
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # Filtrar errores y ordenar por accuracy
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        raise RuntimeError("Todos los intentos fallaron")

    results_sorted = sorted(
        valid_results, key=lambda r: r['accuracy'], reverse=True)

    # Mostrar resumen
    print("Resumen de intentos:")
    for r in results_sorted:
        print(f"  Intento {r['thread_id']+1}: Accuracy={r['accuracy']:.2%}, "
              f"Tiempo={r['time']:.1f}s")

    # Retornar mejor
    best = results_sorted[0]
    return best['classifier'], best['training_result'], best['accuracy'], results_sorted


def main():
    """Pipeline principal del clasificador cuántico."""

    print("=" * 60)
    print("CLASIFICADOR CUÁNTICO VARIACIONAL (VQC)")
    print("=" * 60)

    # =========================================================================
    # 1. GENERACIÓN/CARGA DE DATOS
    # =========================================================================
    print("\n[1/5] Generando dataset de espirales...")

    # Configuración del dataset
    # n_points: 50 (pruebas rápidas), 150 (estándar), 400 (producción)
    X, y = make_spiral_dataset(n_points=150, noise=0.1, normalize=True)

    print(f"Dataset generado: {X.shape[0]} puntos")
    print(f"Distribución: Clase 0={np.sum(y==0)}, Clase 1={np.sum(y==1)}")

    # =========================================================================
    # SPLIT TRAIN/VALIDATION (80/20)
    # =========================================================================
    # Usar semilla fija para reproducibilidad
    np.random.seed(42)

    # Crear índices aleatorios
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"\nSplit train/validation (80/20):")
    print(
        f"  Train: {len(X_train)} puntos (Clase 0={np.sum(y_train==0)}, Clase 1={np.sum(y_train==1)})")
    print(
        f"  Val:   {len(X_val)} puntos (Clase 0={np.sum(y_val==0)}, Clase 1={np.sum(y_val==1)})")

    # Visualizar dataset completo
    plot_dataset(X, y,
                 title="Dataset de Espirales Entrelazadas (150 puntos)",
                 save_path="results/dataset.png")

    # =========================================================================
    # 2. ENTRENAMIENTO CON MÚLTIPLES INTENTOS
    # =========================================================================
    print("\n[2/5] Entrenando clasificador (múltiples intentos)...")

    # Configuración
    n_attempts = 3  # 1: rápido, 3: estándar, 5-10: producción
    USE_THREADING = False  # Desactivado (GIL no permite paralelismo real)

    # Parámetros del clasificador
    n_params = 8      # 2 capas × 2 qubits × 2 rotaciones
    shots = 100
    n_layers = 2
    max_iter = 80

    if USE_THREADING:
        # Versión paralela (no recomendada - GIL issue)
        best_classifier, best_training_result, best_accuracy, all_results = train_parallel(
            X_train, y_train,
            n_attempts=n_attempts,
            n_params=n_params,
            shots=shots,
            n_layers=n_layers,
            max_iter=max_iter
        )
    else:
        # Versión secuencial (recomendada)
        print(f"Intentos de entrenamiento: {n_attempts} (secuencial)\n")
        best_accuracy = 0
        best_val_accuracy = 0
        best_classifier = None
        best_training_result = None

        for attempt in range(n_attempts):
            print(f"--- Intento {attempt + 1}/{n_attempts} ---")

            classifier = QuantumClassifier(
                n_qubits=2,
                n_params=n_params,
                shots=shots,
                n_layers=n_layers
            )

            # Entrenar solo con datos de entrenamiento
            training_result = classifier.train(
                X_train, y_train,
                max_iter=max_iter,
                # method='COBYLA',  # Anterior: oscilaciones fuertes, convergencia lenta
                method='SLSQP',     # Nuevo: convergencia más suave y rápida
                verbose=True,
                patience=30,        # Ajustado de 20 (era muy estricto)
                # Ajustado de 1e-4 (ignoraba oscilaciones naturales)
                min_delta=0.003
            )

            # Evaluar en train y validation
            train_accuracy = classifier.evaluate(X_train, y_train)
            val_accuracy = classifier.evaluate(X_val, y_val)

            print(f"\nTrain Accuracy: {train_accuracy:.2%}")
            print(f"Val Accuracy:   {val_accuracy:.2%}")
            print(f"Costo final: {training_result['final_cost']:.4f}")
            print(f"Tiempo: {training_result['time']:.1f}s")

            # Calcular overfitting gap
            overfit_gap = train_accuracy - val_accuracy
            if overfit_gap > 0.1:
                print(f"⚠️  Posible overfitting (gap: {overfit_gap:.2%})")
            print()

            # Seleccionar mejor modelo basado en VALIDATION accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_accuracy = train_accuracy
                best_classifier = classifier
                best_training_result = training_result
                print("★ Nuevo mejor modelo guardado (basado en val accuracy)!\n")

    print(f"\n★ Mejor Train Accuracy: {best_accuracy:.2%}")
    print(f"★ Mejor Val Accuracy:   {best_val_accuracy:.2%}")

    # =========================================================================
    # 3. EVALUACIÓN DETALLADA
    # =========================================================================
    print("\n[3/5] Evaluando mejor modelo...")

    # Evaluar en ambos conjuntos
    print("\nEvaluación en Train:")
    y_pred_train = best_classifier.predict(X_train)
    metrics_train = calculate_metrics(y_train, y_pred_train)
    print_metrics(metrics_train)

    print("\n" + "-" * 40)
    print("Evaluación en Validation:")
    y_pred_val = best_classifier.predict(X_val)
    metrics_val = calculate_metrics(y_val, y_pred_val)
    print_metrics(metrics_val)

    # Métricas finales (usar validación para reporte)
    final_metrics = metrics_val

    # =========================================================================
    # 4. VISUALIZACIÓN DE RESULTADOS
    # =========================================================================
    print("\n[4/5] Generando visualizaciones...")

    # Visualización de frontera de decisión (con todos los datos)
    # resolution: 30 (rápido), 40 (estándar), 100 (alta calidad)
    plot_decision_boundary(
        best_classifier, X, y,
        resolution=40,
        title="Frontera de Decisión Aprendida (Dataset Completo)",
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

    # Guardar métricas (validación)
    save_results(final_metrics, best_training_result, "results/metrics.txt")

    # Guardar parámetros del mejor modelo
    best_classifier.save_params("results/best_model_params.pkl")

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(
        f"Dataset: {len(X)} puntos (Train: {len(X_train)}, Val: {len(X_val)})")
    print(f"Intentos de entrenamiento: {n_attempts}")
    print(f"Mejor Train Accuracy: {best_accuracy:.2%}")
    print(f"Mejor Val Accuracy:   {best_val_accuracy:.2%}")
    print(f"Overfitting gap:      {(best_accuracy - best_val_accuracy):.2%}")
    print(f"Tiempo mejor modelo: {best_training_result['time']:.2f}s")
    print(f"Iteraciones: {best_training_result['iterations']}")
    if best_training_result.get('stopped_early'):
        print(f"🛑 Early stopping activado")
    print("\nArchivos generados:")
    print("  - results/dataset.png")
    print("  - results/decision_boundary.png")
    print("  - results/training_convergence.png")
    print("  - results/metrics.txt (validation metrics)")
    print("  - results/best_model_params.pkl")
    print("\nPipeline completado exitosamente!")
    print("\nNota: El modelo se seleccionó usando validation accuracy")
    print("      para evitar overfitting.")


if __name__ == "__main__":
    # Crear directorio de resultados
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Ejecutar pipeline
    main()
