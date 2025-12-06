"""
Test: Verificación de predict_batch vs predict_single_point

Comprueba que predict_batch devuelve exactamente los mismos resultados
que llamar a predict_single_point individualmente para cada punto.

Esto es crítico porque el cambio a batch predictions podría haber
introducido un bug que explique la degradación del 82% → 53% accuracy.

Uso:
    python test_batch_vs_single.py
"""

import numpy as np
import sys
import os

# Añadir directorio raíz al path (tests/ -> raíz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.quantum_circuit import predict_single_point, predict_batch
from data.dataset_generator import make_spiral_dataset


def test_batch_vs_single_predictions(X, params, shots, n_layers):
    """
    Compara las predicciones de batch vs single para todos los puntos.

    Returns:
        tuple: (all_match, differences, batch_preds, single_preds)
    """
    print(f"Comparando predicciones para {len(X)} puntos...")
    print(f"  Parámetros: {len(params)} params")
    print(f"  Shots: {shots}")
    print(f"  Layers: {n_layers}\n")

    # Predicciones con batch
    print("Ejecutando predict_batch...")
    batch_preds = predict_batch(X, params, shots, n_layers)

    # Predicciones individuales
    print("Ejecutando predict_single_point para cada punto...")
    single_preds = np.array([
        predict_single_point(X[i, 0], X[i, 1], params, shots, n_layers)
        for i in range(len(X))
    ])

    # Comparar
    matches = batch_preds == single_preds
    all_match = np.all(matches)

    differences = []
    for i in range(len(X)):
        if batch_preds[i] != single_preds[i]:
            differences.append({
                'index': i,
                'point': X[i],
                'batch_pred': batch_preds[i],
                'single_pred': single_preds[i]
            })

    return all_match, differences, batch_preds, single_preds


def test_with_multiple_trials(X, params, shots, n_layers, n_trials=5):
    """
    Ejecuta múltiples trials para detectar inconsistencias estocásticas.

    Con mediciones cuánticas, las predicciones pueden variar entre ejecuciones
    debido al ruido de shot. Esto verifica si batch y single tienen la misma
    distribución de resultados.
    """
    print(f"\nEjecutando {n_trials} trials para análisis estadístico...")

    batch_results = []
    single_results = []

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='', flush=True)

        batch_preds = predict_batch(X, params, shots, n_layers)
        single_preds = np.array([
            predict_single_point(X[i, 0], X[i, 1], params, shots, n_layers)
            for i in range(len(X))
        ])

        batch_results.append(batch_preds)
        single_results.append(single_preds)

        print(" OK")

    # Analizar consistencia
    batch_results = np.array(batch_results)  # (n_trials, n_points)
    single_results = np.array(single_results)

    # Para cada punto, calcular proporción de veces que predice clase 1
    batch_class1_rate = np.mean(batch_results, axis=0)  # (n_points,)
    single_class1_rate = np.mean(single_results, axis=0)

    # Calcular diferencias
    rate_differences = np.abs(batch_class1_rate - single_class1_rate)

    return batch_class1_rate, single_class1_rate, rate_differences


if __name__ == "__main__":
    print("="*60)
    print("TEST: Verificación predict_batch vs predict_single_point")
    print("="*60)

    # Generar dataset pequeño de prueba
    print("\n[1/3] Generando dataset de prueba...")
    np.random.seed(42)
    X_test, y_test = make_spiral_dataset(n_points=30, noise=0.1, normalize=True)
    print(f"Dataset: {len(X_test)} puntos")

    # Parámetros de prueba (aleatorios pero fijos)
    n_params = 8
    n_layers = 2
    shots = 100

    params = np.random.rand(n_params) * 2 * np.pi
    print(f"Parámetros aleatorios: {n_params} valores")
    print(f"Configuración: {n_layers} layers, {shots} shots")

    # Test 1: Comparación directa
    print("\n" + "="*60)
    print("[2/3] TEST 1: Comparación Directa (1 ejecución)")
    print("="*60)

    all_match, differences, batch_preds, single_preds = test_batch_vs_single_predictions(
        X_test, params, shots, n_layers
    )

    if all_match:
        print("\n✅ TODAS las predicciones coinciden!")
        print(f"   {len(X_test)}/{len(X_test)} puntos predichos idénticamente")
    else:
        print(f"\n❌ DISCREPANCIAS DETECTADAS!")
        print(f"   {len(differences)}/{len(X_test)} puntos difieren")
        print(f"\nDetalles de las discrepancias:")
        for diff in differences[:10]:  # Mostrar primeras 10
            print(f"  Punto {diff['index']}: {diff['point']}")
            print(f"    batch_pred:  {diff['batch_pred']}")
            print(f"    single_pred: {diff['single_pred']}")

    # Mostrar primeras predicciones
    print("\nPrimeras 10 predicciones:")
    print("  Index | X[0]   | X[1]   | Batch | Single | Match")
    print("  " + "-"*54)
    for i in range(min(10, len(X_test))):
        match_symbol = "✓" if batch_preds[i] == single_preds[i] else "✗"
        print(f"  {i:5d} | {X_test[i,0]:6.3f} | {X_test[i,1]:6.3f} | "
              f"{batch_preds[i]:5d} | {single_preds[i]:6d} | {match_symbol}")

    # Test 2: Análisis estadístico con múltiples trials
    print("\n" + "="*60)
    print("[3/3] TEST 2: Análisis Estadístico (5 trials)")
    print("="*60)

    batch_rates, single_rates, rate_diffs = test_with_multiple_trials(
        X_test, params, shots, n_layers, n_trials=5
    )

    print("\nAnálisis de tasas de predicción (proporción de clase 1):")
    print("  Máxima diferencia: {:.2%}".format(np.max(rate_diffs)))
    print("  Promedio diferencias: {:.2%}".format(np.mean(rate_diffs)))
    print("  Desviación estándar: {:.2%}".format(np.std(rate_diffs)))

    # Determinar si las diferencias son significativas
    # Con shots=100, esperamos variación de ~10% debido a ruido cuántico
    expected_shot_noise = 1.0 / np.sqrt(shots)  # ~10% para shots=100

    if np.max(rate_diffs) < expected_shot_noise * 2:
        print(f"\n✅ Diferencias DENTRO del ruido cuántico esperado (~{expected_shot_noise:.1%})")
        print("   batch y single son estadísticamente equivalentes")
    else:
        print(f"\n⚠️ Diferencias MAYORES al ruido esperado (~{expected_shot_noise:.1%})")
        print("   Posible inconsistencia entre batch y single")

    # Puntos con mayor discrepancia
    top_discrepancies = np.argsort(rate_diffs)[::-1][:5]
    print("\nPuntos con mayor discrepancia (top 5):")
    print("  Index | Batch Rate | Single Rate | Diferencia")
    print("  " + "-"*50)
    for idx in top_discrepancies:
        print(f"  {idx:5d} | {batch_rates[idx]:10.2%} | {single_rates[idx]:11.2%} | "
              f"{rate_diffs[idx]:10.2%}")

    # Conclusión
    print("\n" + "="*60)
    print("CONCLUSIÓN")
    print("="*60)

    if all_match and np.max(rate_diffs) < expected_shot_noise * 2:
        print("✅ predict_batch y predict_single_point son EQUIVALENTES")
        print("   El cambio a batch predictions NO es la causa del problema")
        print("\nEl problema debe estar en otro lado:")
        print("  - Cambios en el optimizer (COBYLA/SLSQP)")
        print("  - Parámetros de early stopping")
        print("  - Dataset demasiado pequeño (120 train vs 2000 típico)")
        print("  - Pocas layers (2 vs 5-6 típico)")
    elif not all_match:
        print("❌ predict_batch y predict_single_point NO coinciden")
        print("   BUG DETECTADO en predict_batch!")
        print("\nEsto podría explicar la degradación 82% → 53%")
    else:
        print("⚠️ Diferencias estadísticas detectadas entre batch y single")
        print("   Revisar implementación de predict_batch")

    print("="*60)
