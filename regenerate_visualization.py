#!/usr/bin/env python3
"""
Script para regenerar visualizaciones con mejor calidad
sin necesidad de re-entrenar el modelo.

Usa el modelo guardado y aumenta shots + resolution
para obtener fronteras de decisión más suaves.
"""

import numpy as np
import pickle
from src.classifier import QuantumClassifier
from src.utils import plot_decision_boundary
from data.dataset_generator import make_spiral_dataset

def main():
    print("=" * 70)
    print("REGENERACIÓN DE VISUALIZACIÓN CON ALTA CALIDAD")
    print("=" * 70)

    # 1. Cargar datos (misma configuración que training)
    print("\n[1/3] Cargando dataset...")
    X, y = make_spiral_dataset(n_points=100, noise=0.1, normalize=True)
    print(f"Dataset cargado: {X.shape[0]} puntos")

    # 2. Cargar modelo guardado
    print("\n[2/3] Cargando modelo entrenado...")
    try:
        with open('results/best_model_params.pkl', 'rb') as f:
            data = pickle.load(f)

        # El pkl guarda un diccionario completo, no solo los parámetros
        best_params = data['params']
        n_params = data['n_params']
        n_layers = data['n_layers']

        print(f"Modelo cargado:")
        print(f"  - Parámetros: {n_params} valores")
        print(f"  - Layers: {n_layers}")
        print(f"  - Array shape: {best_params.shape}")

        # Recrear clasificador con más shots para visualización suave
        classifier = QuantumClassifier(
            n_qubits=2,
            n_params=n_params,
            shots=300,  # Aumentado de 100 para reducir ruido
            n_layers=n_layers
        )
        classifier.params = best_params
        print(f"Clasificador listo: shots={classifier.shots}, layers={n_layers}")

    except FileNotFoundError:
        print("Error: No se encontró results/best_model_params.pkl")
        print("Debes entrenar el modelo primero (ejecuta main.py)")
        return

    # 3. Regenerar visualización con alta calidad
    print("\n[3/3] Regenerando frontera de decisión...")
    print("Configuración de alta calidad:")
    print(f"  - Resolution: 60×60 = 3600 puntos (vs 1600 anterior)")
    print(f"  - Shots: 300 por punto (vs 100 anterior)")
    print(f"  - Estimado: ~8-12 minutos")
    print("\nGenerando...")

    plot_decision_boundary(
        classifier, X, y,
        resolution=60,  # Mayor resolución para mejor detalle
        title="Frontera de Decisión (Alta Calidad: 300 shots, 60×60 res)",
        save_path="results/decision_boundary_high_quality.png"
    )

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)
    print("\nArchivo generado:")
    print("  - results/decision_boundary_high_quality.png")
    print("\nComparalo con:")
    print("  - results/decision_boundary.png (original)")
    print("\nMejoras con 300 shots y resolución 60:")
    print("  - Frontera más suave")
    print("  - Menos pixelación")
    print("  - Reducción de ruido cuántico")
    print("=" * 70)

if __name__ == "__main__":
    main()
