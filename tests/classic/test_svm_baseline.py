"""
Test: SVM Clásico con Kernel RBF - Baseline de Comparación

Entrena un SVM clásico con kernel RBF en el mismo dataset de espirales
para tener un baseline de comparación con el clasificador cuántico.

Uso:
    python test_svm_baseline.py
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import os

# Añadir directorio raíz al path (tests/classic/ -> raíz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.dataset_generator import make_spiral_dataset

def plot_svm_decision_boundary(X, y, clf, title="SVM Decision Boundary"):
    """Visualiza la frontera de decisión del SVM."""
    h = 0.02  # step size en mesh

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Clase 0', edgecolors='k', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Clase 1', edgecolors='k', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/classic/svm_baseline_boundary.png', dpi=150)
    print(f"Frontera de decisión guardada en: results/classic/svm_baseline_boundary.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("TEST: SVM Clásico con Kernel RBF - Baseline")
    print("="*60)

    # Generar dataset (mismo que el VQC)
    print("\n[1/4] Generando dataset de espirales...")
    X, y = make_spiral_dataset(n_points=150, noise=0.1, normalize=True)
    print(f"Dataset generado: {len(X)} puntos")
    print(f"Distribución: Clase 0={np.sum(y==0)}, Clase 1={np.sum(y==1)}")

    # Split train/val (80/20 - igual que VQC)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"\nSplit train/validation (80/20):")
    print(f"  Train: {len(X_train)} puntos (Clase 0={np.sum(y_train==0)}, Clase 1={np.sum(y_train==1)})")
    print(f"  Val:   {len(X_val)} puntos (Clase 0={np.sum(y_val==0)}, Clase 1={np.sum(y_val==1)})")

    # Probar diferentes configuraciones de SVM
    print("\n[2/4] Entrenando SVM con diferentes configuraciones...\n")

    configs = [
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'name': 'RBF (C=1.0, gamma=scale)'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'name': 'RBF (C=10.0, gamma=scale)'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto', 'name': 'RBF (C=1.0, gamma=auto)'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0, 'name': 'RBF (C=1.0, gamma=1.0)'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 2.0, 'name': 'RBF (C=1.0, gamma=2.0)'},
    ]

    best_val_acc = 0
    best_config = None
    best_clf = None

    results = []

    for config in configs:
        name = config.pop('name')

        # Entrenar SVM
        clf = SVC(**config, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluar
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)

        results.append({
            'name': name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'clf': clf,
            'config': config
        })

        print(f"{name}")
        print(f"  Train Accuracy:  {train_acc*100:.2f}%")
        print(f"  Val Accuracy:    {val_acc*100:.2f}%")
        print(f"  Val Precision:   {val_precision*100:.2f}%")
        print(f"  Val Recall:      {val_recall*100:.2f}%")
        print()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = name
            best_clf = clf

    # Mostrar mejor configuración
    print("\n[3/4] Mejor configuración encontrada:")
    print(f"  {best_config}")
    print(f"  Validation Accuracy: {best_val_acc*100:.2f}%")

    # Análisis detallado del mejor modelo
    print("\n[4/4] Análisis detallado del mejor modelo:\n")

    y_train_pred = best_clf.predict(X_train)
    y_val_pred = best_clf.predict(X_val)

    print("=== Métricas en TRAIN ===")
    print(f"Accuracy:  {accuracy_score(y_train, y_train_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_train, y_train_pred, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_train, y_train_pred, zero_division=0)*100:.2f}%")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_train, y_train_pred))

    print("\n=== Métricas en VALIDATION ===")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_val, y_val_pred, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_val, y_val_pred, zero_division=0)*100:.2f}%")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Clase 0', 'Clase 1']))

    # Visualizar frontera de decisión
    plot_svm_decision_boundary(X, y, best_clf,
                               f"SVM Baseline - {best_config}\nVal Accuracy: {best_val_acc*100:.2f}%")

    # Comparación con VQC
    print("\n" + "="*60)
    print("COMPARACIÓN CON VQC")
    print("="*60)
    print(f"SVM (RBF kernel) Val Accuracy:  {best_val_acc*100:.2f}%")
    print(f"VQC (reportado)  Val Accuracy:  53.33%")
    print(f"\nDiferencia: {(best_val_acc*100 - 53.33):.2f}% a favor del SVM")
    print("\nConclusión:")
    if best_val_acc > 0.70:
        print("  ✓ El SVM clásico supera significativamente al VQC actual")
        print("  ✓ Esto sugiere que el problema NO está en el dataset")
        print("  ✓ El VQC tiene margen de mejora (bug, hiperparámetros, o arquitectura)")
    else:
        print("  ⚠️ El SVM también tiene dificultades con este dataset")
        print("  ⚠️ Esto sugiere dataset muy pequeño o muy ruidoso")

    print("\n" + "="*60)
    print("Test completado exitosamente")
    print("="*60)
