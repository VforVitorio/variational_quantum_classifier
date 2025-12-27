"""
Test: Classical SVM with RBF Kernel - Comparison Baseline

Trains a classical SVM with RBF kernel on the same spiral dataset
to have a comparison baseline with the quantum classifier.

Usage:
    python test_svm_baseline.py
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import sys
import os

# Add root directory to path (tests/classic/ -> root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data.dataset_generator import make_spiral_dataset

def plot_svm_decision_boundary(X, y, clf, title="SVM Decision Boundary"):
    """Visualizes the SVM decision boundary."""
    h = 0.02  # step size in mesh

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0', edgecolors='k', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1', edgecolors='k', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/classic/svm_baseline_boundary.png', dpi=150)
    print(f"Decision boundary saved at: results/classic/svm_baseline_boundary.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("TEST: Classical SVM with RBF Kernel - Baseline")
    print("="*60)

    # Generate dataset (same as VQC)
    print("\n[1/4] Generating spiral dataset...")
    X, y = make_spiral_dataset(n_points=150, noise=0.1, normalize=True)
    print(f"Dataset generated: {len(X)} points")
    print(f"Distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")

    # Split train/val (80/20 - same as VQC)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"\nSplit train/validation (80/20):")
    print(f"  Train: {len(X_train)} points (Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)})")
    print(f"  Val:   {len(X_val)} points (Class 0={np.sum(y_val==0)}, Class 1={np.sum(y_val==1)})")

    # Test different SVM configurations
    print("\n[2/4] Training SVM with different configurations...\n")

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

        # Train SVM
        clf = SVC(**config, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
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

    # Show best configuration
    print("\n[3/4] Best configuration found:")
    print(f"  {best_config}")
    print(f"  Validation Accuracy: {best_val_acc*100:.2f}%")

    # Detailed analysis of best model
    print("\n[4/4] Detailed analysis of best model:\n")

    y_train_pred = best_clf.predict(X_train)
    y_val_pred = best_clf.predict(X_val)

    print("=== TRAIN Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_train, y_train_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_train, y_train_pred, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_train, y_train_pred, zero_division=0)*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))

    print("\n=== VALIDATION Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_val, y_val_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_val, y_val_pred, zero_division=0)*100:.2f}%")
    print(f"Recall:    {recall_score(y_val, y_val_pred, zero_division=0)*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Class 0', 'Class 1']))

    # Visualize decision boundary
    plot_svm_decision_boundary(X, y, best_clf,
                               f"SVM Baseline - {best_config}\nVal Accuracy: {best_val_acc*100:.2f}%")

    # Comparison with VQC
    print("\n" + "="*60)
    print("COMPARISON WITH VQC")
    print("="*60)
    print(f"SVM (RBF kernel) Val Accuracy:  {best_val_acc*100:.2f}%")
    print(f"VQC (reported)   Val Accuracy:  53.33%")
    print(f"\nDifference: {(best_val_acc*100 - 53.33):.2f}% in favor of SVM")
    print("\nConclusion:")
    if best_val_acc > 0.70:
        print("  ✓ Classical SVM significantly outperforms current VQC")
        print("  ✓ This suggests the problem is NOT in the dataset")
        print("  ✓ VQC has room for improvement (bug, hyperparameters, or architecture)")
    else:
        print("  ⚠️ SVM also struggles with this dataset")
        print("  ⚠️ This suggests dataset too small or too noisy")

    print("\n" + "="*60)
    print("Test completed successfully")
    print("="*60)
