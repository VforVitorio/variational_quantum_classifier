# Variational Quantum Classifier (VQC)

A hybrid quantum-classical machine learning project that implements a Variational Quantum Classifier to solve non-linearly separable classification problems using parametrized quantum circuits.

## 🎯 Project Overview

This project develops a quantum classifier that:

- Encodes classical data into quantum states
- Processes information through parametrized quantum gates
- Learns to classify data via iterative parameter optimization
- Demonstrates practical quantum machine learning applications

  **Problem** : Binary classification of intertwined spiral dataset (non-linearly separable)

  **Approach** : Hybrid quantum-classical algorithm combining PyQuil quantum circuits with classical optimization (SciPy)

## 🚀 Quick Start

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd proyecto_clasificador_cuantico
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Classifier

Execute the complete pipeline:

```bash
python main.py
```

This will:

- Generate the spiral dataset
- Train the quantum classifier
- Display accuracy metrics
- Save visualizations to `results/`

### Interactive Analysis

For detailed exploration and step-by-step analysis, open the Jupyter notebook:

```bash
jupyter notebook full_notebook.ipynb
```

## 📁 Project Structure

```
proyecto_clasificador_cuantico/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/
│   └── dataset_generator.py      # Spiral dataset generator
├── src/
│   ├── quantum_circuit.py        # Encoder + Variational Layer + Measurement
│   ├── classifier.py             # VQC class + optimization logic
│   └── utils.py                  # Visualization + metrics
├── results/                      # Auto-generated outputs
│   ├── decision_boundary.png     # Classification boundary plot
│   └── metrics.txt               # Performance metrics
├── main.py                       # Quick demo script
└── full_notebook.ipynb           # Complete interactive analysis
```

## 🛠 Technology Stack

- **PyQuil 3.2.1** : Quantum circuit framework
- **SciPy** : Classical optimization (COBYLA, Nelder-Mead)
- **NumPy** : Numerical operations
- **Matplotlib** : Visualization
- **scikit-learn** : Performance metrics

## 📊 Expected Results

- **Training Accuracy** : >85%
- **Execution Time** : ~1-2 minutes
- **Output Files** : Decision boundary plots + metrics report

## 🎓 Academic Context

**Course** : Quantum & Natural Computing

**Institution** : Universidad Intercontinental de la Empresa (UIE)

**Program** : 4th Year Intelligent Systems Engineering

**Development Time** : 5-6 weeks

## 👥 Authors

- Víctor Vega Sobral
- Santiago Souto Ortega

## 📚 References

- Havlíček et al. (2019) - _Supervised learning with quantum-enhanced feature spaces_
- Schuld & Petruccione (2018) - _Quantum Machine Learning_
- [PyQuil Documentation](https://pyquil-docs.rigetti.com/)
- [PennyLane VQC Tutorials](https://pennylane.ai/)

## 📝 License

Licensed under the Apache License 2.0 - see [LICENSE](https://claude.ai/chat/LICENSE) file for details.

---

**Note** : This project uses quantum simulation. No access to physical quantum hardware is required.
