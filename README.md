# EvoX-Boost: Hybrid Evolutionary GPU-XGBoost for Multiclass Thyroid Disease Classification

## 📌 Overview

EvoX-Boost is a hybrid evolutionary optimization framework designed to enhance the performance of XGBoost for large-scale multiclass classification problems. This work focuses on thyroid disease classification using Electronic Medical Record (EMR)-scale data.

The framework integrates multiple optimization strategies in a sequential pipeline to achieve superior predictive performance, computational efficiency, and robustness.

---

## 🚀 Key Contributions

* Hybrid optimization pipeline: **Optuna → Differential Evolution (DE) → GA-DE Hybrid**
* GPU-accelerated XGBoost for scalable learning
* Adaptive stratified cross-validation
* Robustness evaluation (noise + missing data)
* Hyperparameter sensitivity analysis
* Statistical significance validation (Friedman + Wilcoxon tests)
* Convergence and runtime analysis
* Publication-ready visualization suite

---

## 🧠 Methodology

The EvoX-Boost pipeline consists of:

1. **Initial Search (Optuna)**
   Bayesian optimization to explore the hyperparameter space.

2. **Refinement (Differential Evolution)**
   Fine-tunes the best Optuna solution.

3. **Hybrid Optimization (GA + DE)**
   Combines global exploration and local exploitation.

4. **Final Evaluation**
   Stratified K-Fold cross-validation with performance logging.

---

## 📂 Repository Structure

```
├── extreme_run_gpu.py        # Main experimental pipeline
├── paper_dashboard.py        # Visualization & analysis dashboard
├── results/
│   ├── csv/                 # All numerical results
│   ├── figures/             # Publication-ready plots
│   └── models/              # Saved model & parameters
├── requirements.txt
└── README.md
```

---

## 📊 Experimental Results

### Performance (XGBoost - EvoX)

* Accuracy: **0.9890**
* Precision: **0.9890**
* Recall: **0.9890**
* F1-score: **0.9890**

### Validation Experiments

The repository includes extended validation:

* ✅ Ablation Study (Optuna vs Optuna+DE vs EvoX)
* ✅ Optimization Runtime Analysis
* ✅ Hyperparameter Sensitivity Analysis
* ✅ Robustness Testing
* ✅ Statistical Significance Testing
* ✅ Convergence Analysis

---

## 📁 Results Directory

| Type        | Location           |
| ----------- | ------------------ |
| CSV Results | `results/csv/`     |
| Figures     | `results/figures/` |
| Model Files | `results/models/`  |

---

## ⚙️ Reproducibility

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run experiments

```
python extreme_run_gpu.py
```

### 3. Generate visualizations

```
python paper_dashboard.py
```

---

## 💻 System Requirements

* Python 3.11
* CUDA-enabled GPU (recommended)
* Minimum 16GB RAM

---

## 📊 Dataset

The thyroid dataset used in this study contains **22,632 samples across 15 classes** and is publicly available via Figshare (as referenced in the manuscript).

A preprocessed version is included for reproducibility.

---

## 📈 Output Files

Running the pipeline generates:

* Prediction logs (per fold)
* Tuned hyperparameters
* Model file (`.pkl`)
* Convergence history
* Statistical test results
* Evaluation metrics

---

## 📬 Correspondence

**Mohan Babu M**
School of Computer Science and Engineering
VIT University, Vellore

For academic queries, please contact via institutional affiliation.

---

## ⚠️ Note

This repository corresponds to a **submitted research manuscript**.
Public release is limited to reproducibility purposes until formal publication.

---

## 📜 License

This project is intended for academic and research purposes only.
