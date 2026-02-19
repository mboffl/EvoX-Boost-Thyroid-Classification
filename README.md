EvoX-Boost: Hybrid Evolutionary GPU-XGBoost for Multiclass Thyroid Disease Classification

Overview

This repository contains the complete experimental implementation and results for the EvoX-Boost framework — a hybrid evolutionary optimization approach applied to multiclass thyroid disorder classification.

The framework integrates:
-Optuna (Bayesian Optimization – TPE sampler)
-Differential Evolution (DE)
-GA–DE hybrid refinement
-GPU-accelerated XGBoost

Contents

-Hybrid hyperparameter optimization pipeline (Optuna → DE → GA–DE)
-GPU-enabled XGBoost training
-Stratified 5-fold cross-validation
-Fold-wise prediction logs
-Tuned hyperparameters (xgb_tuned_params.json)
-Runtime profiling logs
-Gain and permutation feature importance analysis
-Confusion matrices and ROC visualizations

Dataset

-Total samples: 22,632
-Number of classes: 15
-Multiclass thyroid disorder classification

The dataset used in this work is publicly available via Figshare (as referenced in the associated research work).
A cleaned and preprocessed version used for experimentation is included for reproducibility purposes.

Reproducibility

To reproduce the experimental pipeline;
pip install -r requirements.txt
python extreme_run_gpu.py
python paper_dashboard.py

A CUDA-enabled GPU environment is recommended for full optimization performance.

Correspondence

For academic queries:
Mohan Babu M
School of Computer Science and Engineering
VIT University, Vellore
