# AI3023 Machine Learning Workshop — Spaceship Titanic

## 1. Project overview

This repository contains our Kaggle Spaceship Titanic binary classification project. We implemented and compared Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, and CatBoost.

The main reproducible training pipeline consists of model scripts in `src/train_*.py`, followed by `src/compare_models.py` and `src/analyze_fusion_vs_catboost.py`.

Reference-aided exploration was used only as an audit insight during experimentation and is excluded from the final reproducible training pipeline.

## 2. Repository structure

```text
spaceship-titanic-mlw/
├── archive/
│   └── reference_aided/
│       └── README.md
├── analysis/
│   ├── catboost_lean_experiment/
│   └── fusion_experiments/
├── data/
│   └── README.md
├── src/
│   ├── utils.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_catboost.py
│   ├── run_all.py
│   ├── compare_models.py
│   └── analyze_fusion_vs_catboost.py
├── outputs/
├── figures/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 3. Required data files

Place Kaggle files in `data/`:
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (recommended)

## 4. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Final model positioning (unified)

- `src/train_catboost.py` reproduces the **progress-stage CatBoost baseline**.
- `analysis/catboost_lean_experiment/` contains the stronger clean **Lean CatBoost + per-Side threshold** experiment.
- The best clean reproducible Kaggle-side result reported in this final code repository is **Lean CatBoost + per-Side threshold**, public score about **0.80851**.
- LightGBM is an additional standalone boosting experiment, public score about **0.80687**.
- Reference-aided post-hoc results are excluded from the final submitted repository and are not treated as standalone ML model results.

## 6. Quick demo

Use this path for fast TA inspection of code structure (skip heavy models):

```bash
python src/run_all.py --skip-heavy
python src/compare_models.py
python src/analyze_fusion_vs_catboost.py
```

`--skip-heavy` skips slower models such as XGBoost / LightGBM / CatBoost and is intended for quick checking only.

## 7. Full reproducibility run

Use this path for full training and comparison:

```bash
python src/run_all.py
python src/compare_models.py
python src/analyze_fusion_vs_catboost.py
```

The full run takes longer. LightGBM additionally requires `data/spaceship_catboost_preprocessed_package.zip` or extracted `data/spaceship_catboost_preprocessed_package/`.

## 8. Individual model commands

```bash
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_svm.py
python src/train_xgboost.py --make-submission
python src/train_lightgbm.py
python src/train_catboost.py
```

Optional clean analysis to reproduce Lean CatBoost experiment:

```bash
python analysis/catboost_lean_experiment/preprocess_lean.py
python analysis/catboost_lean_experiment/train_catboost_lean.py
```

## 9. Optional analysis modules

- Fusion experiment scripts are under `analysis/fusion_experiments/`.
- Lean CatBoost experiment scripts are under `analysis/catboost_lean_experiment/`.
- These are analysis extensions and are not required for the quick demo path.
