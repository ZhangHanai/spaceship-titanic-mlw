# AI3023 Machine Learning Workshop — Spaceship Titanic

Binary classification project for Kaggle **Spaceship Titanic**: predict `Transported` for each passenger.

## Repository structure

```text
spaceship-titanic-mlw/
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
│   └── generate_best_submission_082183.py
├── outputs/
├── figures/
├── submissions/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Required data files

Place Kaggle files in `data/`:
- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv` (recommended)

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run one model

```bash
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_svm.py
python src/train_xgboost.py --make-submission
python src/train_lightgbm.py
python src/train_catboost.py
```

## Run all models

```bash
python src/run_all.py
```

Fast smoke run:

```bash
python src/run_all.py --skip-heavy
```

## Final model list

- Logistic Regression
- Random Forest
- SVM
- XGBoost
- LightGBM
- CatBoost

## Main file guide

- `src/run_all.py`: runs integrated training scripts and writes `outputs/results_summary.csv`.
- `src/compare_models.py`: reads summary output and writes comparison table/plots.
- `src/train_catboost.py`: scripted CatBoost progress-stage baseline migrated from notebook; writes `outputs/submission_catboost_v1.csv`, `outputs/catboost_fold_results.csv`, `outputs/catboost_oof_predictions.csv`, and `outputs/catboost_feature_importance.csv`.
- `src/train_*.py`: individual teammate model training/evaluation/submission scripts.
- `src/preprocessing.py`, `src/metrics.py`, `src/utils.py`: shared utilities.

## Reproducibility notes

- Fixed random seed settings are used where applicable.
- `test.csv` labels are never used (Kaggle test has no labels).
- The CatBoost script keeps the original progress-stage notebook pipeline to reproduce the reported baseline; stricter leakage-safe variants can be evaluated separately as ablation work.

## Links

- Kaggle competition: https://www.kaggle.com/competitions/spaceship-titanic
- GitHub repository: (filled in after pushing to GitHub)

## Optional: reproduce the reference-aided submission

The script `src/generate_best_submission_082183.py` reproduces an auxiliary post-processing submission that combines our CatBoost baseline with a public reference submission. This is a rule-based post-hoc analysis used in our report's error analysis section and is not produced by a standalone trained model. The main reproducible results come from the `src/train_*.py` scripts.

Required input files:
- `data/test.csv`
- `submissions/reference_aided_inputs/submission_catboost_threshold_050.csv`
- `submissions/reference_aided_inputs/reference_submission_082137.csv`

Run from repository root:

```bash
python src/generate_best_submission_082183.py
```

Output files are written to `outputs/`.
