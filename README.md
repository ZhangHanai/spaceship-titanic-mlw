# AI3023 Machine Learning Workshop: Spaceship Titanic

This repository contains the unified group project for the Kaggle **Spaceship Titanic** binary classification task. The goal is to predict whether each passenger was `Transported` using reproducible machine learning scripts under one shared project structure.

Earlier notebook experiments were converted into runnable Python scripts for reproducibility. The final source code lives in `src/`, model outputs are written to `outputs/`, and report/demo figures are written to `figures/`.

## Model list

The project currently includes five model tracks:

- **SVM** using raw Kaggle CSV files and the SVM notebook-equivalent preprocessing pipeline.
- **LightGBM** using the preprocessed CatBoost-style feature package from the LightGBM experiment.
- **Logistic Regression** using the teammate package's preprocessing, Optuna hyperparameter search, leakage-safe validation evaluation, reports, and performance visualization.
- **Random Forest** using spending features, age binning, train-learned imputation values, one-hot encoding, 5-fold cross-validation, validation reports, feature importance, and optional Kaggle submission generation (`--make-submission`).
- **XGBoost** using feature engineering, train/test-safe preprocessing pipeline, randomized hyperparameter search, out-of-fold threshold tuning, and optional Kaggle submission generation (`--make-submission`).

## Repository structure

```text
spaceship-titanic-mlw/
├── README.md
├── requirements.txt
├── src/
│   ├── run_all.py
│   ├── compare_models.py
│   ├── train_svm.py
│   ├── train_lightgbm.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
├── outputs/
│   ├── .gitkeep
│   └── kaggle_score_log_template.csv
├── figures/
│   └── .gitkeep
└── data/
    ├── README.md
    └── .gitkeep
```

Generated files are intentionally ignored by Git. Running scripts creates submissions, metrics tables, feature importances, comparison artifacts, and plots in `outputs/` and `figures/`.

## Environment and dependencies

Python 3.10+ is recommended. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Core packages include `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, and `optuna`.

## Data setup

Kaggle data and generated feature packages are intentionally ignored by Git. Place local data under `data/`.

### Raw Kaggle CSV inputs

The SVM, Logistic Regression, and Random Forest scripts expect the raw Kaggle files:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

`sample_submission.csv` is optional for the current scripts, but it is part of the normal Kaggle download and useful for checking the expected submission format.

### LightGBM feature package inputs

The LightGBM script expects either this zip file:

```text
data/spaceship_catboost_preprocessed_package.zip
```

or the already extracted folder:

```text
data/spaceship_catboost_preprocessed_package/
├── X_train_catboost_features.csv
├── X_test_catboost_features.csv
├── y_train_with_ids.csv
├── test_passenger_ids.csv
└── catboost_preprocessing_metadata.json
```

If the zip exists and the folder does not, `src/train_lightgbm.py` extracts it automatically.

If required local data is missing, scripts raise clear file-not-found messages instead of faking results.

## How to run all models

Use the unified runner from the repository root:

```bash
python src/run_all.py
```

Run a selected subset:

```bash
python src/run_all.py --models svm random_forest logistic_regression
python src/run_all.py --models lightgbm
python src/run_all.py --models xgboost
```

Run a faster demo/smoke pass that skips the heavier LightGBM track, skips SVM tuning, and reduces Logistic Regression Optuna trials:

```bash
python src/run_all.py --skip-heavy
```

The runner executes each selected script in a subprocess, prints which models succeeded or failed, and continues after individual failures such as missing local data.

## How to run a single model

From the repository root:

```bash
python src/train_svm.py
python src/train_lightgbm.py
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_xgboost.py --make-submission
```

Each script accepts command-line options for data and output locations. Use `--help` on a script to see available options.

Faster local examples:

```bash
python src/train_svm.py --skip-tuning
python src/train_logistic_regression.py --optuna-trials 5
```

## How to generate the model comparison table

After running one or more models, generate a summary CSV and comparison plots:

```bash
python src/compare_models.py
```

This reads available result files from `outputs/` and writes:

```text
outputs/model_comparison_summary.csv
figures/model_comparison_accuracy.png
figures/model_comparison_training_time.png
```

Kaggle public leaderboard scores are not available from the training code. Enter any manually submitted Kaggle scores in:

```text
outputs/kaggle_score_log_template.csv
```

with columns:

```text
model,submission_file,public_score,notes
```

Then rerun `python src/compare_models.py` to include those scores in the comparison table.

## Expected outputs by model

### SVM

```text
outputs/svm_submission.csv
outputs/svm_evaluation_summary.csv
outputs/svm_validation_metrics.json
outputs/svm_classification_report.txt
figures/svm_confusion_matrix.png
```

### LightGBM

```text
outputs/submission_lgbm_v1.csv
outputs/lgbm_oof_predictions.csv
outputs/lgbm_test_predictions.csv
outputs/lgbm_feature_importance.csv
outputs/lgbm_fold_results.csv
outputs/lgbm_threshold_search.csv
figures/lgbm_feature_importance.png
```

### Logistic Regression

```text
outputs/logistic_regression_submission.csv
outputs/logistic_regression_validation_summary.csv
outputs/logistic_regression_metrics.csv
outputs/logistic_regression_feature_importance.csv
outputs/logistic_regression_classification_report.txt
figures/logistic_regression_performance_analysis.png
```

Validation metrics are computed with a model trained only on the training split. A separate final model is then fit on the full training set for the Kaggle submission.

### Random Forest

```text
outputs/random_forest_submission.csv
outputs/random_forest_validation_summary.csv
outputs/random_forest_classification_report.txt
outputs/random_forest_feature_importance.csv
figures/random_forest_confusion_matrix.png
figures/random_forest_feature_importance.png
```

Random Forest preprocessing learns the Age median and categorical modes from the training data and applies those values consistently to the test data.

## Demo checklist for final submission

1. Confirm dependencies install successfully: `pip install -r requirements.txt`.
2. Place `data/train.csv` and `data/test.csv` under `data/` for SVM, Logistic Regression, and Random Forest.
3. Place or extract the LightGBM feature package under `data/` if demoing LightGBM.
4. Run a fast smoke test: `python src/run_all.py --skip-heavy`.
5. Run the final selected model scripts with full settings when local data is available.
6. Generate comparison artifacts: `python src/compare_models.py`.
7. If Kaggle submissions were made manually, update `outputs/kaggle_score_log_template.csv` and rerun the comparison script.
8. Open the generated confusion matrices, feature-importance plots, and model-comparison plots for presentation screenshots.

## Reproducibility notes

- Scripts use relative project paths based on the repository root through `src/utils.py`.
- Random seeds are fixed at `42` where used in the original experiment logic.
- Raw data, local feature packages, generated outputs, caches, virtual environments, and zip archives are excluded through `.gitignore`.
- The repository is organized as one group project rather than separate personal folders.


### XGBoost

```text
outputs/xgboost_submission.csv
outputs/xgboost_cv_results.csv
outputs/xgboost_validation_summary.csv
```
