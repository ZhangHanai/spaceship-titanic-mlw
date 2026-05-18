# AI3023 Machine Learning Workshop: Spaceship Titanic

This repository contains the unified group project for the Kaggle **Spaceship Titanic** binary classification task. The goal is to predict whether each passenger was `Transported` using reproducible machine learning scripts under one shared project structure.

Earlier notebook experiments were converted into runnable Python scripts for reproducibility. The final source code lives in `src/`, with model outputs written to `outputs/` and figures written to `figures/`.

## Model list

The project currently includes four model tracks:

- **SVM** using raw Kaggle CSV files and the SVM notebook-equivalent preprocessing pipeline.
- **LightGBM** using the preprocessed CatBoost-style feature package from the LightGBM experiment.
- **Logistic Regression** using the teammate package's preprocessing, Optuna hyperparameter search, evaluation reports, and performance visualization.
- **Random Forest** using the teammate implementation's spending features, age binning, one-hot encoding, 5-fold cross-validation, validation reports, feature importance, and Kaggle submission generation.

## Repository structure

```text
spaceship-titanic-mlw/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train_svm.py
│   ├── train_lightgbm.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
├── outputs/
│   └── .gitkeep
├── figures/
│   ├── .gitkeep
│   ├── age_by_transported.png
│   ├── bivariate_summary_fixed.png
│   ├── boxplots_outliers.png
│   ├── cabin_analysis.png
│   ├── cabin_analysis_fixed.png
│   ├── categorical_analysis.png
│   ├── correlation_heatmap_fixed.png
│   ├── missing_values.png
│   ├── model comparison graph.png
│   ├── model comparison.png
│   ├── numerical_distributions.png
│   ├── spending_analysis.png
│   ├── spending_heatmap.png
│   └── target_distribution.png
└── data/
    ├── README.md
    └── .gitkeep
```

Generated files are intentionally ignored by Git. Running the scripts may create additional files such as submissions, metrics tables, feature importances, and model-specific plots in `outputs/` and `figures/`.

## Environment and dependencies

Python 3.10+ is recommended. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Core packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `lightgbm`
- `optuna`

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

## How to run all available models

From the repository root, run any model script independently:

```bash
python src/train_svm.py
python src/train_lightgbm.py
python src/train_logistic_regression.py
python src/train_random_forest.py
```

Each script accepts command-line options for data and output locations. Use `--help` on a script to see available options.

## SVM model

From the repository root:

```bash
python src/train_svm.py
```

The SVM pipeline follows the original notebook logic:

- fills spending features with `0`,
- applies `log1p` to spending features,
- fills selected categorical columns with training-set mode,
- fills `Age` with training-set median,
- bins age into four ordinal groups,
- drops `Name`, `PassengerId`, `Cabin`, and original `Age`,
- one-hot encodes categorical features,
- standardizes numeric features,
- evaluates a baseline RBF SVM,
- runs the Optuna tuning search by default using 20 trials,
- trains the selected SVM and creates a Kaggle submission.

For a faster local smoke test, skip Optuna and use baseline parameters:

```bash
python src/train_svm.py --skip-tuning
```

Expected SVM outputs:

```text
outputs/svm_submission.csv
outputs/svm_evaluation_summary.csv
outputs/svm_validation_metrics.json
```

## LightGBM model

From the repository root:

```bash
python src/train_lightgbm.py
```

The LightGBM pipeline follows the original notebook logic:

- loads the preprocessed CatBoost-style feature package,
- reads categorical feature metadata,
- aligns train/test categorical columns using shared `pandas.Categorical` categories,
- trains `lightgbm.LGBMClassifier` with 5-fold `StratifiedKFold`,
- uses the same fixed LightGBM hyperparameters and early stopping setup,
- averages test probabilities across folds,
- searches the best OOF classification threshold from `0.350` to `0.650`,
- evaluates OOF predictions with the best threshold,
- saves prediction, fold result, threshold, and feature importance artifacts.

Expected LightGBM outputs:

```text
outputs/submission_lgbm_v1.csv
outputs/lgbm_oof_predictions.csv
outputs/lgbm_test_predictions.csv
outputs/lgbm_feature_importance.csv
outputs/lgbm_fold_results.csv
outputs/lgbm_threshold_search.csv
figures/lgbm_feature_importance.png
```

## Logistic Regression model

From the repository root:

```bash
python src/train_logistic_regression.py
```

The Logistic Regression pipeline integrates the teammate zip package into the shared `src/` layout while preserving the original model approach:

- loads raw `data/train.csv` and `data/test.csv`,
- fills monetary missing values with `0`,
- applies `log1p` to monetary features,
- imputes categorical features with the mode,
- imputes age with the median and bins it into four ordinal groups,
- extracts `Cabin_deck`,
- creates `TotalSpend`, `TotalSpend_log`, `HasSpending`, `CryoSleep_bin`, and `VIP_bin`,
- label-encodes `HomePlanet`, `Destination`, and `Cabin_deck`,
- standardizes the selected model features,
- tunes Logistic Regression with Optuna using 50 trials by default,
- evaluates ROC-AUC, accuracy, precision, recall, F1, confusion matrix, validation summary, and classification reports,
- creates a Kaggle submission and a performance analysis figure.

For a faster local run, reduce the Optuna trials:

```bash
python src/train_logistic_regression.py --optuna-trials 5
```

Expected Logistic Regression outputs:

```text
outputs/logistic_regression_submission.csv
outputs/logistic_regression_validation_summary.csv
outputs/logistic_regression_metrics.csv
outputs/logistic_regression_feature_importance.csv
outputs/logistic_regression_classification_report.txt
figures/logistic_regression_performance_analysis.png
```

## Random Forest model

From the repository root:

```bash
python src/train_random_forest.py
```

The Random Forest pipeline integrates the uploaded teammate implementation into the shared `src/` layout while preserving the original model approach:

- loads raw `data/train.csv` and `data/test.csv`,
- fills spending feature missing values with `0`,
- creates `TotalSpending` before applying `log1p` to spending features,
- imputes `Age` with the median and bins it into four ordinal groups,
- imputes categorical features with the mode,
- drops `PassengerId`, `Name`, and `Cabin`,
- one-hot encodes `HomePlanet`, `Destination`, `CryoSleep`, and `VIP`,
- trains `RandomForestClassifier` with `n_estimators=300`, `max_depth=10`, `min_samples_split=5`, `random_state=42`, `n_jobs=-1`, and `criterion="gini"`,
- evaluates 5-fold `StratifiedKFold` cross-validation and a train/validation split,
- prints validation accuracy, a classification report, confusion matrix, training time, and saved output locations,
- saves validation metrics, feature importance, report-ready figures, and a Kaggle submission.

Expected Random Forest outputs:

```text
outputs/random_forest_submission.csv
outputs/random_forest_validation_summary.csv
outputs/random_forest_feature_importance.csv
figures/random_forest_confusion_matrix.png
figures/random_forest_feature_importance.png
```

## Reproducibility notes

- Scripts use relative project paths by default and are intended to run from the repository root.
- Random seeds are fixed at `42` where used in the original experiment logic.
- Raw data, local feature packages, generated outputs, caches, virtual environments, and zip archives are excluded through `.gitignore`.
- The repository is organized as one group project rather than separate personal folders.
