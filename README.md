# AI3023 Machine Learning Workshop: Spaceship Titanic

This repository contains a reproducible Python implementation for the Kaggle **Spaceship Titanic** binary classification task. The goal is to predict whether each passenger was `Transported` using two model tracks from the original experiments:

1. an SVM model using raw Kaggle CSV files and notebook-equivalent preprocessing, and
2. a LightGBM model using the preprocessed CatBoost-style feature package from the LightGBM notebook.

The original notebooks are kept for experiment history, but the final project code has been refactored into runnable Python scripts for cleaner GitHub submission, easier reproducibility, and inclusion in a final ZIP archive.

## Repository structure

```text
spaceship-titanic-mlw/
├── README.md
├── requirements.txt
├── .gitignore
├── SVM_Model_Spaceship_Titanic.ipynb
├── lightgbm_final.ipynb
├── src/
│   ├── train_svm.py
│   ├── train_lightgbm.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
├── outputs/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
└── data/
    ├── README.md
    └── .gitkeep
```

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
- `lightgbm`
- `optuna` for SVM hyperparameter tuning

## Data setup

Kaggle data and generated feature packages are intentionally ignored by Git. Place local data under `data/`.

### SVM inputs

The SVM script expects raw Kaggle files:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

`sample_submission.csv` is optional for the current script, but it is part of the normal Kaggle download and useful for checking the expected submission format.

### LightGBM inputs

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

## How to run the SVM model

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

For a faster local smoke test, you can skip Optuna and use baseline parameters:

```bash
python src/train_svm.py --skip-tuning
```

Expected SVM outputs:

```text
outputs/svm_submission.csv
outputs/svm_evaluation_summary.csv
outputs/svm_validation_metrics.json
```

## How to run the LightGBM model

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

## Reproducibility notes

- Scripts use relative paths and are intended to run from the repository root.
- Random seeds are fixed at `42` where used in the original notebook logic.
- The notebooks remain in the repository as experimental records, while the `src/` scripts are the clean runnable source code for the final project.
- Generated outputs, local Kaggle data, zip packages, caches, and virtual environments are excluded through `.gitignore`.
