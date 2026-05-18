"""Train the LightGBM model for the Kaggle Spaceship Titanic competition.

This script uses the preprocessed CatBoost-style feature package from the
LightGBM notebook and keeps the same 5-fold StratifiedKFold validation,
LightGBM parameters, categorical handling, threshold search, and saved outputs.

Run from the repository root:
    python src/train_lightgbm.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - gives a clear runtime message
    raise ImportError("LightGBM is required. Install dependencies with: pip install -r requirements.txt") from exc

from metrics import threshold_search
from preprocessing import align_categorical_columns
from utils import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, ensure_directory, extract_zip_if_needed, read_json, require_file

N_SPLITS = 5
RANDOM_STATE = 42
PACKAGE_NAME = "spaceship_catboost_preprocessed_package"

LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.015,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the LightGBM Spaceship Titanic model.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="Directory containing the preprocessed package.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path, help="Directory for prediction CSV outputs.")
    parser.add_argument("--figures-dir", default=FIGURES_DIR, type=Path, help="Directory for generated plots.")
    return parser.parse_args()


def resolve_package_dir(data_dir: Path) -> Path:
    """Return the extracted feature package directory, extracting the zip if needed."""
    extract_dir = data_dir / PACKAGE_NAME
    zip_path = data_dir / f"{PACKAGE_NAME}.zip"

    if extract_dir.exists():
        return extract_dir

    help_message = (
        f"Place {PACKAGE_NAME}.zip in data/ or extract it to data/{PACKAGE_NAME}/."
    )
    require_file(zip_path, help_message)
    return extract_zip_if_needed(zip_path, extract_dir)


def load_lightgbm_package(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Load feature matrices, labels, IDs, and categorical column metadata."""
    X_train = pd.read_csv(require_file(package_dir / "X_train_catboost_features.csv"), low_memory=False)
    X_test = pd.read_csv(require_file(package_dir / "X_test_catboost_features.csv"), low_memory=False)
    y_df = pd.read_csv(require_file(package_dir / "y_train_with_ids.csv"))
    test_ids = pd.read_csv(require_file(package_dir / "test_passenger_ids.csv"))
    metadata = read_json(require_file(package_dir / "catboost_preprocessing_metadata.json"))

    y = y_df["Transported"].astype(int)
    categorical_cols = [col for col in metadata["categorical_columns"] if col in X_train.columns]
    missing_categorical_cols = [col for col in metadata["categorical_columns"] if col not in X_train.columns]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y shape:", y.shape)
    print("Number of categorical columns used:", len(categorical_cols))

    if missing_categorical_cols:
        print("Categorical columns not found:", missing_categorical_cols)
    else:
        print("All metadata categorical columns are found.")

    return X_train, X_test, y_df, test_ids, y, categorical_cols


def train_cv_lightgbm(
    X_train_lgb: pd.DataFrame,
    X_test_lgb: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, float, float]:
    """Train LightGBM with 5-fold StratifiedKFold and collect predictions."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred = np.zeros(len(X_train_lgb))
    test_pred = np.zeros(len(X_test_lgb))
    fold_results = []
    feature_gain_list = []

    start_total = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lgb, y), start=1):
        X_tr = X_train_lgb.iloc[train_idx]
        X_val = X_train_lgb.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)

        start_fold = time.time()
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(period=250),
            ],
        )
        fold_time = time.time() - start_fold

        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test_lgb)[:, 1]

        oof_pred[val_idx] = val_proba
        test_pred += test_proba / N_SPLITS

        val_label_05 = (val_proba >= 0.5).astype(int)
        fold_acc = accuracy_score(y_val, val_label_05)
        best_iteration = model.best_iteration_ if model.best_iteration_ else LGB_PARAMS["n_estimators"]

        fold_results.append(
            {
                "fold": fold,
                "accuracy_threshold_0.5": fold_acc,
                "best_iteration": best_iteration,
                "training_time_seconds": fold_time,
            }
        )

        gain = model.booster_.feature_importance(importance_type="gain")
        feature_gain_list.append(gain)

        print(
            f"Fold {fold}: accuracy@0.5 = {fold_acc:.5f}, "
            f"best_iteration = {best_iteration}, time = {fold_time:.2f}s"
        )

    total_time = time.time() - start_total
    fold_results_df = pd.DataFrame(fold_results)
    mean_gain = np.mean(np.vstack(feature_gain_list), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": X_train_lgb.columns,
            "importance_gain": mean_gain,
        }
    ).sort_values("importance_gain", ascending=False)

    cv_acc_05 = accuracy_score(y, (oof_pred >= 0.5).astype(int))
    return oof_pred, test_pred, fold_results_df, importance_df, cv_acc_05, total_time


def save_feature_importance_plot(importance_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Save the top LightGBM feature importance plot."""
    ensure_directory(figures_dir)
    plot_path = figures_dir / "lgbm_feature_importance.png"

    plt.figure(figsize=(8, 7))
    top_n = 25
    plt.barh(
        importance_df.head(top_n)["feature"][::-1],
        importance_df.head(top_n)["importance_gain"][::-1],
    )
    plt.xlabel("Mean Gain Importance")
    plt.title("Top LightGBM Feature Importances")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(args.figures_dir)

    print("LightGBM version:", lgb.__version__)

    package_dir = resolve_package_dir(args.data_dir)
    print("Using feature package directory:", package_dir)
    print("Package files:")
    for file_path in sorted(package_dir.iterdir()):
        print("-", file_path.name)

    X_train, X_test, y_df, test_ids, y, categorical_cols = load_lightgbm_package(package_dir)
    X_train_lgb, X_test_lgb = align_categorical_columns(X_train, X_test, categorical_cols)

    print("Object columns after conversion:")
    print(X_train_lgb.select_dtypes(include=["object"]).columns.tolist())
    print("\nCategorical dtypes:")
    print(X_train_lgb[categorical_cols].dtypes.head(10))

    oof_pred, test_pred, fold_results_df, importance_df, cv_acc_05, total_time = train_cv_lightgbm(
        X_train_lgb,
        X_test_lgb,
        y,
        categorical_cols,
    )

    print("\nFold results:")
    print(fold_results_df)
    print(f"\nOOF accuracy at threshold 0.5: {cv_acc_05:.5f}")
    print(f"Total CV training time: {total_time:.2f}s")

    best_threshold, best_oof_acc, threshold_results = threshold_search(y, oof_pred)
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best OOF accuracy: {best_oof_acc:.5f}")
    print(f"OOF accuracy at 0.5: {cv_acc_05:.5f}")

    oof_label = (oof_pred >= best_threshold).astype(int)
    print("\nClassification report based on best threshold:")
    print(classification_report(y, oof_label, target_names=["Not Transported", "Transported"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, oof_label))

    print("\nTop feature importances:")
    print(importance_df.head(30))
    plot_path = save_feature_importance_plot(importance_df, figures_dir)

    test_label = (test_pred >= best_threshold).astype(bool)
    submission = pd.DataFrame(
        {
            "PassengerId": test_ids["PassengerId"],
            "Transported": test_label,
        }
    )

    oof_output = pd.DataFrame(
        {
            "PassengerId": y_df["PassengerId"],
            "y_true": y,
            "lgbm_oof_proba": oof_pred,
            "lgbm_oof_label_best_threshold": oof_label,
        }
    )

    test_output = pd.DataFrame(
        {
            "PassengerId": test_ids["PassengerId"],
            "lgbm_test_proba": test_pred,
            "lgbm_test_label_best_threshold": test_label,
        }
    )

    submission_path = output_dir / "submission_lgbm_v1.csv"
    oof_output_path = output_dir / "lgbm_oof_predictions.csv"
    test_output_path = output_dir / "lgbm_test_predictions.csv"
    importance_path = output_dir / "lgbm_feature_importance.csv"
    fold_results_path = output_dir / "lgbm_fold_results.csv"
    threshold_results_path = output_dir / "lgbm_threshold_search.csv"

    submission.to_csv(submission_path, index=False)
    oof_output.to_csv(oof_output_path, index=False)
    test_output.to_csv(test_output_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    fold_results_df.to_csv(fold_results_path, index=False)
    threshold_results.to_csv(threshold_results_path, index=False)

    print("\nSaved:")
    for path in [
        submission_path,
        oof_output_path,
        test_output_path,
        importance_path,
        fold_results_path,
        threshold_results_path,
        plot_path,
    ]:
        print("-", path)

    print("\nSubmission preview:")
    print(submission.head())
    print(submission["Transported"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
