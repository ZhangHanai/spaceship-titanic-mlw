"""Train the SVM model for the Kaggle Spaceship Titanic competition.

Run from the repository root:
    python src/train_svm.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import optuna
except ImportError as exc:  # pragma: no cover - gives a clear runtime message
    raise ImportError("Optuna is required for SVM tuning. Install dependencies with: pip install -r requirements.txt") from exc

from preprocessing import (
    SVM_CATEGORICAL_FEATURES,
    SVM_NUMERIC_FEATURES,
    preprocess_svm_data,
)
from utils import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, ensure_directory, require_file

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """Build the SVM encoder/scaler from the original notebook."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SVM_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), SVM_CATEGORICAL_FEATURES),
        ]
    )


def build_svm_pipeline(c: float = 1.0, gamma: float | str = "scale") -> Pipeline:
    """Create an RBF-kernel SVM pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", SVC(kernel="rbf", C=c, gamma=gamma)),
        ]
    )


def tune_svm(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int) -> tuple[dict[str, float], float, float]:
    """Run the Optuna search used by the SVM notebook."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        c_value = trial.suggest_float("C", 0.1, 100, log=True)
        gamma_value = trial.suggest_float("gamma", 0.001, 1, log=True)

        svm_model = build_svm_pipeline(c=c_value, gamma=gamma_value)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            svm_model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials)
    tuning_time = time.time() - start_time

    return study.best_params, float(study.best_value), tuning_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SVM Spaceship Titanic model.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="Directory containing train.csv/test.csv.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path, help="Directory for saved outputs.")
    parser.add_argument("--figures-dir", default=FIGURES_DIR, type=Path, help="Directory for saved figures.")
    parser.add_argument("--optuna-trials", default=20, type=int, help="Number of Optuna trials; notebook used 20.")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna and use the baseline C=1.0, gamma='scale' model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(args.figures_dir)

    train_path = require_file(args.data_dir / "train.csv", "Download the Kaggle train.csv into data/.")
    test_path = require_file(args.data_dir / "test.csv", "Download the Kaggle test.csv into data/.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)

    X, X_test, y, test_passenger_id = preprocess_svm_data(train_df, test_df)

    print("Processed training features shape:", X.shape)
    print("Processed test features shape:", X_test.shape)
    print("Target shape:", y.shape)
    print("\nMissing values in processed training data:")
    print(X.isnull().sum())

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nX_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("y_train shape:", y_train.shape)
    print("y_valid shape:", y_valid.shape)
    print("\nTarget distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nTarget distribution in validation set:")
    print(y_valid.value_counts(normalize=True))

    baseline_svm = build_svm_pipeline(c=1.0, gamma="scale")
    start_time = time.time()
    baseline_svm.fit(X_train, y_train)
    baseline_training_time = time.time() - start_time

    baseline_pred = baseline_svm.predict(X_valid)
    baseline_accuracy = accuracy_score(y_valid, baseline_pred)

    print("\nBaseline SVM Validation Accuracy:", baseline_accuracy)
    print("Baseline SVM Training Time:", baseline_training_time, "seconds")
    print("\nBaseline Classification Report:")
    print(classification_report(y_valid, baseline_pred))
    print("Baseline Confusion Matrix:")
    print(confusion_matrix(y_valid, baseline_pred))

    best_params: dict[str, float] | dict[str, str | float]
    best_cv_accuracy = None
    tuning_time = 0.0

    if args.skip_tuning:
        best_params = {"C": 1.0, "gamma": "scale"}
        print("\nSkipping Optuna tuning; using baseline parameters for final model.")
    else:
        best_params, best_cv_accuracy, tuning_time = tune_svm(X_train, y_train, args.optuna_trials)
        print("\nBest parameters:", best_params)
        print("Best 5-fold CV accuracy:", best_cv_accuracy)
        print("Tuning time:", tuning_time, "seconds")

    best_svm = build_svm_pipeline(c=float(best_params["C"]), gamma=best_params["gamma"])
    start_time = time.time()
    best_svm.fit(X_train, y_train)
    best_training_time = time.time() - start_time

    y_valid_pred = best_svm.predict(X_valid)
    best_valid_accuracy = accuracy_score(y_valid, y_valid_pred)

    print("\nBest SVM Validation Accuracy:", best_valid_accuracy)
    print("Best SVM Training Time:", best_training_time, "seconds")
    print("\nBest Parameters:")
    print(best_params)
    final_report = classification_report(y_valid, y_valid_pred, target_names=["Not Transported", "Transported"])
    final_cm = confusion_matrix(y_valid, y_valid_pred)
    print("\nClassification Report:")
    print(final_report)
    print("Confusion Matrix:")
    print(final_cm)

    report_path = output_dir / "svm_classification_report.txt"
    report_path.write_text(final_report, encoding="utf-8")

    cm_plot_path = figures_dir / "svm_confusion_matrix.png"
    disp = ConfusionMatrixDisplay(
        confusion_matrix=final_cm,
        display_labels=["Not Transported", "Transported"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    svm_summary = pd.DataFrame(
        {
            "Model": ["Baseline SVM", "Tuned SVM" if not args.skip_tuning else "Final SVM"],
            "Kernel": ["RBF", "RBF"],
            "C": [1.0, best_params["C"]],
            "Gamma": ["scale", best_params["gamma"]],
            "Validation Accuracy": [baseline_accuracy, best_valid_accuracy],
            "5-fold CV Accuracy": [None, best_cv_accuracy],
            "Training Time (seconds)": [baseline_training_time, best_training_time],
            "Tuning Time (seconds)": [None, tuning_time if not args.skip_tuning else None],
        }
    )

    summary_path = output_dir / "svm_evaluation_summary.csv"
    svm_summary.to_csv(summary_path, index=False)

    metrics_path = output_dir / "svm_validation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "baseline_validation_accuracy": float(baseline_accuracy),
                "best_validation_accuracy": float(best_valid_accuracy),
                "best_params": best_params,
                "best_cv_accuracy": best_cv_accuracy,
                "baseline_training_time_seconds": float(baseline_training_time),
                "best_training_time_seconds": float(best_training_time),
                "tuning_time_seconds": float(tuning_time),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final_svm = build_svm_pipeline(c=float(best_params["C"]), gamma=best_params["gamma"])
    final_svm.fit(X, y)
    test_predictions = final_svm.predict(X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_passenger_id,
            "Transported": test_predictions.astype(bool),
        }
    )
    submission_path = output_dir / "svm_submission.csv"
    submission.to_csv(submission_path, index=False)

    print("\nSaved SVM submission to:", submission_path)
    print("Saved SVM evaluation summary to:", summary_path)
    print("Saved SVM validation metrics to:", metrics_path)
    print("Saved SVM classification report to:", report_path)
    print("Saved SVM confusion matrix plot to:", cm_plot_path)
    print(submission.head())
    print(submission.shape)


if __name__ == "__main__":
    main()
