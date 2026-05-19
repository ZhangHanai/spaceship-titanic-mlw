"""Create a report-ready comparison table and plots from saved model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURES_DIR, OUTPUT_DIR, ensure_directory

KAGGLE_TEMPLATE_COLUMNS = ["model", "submission_file", "public_score", "notes"]
SUMMARY_COLUMNS = [
    "Model",
    "validation/CV accuracy",
    "Kaggle public score",
    "training time",
    "output submission file",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare model outputs saved in outputs/.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path, help="Directory containing result CSV/JSON files.")
    parser.add_argument("--figures-dir", default=FIGURES_DIR, type=Path, help="Directory for comparison plots.")
    return parser.parse_args()


def read_kaggle_scores(output_dir: Path) -> pd.DataFrame:
    """Read or create the manual Kaggle public score log template."""
    template_path = output_dir / "kaggle_score_log_template.csv"
    if not template_path.exists():
        pd.DataFrame(columns=KAGGLE_TEMPLATE_COLUMNS).to_csv(template_path, index=False)
        print(f"Created Kaggle score template: {template_path}")
    scores = pd.read_csv(template_path)
    for column in KAGGLE_TEMPLATE_COLUMNS:
        if column not in scores.columns:
            scores[column] = pd.NA
    return scores[KAGGLE_TEMPLATE_COLUMNS]


def score_for(scores: pd.DataFrame, model: str, submission_file: str) -> object:
    """Return a manually logged Kaggle public score when available."""
    if scores.empty:
        return pd.NA
    model_key = model.lower().replace(" ", "_")
    matches = scores[
        (scores["model"].astype(str).str.lower().str.replace(" ", "_", regex=False) == model_key)
        | (scores["submission_file"].astype(str) == submission_file)
    ]
    if matches.empty:
        return pd.NA
    return matches.iloc[0].get("public_score", pd.NA)


def add_row(
    rows: list[dict[str, object]],
    scores: pd.DataFrame,
    model: str,
    accuracy: object,
    training_time: object,
    submission_file: str,
    notes: str,
) -> None:
    """Append one normalized model-comparison row."""
    rows.append(
        {
            "Model": model,
            "validation/CV accuracy": accuracy,
            "Kaggle public score": score_for(scores, model, submission_file),
            "training time": training_time,
            "output submission file": submission_file,
            "notes": notes,
        }
    )


def collect_rows(output_dir: Path, scores: pd.DataFrame) -> pd.DataFrame:
    """Collect comparison rows from available model artifact files."""
    rows: list[dict[str, object]] = []

    svm_json = output_dir / "svm_validation_metrics.json"
    if svm_json.exists():
        metrics = json.loads(svm_json.read_text(encoding="utf-8"))
        add_row(
            rows,
            scores,
            "SVM",
            metrics.get("best_validation_accuracy"),
            metrics.get("best_training_time_seconds"),
            "outputs/svm_submission.csv",
            "Validation split accuracy; CV accuracy available in svm_evaluation_summary.csv when tuning ran.",
        )

    rf_summary = output_dir / "random_forest_validation_summary.csv"
    if rf_summary.exists():
        summary = pd.read_csv(rf_summary).iloc[0]
        accuracy = summary.get("validation_accuracy", summary.get("mean_cv_accuracy", pd.NA))
        add_row(
            rows,
            scores,
            "Random Forest",
            accuracy,
            summary.get("training_time_seconds", pd.NA),
            "outputs/random_forest_submission.csv",
            "Validation split accuracy; mean CV accuracy also saved in random_forest_validation_summary.csv.",
        )

    log_summary = output_dir / "logistic_regression_validation_summary.csv"
    log_metrics = output_dir / "logistic_regression_metrics.csv"
    if log_metrics.exists():
        metrics = pd.read_csv(log_metrics)
        val_rows = metrics[metrics["split"] == "validation"]
        accuracy = val_rows.iloc[0].get("accuracy", pd.NA) if not val_rows.empty else pd.NA
        train_rows = metrics[metrics["split"] == "training"]
        training_time = pd.NA
        notes = "Validation split accuracy from leakage-safe split-trained model."
        if log_summary.exists():
            notes += " Full summary saved in logistic_regression_validation_summary.csv."
        if not train_rows.empty and "training_time_seconds" in train_rows.columns:
            training_time = train_rows.iloc[0]["training_time_seconds"]
        add_row(
            rows,
            scores,
            "Logistic Regression",
            accuracy,
            training_time,
            "outputs/logistic_regression_submission.csv",
            notes,
        )

    lgbm_threshold = output_dir / "lgbm_threshold_search.csv"
    lgbm_folds = output_dir / "lgbm_fold_results.csv"
    if lgbm_threshold.exists() or lgbm_folds.exists():
        accuracy = pd.NA
        notes = []
        if lgbm_threshold.exists():
            threshold_df = pd.read_csv(lgbm_threshold)
            if "accuracy" in threshold_df.columns:
                accuracy = threshold_df["accuracy"].max()
                notes.append("Best OOF threshold accuracy from threshold search.")
        training_time = pd.NA
        if lgbm_folds.exists():
            folds = pd.read_csv(lgbm_folds)
            if "training_time_seconds" in folds.columns:
                training_time = folds["training_time_seconds"].sum()
            if pd.isna(accuracy) and "accuracy_threshold_0.5" in folds.columns:
                accuracy = folds["accuracy_threshold_0.5"].mean()
                notes.append("Mean fold accuracy at threshold 0.5.")
        add_row(
            rows,
            scores,
            "LightGBM",
            accuracy,
            training_time,
            "outputs/submission_lgbm_v1.csv",
            " ".join(notes) or "LightGBM artifacts found.",
        )


    xgb_summary = output_dir / "xgboost_validation_summary.csv"
    if xgb_summary.exists():
        summary = pd.read_csv(xgb_summary).iloc[0]
        add_row(
            rows,
            scores,
            "XGBoost",
            summary.get("validation_accuracy", pd.NA),
            summary.get("training_time_seconds", pd.NA),
            "outputs/xgboost_submission.csv",
            "Leakage-safe CV/OOF metrics with optional submission mode; details in xgboost_validation_summary.csv.",
        )

    if not rows:
        print("No model result files found. Run one or more training scripts first.")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def plot_metric(summary: pd.DataFrame, column: str, title: str, ylabel: str, output_path: Path) -> None:
    """Save a bar plot for one numeric comparison column."""
    plot_df = summary.copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])
    if plot_df.empty:
        print(f"Skipping {output_path}; no numeric values for {column}.")
        return

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["Model"], plot_df[column], color="#4472C4", edgecolor="black")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main() -> None:
    """Generate comparison CSV and plots."""
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(args.figures_dir)

    kaggle_scores = read_kaggle_scores(output_dir)
    summary = collect_rows(output_dir, kaggle_scores)
    summary_path = output_dir / "model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved comparison summary: {summary_path}")

    plot_metric(
        summary,
        "validation/CV accuracy",
        "Model Comparison: Validation/CV Accuracy",
        "Accuracy",
        figures_dir / "model_comparison_accuracy.png",
    )
    plot_metric(
        summary,
        "training time",
        "Model Comparison: Training Time",
        "Seconds",
        figures_dir / "model_comparison_training_time.png",
    )


if __name__ == "__main__":
    main()
