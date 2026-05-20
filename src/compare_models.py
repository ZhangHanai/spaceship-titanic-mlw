"""Generate model comparison table/plots from outputs/results_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURES_DIR, OUTPUT_DIR, ensure_directory

SUITABILITY_NOTES = {
    "logistic_regression": "Simple baseline, interpretable, limited non-linear ability.",
    "decision_tree": "Interpretable but can overfit without pruning.",
    "random_forest": "Stronger ensemble baseline, usually more stable than one tree.",
    "svm": "Useful non-linear classifier but scale-sensitive and slower on larger feature spaces.",
    "xgboost": "Strong tabular boosting model; captures interactions effectively.",
    "catboost": "Strong boosting model, especially suitable for categorical/tabular data.",
    "lightgbm": "Strong tabular boosting model; fast and interaction-friendly.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=OUTPUT_DIR / "results_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def plot_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        print(f"Skipping {out_path.name}: no numeric {value_col} values.")
        return
    plt.figure(figsize=(10, 5))
    plt.bar(tmp["model_name"], tmp[value_col], color="#4F81BD", edgecolor="black")
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(args.figures_dir)

    if not args.summary.exists():
        raise FileNotFoundError(f"Missing summary file: {args.summary}. Run python src/run_all.py first.")

    df = pd.read_csv(args.summary)
    if "notes" not in df.columns:
        df["notes"] = ""
    df["model_notes"] = df["model_name"].map(SUITABILITY_NOTES).fillna("Model note not defined.")

    comparison_path = output_dir / "model_comparison_summary.csv"
    df.to_csv(comparison_path, index=False)
    print(f"Saved comparison table: {comparison_path}")
    print(df[["model_name", "validation_accuracy", "cv_accuracy", "training_time_seconds", "model_notes"]])

    plot_bar(
        df,
        "validation_accuracy",
        "Validation Accuracy by Model",
        "Validation Accuracy",
        figures_dir / "model_comparison_accuracy.png",
    )
    plot_bar(
        df,
        "training_time_seconds",
        "Training Time by Model",
        "Seconds",
        figures_dir / "model_comparison_training_time.png",
    )


if __name__ == "__main__":
    main()
