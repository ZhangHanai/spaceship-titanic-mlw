from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURES_DIR, OUTPUT_DIR, ensure_directory


def main() -> None:
    rows = [
        {
            "experiment": "CatBoost @0.50",
            "kaggle_public_score": 0.80219,
            "category": "CatBoost threshold",
            "interpretation": "original threshold baseline",
        },
        {
            "experiment": "CatBoost @0.42",
            "kaggle_public_score": 0.80640,
            "category": "CatBoost threshold",
            "interpretation": "threshold calibration improves more than simple fusion",
        },
        {
            "experiment": "Simple average (CB+XGB+HGB)/3",
            "kaggle_public_score": 0.80289,
            "category": "Simple fusion",
            "interpretation": "simple fusion brought very limited gain",
        },
        {
            "experiment": "Pair average (CB+XGB)/2",
            "kaggle_public_score": 0.80289,
            "category": "Simple fusion",
            "interpretation": "pair fusion brought very limited gain",
        },
        {
            "experiment": "Weighted ensemble (4*CB+XGB+HGB)/6",
            "kaggle_public_score": 0.80313,
            "category": "Simple fusion",
            "interpretation": "weighting did not materially improve fusion",
        },
        {
            "experiment": "Ensemble + per-Side threshold",
            "kaggle_public_score": 0.80780,
            "category": "Fusion + calibration",
            "interpretation": "Best fusion-side result, mainly helped by calibration",
        },
        {
            "experiment": "Lean CatBoost + per-Side threshold",
            "kaggle_public_score": 0.80851,
            "category": "Best clean CatBoost",
            "interpretation": "Best clean CatBoost-based result; used as the main comparison point against fusion",
        },
    ]

    ensure_directory(OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "fusion_vs_catboost_summary.csv"
    fig_path = FIGURES_DIR / "fusion_vs_catboost_scores.png"
    summary.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(summary["experiment"], summary["kaggle_public_score"], color="#4C78A8")
    ax.set_title("Fusion vs CatBoost: Public Score Comparison")
    ax.set_ylabel("Kaggle public score")
    ax.set_ylim(0.800, 0.8092)
    ax.tick_params(axis="x", rotation=25)

    for bar, score in zip(bars, summary["kaggle_public_score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.00005, f"{score:.5f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print("Fusion increased complexity but did not reliably replace the CatBoost-based pipeline.")


if __name__ == "__main__":
    main()
