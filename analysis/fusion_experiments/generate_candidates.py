"""Generate optional Tier 1 fusion candidate submissions from saved probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT_FALLBACK / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import OUTPUT_DIR, REPO_ROOT, ensure_directory, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fusion candidate submission CSV files.")
    parser.add_argument("--package-dir", type=Path, default=REPO_ROOT / "data" / "spaceship_catboost_preprocessed_package")
    parser.add_argument("--proba-dir", type=Path, default=OUTPUT_DIR / "fusion_experiments")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "fusion_experiments")
    return parser.parse_args()


def best_thr(p, y):
    grid = np.arange(0.30, 0.60 + 1e-9, 0.005)
    accs = [accuracy_score(y, (p >= t).astype(int)) for t in grid]
    idx = int(np.argmax(accs))
    return float(grid[idx]), float(accs[idx])


def write_submission(path: Path, ids: pd.Series, probas: np.ndarray, threshold: float):
    pred = (probas >= threshold).astype(bool)
    pd.DataFrame({"PassengerId": ids, "Transported": pred}).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    oof = np.load(require_file(args.proba_dir / "oof_probas.npz"))
    test = np.load(require_file(args.proba_dir / "test_probas.npz"))
    y = oof["y"]

    test_ids = pd.read_csv(require_file(args.package_dir / "test_passenger_ids.csv"))["PassengerId"]
    side_train = pd.read_csv(require_file(args.package_dir / "X_train_catboost_features.csv"))["Side"].astype(str).to_numpy()
    side_test = pd.read_csv(require_file(args.package_dir / "X_test_catboost_features.csv"))["Side"].astype(str).to_numpy()

    cb_only = oof["cb"], test["cb"]
    ens111 = oof["ensemble"], test["ensemble"]
    cb_xgb = ((oof["cb"] + oof["xgb"]) / 2, (test["cb"] + test["xgb"]) / 2)
    ens411 = ((4 * oof["cb"] + oof["xgb"] + oof["hgb"]) / 6, (4 * test["cb"] + test["xgb"] + test["hgb"]) / 6)

    candidates = {
        "submission_cb_only_050.csv": (cb_only[1], 0.50, cb_only[0]),
        "submission_cb_only_042.csv": (cb_only[1], 0.42, cb_only[0]),
        "submission_ensemble_111_050.csv": (ens111[1], 0.50, ens111[0]),
        "submission_cb_xgb_050.csv": (cb_xgb[1], 0.50, cb_xgb[0]),
        "submission_4cb_xgb_hgb_050.csv": (ens411[1], 0.50, ens411[0]),
    }

    rows = []
    for filename, (test_prob, thr, oof_prob) in candidates.items():
        write_submission(output_dir / filename, test_ids, test_prob, thr)
        oof_acc = float(accuracy_score(y, (oof_prob >= thr).astype(int)))
        best_t, best_acc = best_thr(oof_prob, y)
        rows.append({
            "candidate": filename,
            "threshold": thr,
            "oof_acc_at_threshold": round(oof_acc, 6),
            "oof_best_threshold": round(best_t, 3),
            "oof_best_acc": round(best_acc, 6),
            "test_true_count": int((test_prob >= thr).sum()),
        })

    perside_oof = ens411[0]
    perside_test = ens411[1]
    side_thresholds: dict[str, float] = {}
    for side in sorted(set(side_train)):
        mask = side_train == side
        thr, _ = best_thr(perside_oof[mask], y[mask])
        side_thresholds[side] = thr
    pred = np.array([(perside_test[i] >= side_thresholds.get(side_test[i], 0.5)) for i in range(len(side_test))], dtype=bool)
    pd.DataFrame({"PassengerId": test_ids, "Transported": pred}).to_csv(output_dir / "submission_ensemble_perside.csv", index=False)

    rows.append({
        "candidate": "submission_ensemble_perside.csv",
        "threshold": "per-side",
        "oof_acc_at_threshold": None,
        "oof_best_threshold": None,
        "oof_best_acc": None,
        "test_true_count": int(pred.sum()),
    })

    pd.DataFrame(rows).to_csv(output_dir / "candidates_summary.csv", index=False)
    analysis = {
        "note": "This file reports internal OOF summaries only. Public Kaggle leaderboard scores are documented separately in src/analyze_fusion_vs_catboost.py.",
        "per_side_thresholds": side_thresholds,
        "candidates": rows,
    }
    (output_dir / "candidate_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated candidate files in: {output_dir}")


if __name__ == "__main__":
    main()
