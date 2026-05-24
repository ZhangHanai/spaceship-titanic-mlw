"""Train lean CatBoost experiment for analysis use (optional path)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import OUTPUT_DIR, ensure_directory, require_file

SEEDS = [42, 123, 2024]
N_SPLITS = 5
PARAMS = {
    "iterations": 4000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 4.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "border_count": 128,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "od_type": "Iter",
    "od_wait": 100,
    "use_best_model": True,
    "verbose": False,
    "allow_writing_files": False,
}


def run_one_seed(x_train, y, x_test, cat_idx, seed, n_splits, params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(x_train), dtype=np.float64)
    test_pred = np.zeros(len(x_test), dtype=np.float64)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(x_train, y)):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = CatBoostClassifier(**params, random_seed=seed * 1000 + fold_idx)
        model.fit(
            Pool(x_tr, y_tr, cat_features=cat_idx),
            eval_set=Pool(x_va, y_va, cat_features=cat_idx),
        )
        oof[va_idx] = model.predict_proba(Pool(x_va, cat_features=cat_idx))[:, 1]
        test_pred += model.predict_proba(Pool(x_test, cat_features=cat_idx))[:, 1] / n_splits

    return oof, test_pred


def best_threshold(y_true, probas):
    best_t, best_acc = 0.5, float(((probas > 0.5).astype(int) == y_true).mean())
    for threshold in np.arange(0.30, 0.701, 0.005):
        acc = float(((probas > threshold).astype(int) == y_true).mean())
        if acc > best_acc:
            best_t, best_acc = float(threshold), acc
    return best_t, best_acc


def best_per_side_thresholds(y_true, probas, side_values):
    mask_p = (side_values == "P").to_numpy()
    mask_s = (side_values == "S").to_numpy()
    t_grid = np.arange(0.30, 0.701, 0.01)

    best = {"threshold_p": 0.5, "threshold_s": 0.5, "accuracy": float(((probas > 0.5).astype(int) == y_true).mean())}
    for t_p in t_grid:
        for t_s in t_grid:
            pred = np.where(mask_p, probas > t_p, probas > t_s).astype(int)
            acc = float((pred == y_true).mean())
            if acc > best["accuracy"]:
                best = {"threshold_p": float(t_p), "threshold_s": float(t_s), "accuracy": acc}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train lean CatBoost experiment.")
    parser.add_argument("--feature-dir", type=Path, default=OUTPUT_DIR / "catboost_lean")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "catboost_lean")
    parser.add_argument("--fast", action="store_true", help="Quick smoke mode with fewer folds/seeds/iterations")
    args = parser.parse_args()

    feature_dir = args.feature_dir
    output_dir = ensure_directory(args.output_dir)

    x_train = pd.read_csv(require_file(feature_dir / "X_train_lean.csv"))
    x_test = pd.read_csv(require_file(feature_dir / "X_test_lean.csv"))
    y = pd.read_csv(require_file(feature_dir / "y_train_lean.csv"))["Transported"].astype(int).to_numpy()
    test_ids = pd.read_csv(require_file(feature_dir / "test_passenger_ids.csv"))["PassengerId"]
    meta = json.loads(require_file(feature_dir / "lean_feature_metadata.json").read_text(encoding="utf-8"))

    cat_cols = meta["categorical"]
    cat_idx = [x_train.columns.get_loc(col) for col in cat_cols]

    seeds = [SEEDS[0]] if args.fast else SEEDS
    n_splits = 3 if args.fast else N_SPLITS
    params = dict(PARAMS)
    if args.fast:
        params["iterations"] = 500
        params["od_wait"] = 50

    start = time.time()
    oof_all, test_all = [], []
    for seed in seeds:
        oof_seed, test_seed = run_one_seed(x_train, y, x_test, cat_idx, seed, n_splits, params)
        oof_all.append(oof_seed)
        test_all.append(test_seed)

    oof_mean = np.mean(np.vstack(oof_all), axis=0)
    test_mean = np.mean(np.vstack(test_all), axis=0)

    t_best, acc_best = best_threshold(y, oof_mean)
    per_side = best_per_side_thresholds(y, oof_mean, x_train["Side"].astype(str))

    np.save(output_dir / "oof_probas_lean.npy", oof_mean)
    np.save(output_dir / "test_probas_lean.npy", test_mean)

    sub_t050 = pd.DataFrame({"PassengerId": test_ids, "Transported": (test_mean > 0.5).astype(bool)})
    sub_tbest = pd.DataFrame({"PassengerId": test_ids, "Transported": (test_mean > t_best).astype(bool)})
    test_side = x_test["Side"].astype(str)
    sub_ps = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Transported": np.where(test_side == "P", test_mean > per_side["threshold_p"], test_mean > per_side["threshold_s"]).astype(bool),
        }
    )

    sub_t050.to_csv(output_dir / "submission_lean_catboost_t050.csv", index=False)
    sub_tbest.to_csv(output_dir / "submission_lean_catboost_tbest.csv", index=False)
    sub_ps.to_csv(output_dir / "submission_lean_perside.csv", index=False)

    run_summary = {
        "seeds": seeds,
        "n_splits": n_splits,
        "fast_mode": args.fast,
        "n_features": int(x_train.shape[1]),
        "oof_acc_t050": float(((oof_mean > 0.5).astype(int) == y).mean()),
        "oof_acc_tbest": acc_best,
        "best_threshold": t_best,
        "elapsed_seconds": time.time() - start,
        "note": "Public score 0.80851 is from an already-submitted Kaggle run, not guaranteed by local reruns.",
    }
    threshold_summary = {
        "global_best_threshold": t_best,
        "global_best_oof_accuracy": acc_best,
        "per_side_best": per_side,
    }

    (output_dir / "lean_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    (output_dir / "lean_threshold_summary.json").write_text(json.dumps(threshold_summary, indent=2), encoding="utf-8")

    print(f"Saved lean CatBoost outputs to: {output_dir}")


if __name__ == "__main__":
    main()
