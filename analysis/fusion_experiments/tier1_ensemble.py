"""Optional Tier 1 fusion experiment (CatBoost + XGBoost + HistGradientBoosting).

This script is analysis-only and intentionally separate from the main demo path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT_FALLBACK / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import OUTPUT_DIR, REPO_ROOT, ensure_directory, require_file

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional Tier 1 fusion training.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=REPO_ROOT / "data" / "spaceship_catboost_preprocessed_package",
        help="Directory containing preprocessed CatBoost package CSV/JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "fusion_experiments",
        help="Directory to save fusion experiment outputs.",
    )
    parser.add_argument("--fast", action="store_true", help="Use smaller CV/model settings for smoke tests.")
    return parser.parse_args()


def load_data(package_dir: Path):
    meta = json.loads(require_file(package_dir / "catboost_preprocessing_metadata.json").read_text(encoding="utf-8"))
    cat_cols = meta["categorical_columns"]

    x_train = pd.read_csv(require_file(package_dir / "X_train_catboost_features.csv"))
    x_test = pd.read_csv(require_file(package_dir / "X_test_catboost_features.csv"))
    y_full = pd.read_csv(require_file(package_dir / "y_train_with_ids.csv"))

    for col in cat_cols:
        x_train[col] = x_train[col].astype(str).fillna("Missing")
        x_test[col] = x_test[col].astype(str).fillna("Missing")

    numeric_cols = [c for c in x_train.columns if c not in cat_cols]
    for col in numeric_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce").fillna(0.0)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(0.0)

    y = y_full["Transported"].astype(int).to_numpy()
    return x_train, x_test, y, cat_cols


def make_ordinal_views(x_train: pd.DataFrame, x_test: pd.DataFrame, cat_cols: list[str]):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    enc.fit(pd.concat([x_train[cat_cols], x_test[cat_cols]], axis=0))
    x_tr = x_train.copy()
    x_te = x_test.copy()
    x_tr[cat_cols] = enc.transform(x_train[cat_cols])
    x_te[cat_cols] = enc.transform(x_test[cat_cols])
    return x_tr.astype(np.float32), x_te.astype(np.float32)


def make_catboost(fast: bool):
    return CatBoostClassifier(
        iterations=800 if fast else 7000,
        learning_rate=0.03 if fast else 0.02,
        depth=6 if fast else 8,
        l2_leaf_reg=5.0,
        random_strength=1.2,
        random_seed=SEED,
        eval_metric="Accuracy",
        early_stopping_rounds=80 if fast else 150,
        verbose=False,
        allow_writing_files=False,
    )


def make_xgb(fast: bool):
    return XGBClassifier(
        n_estimators=500 if fast else 3000,
        learning_rate=0.05 if fast else 0.03,
        max_depth=5 if fast else 6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.5,
        tree_method="hist",
        eval_metric="error",
        early_stopping_rounds=60 if fast else 150,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def make_hgb(fast: bool):
    return HistGradientBoostingClassifier(
        max_iter=300 if fast else 2000,
        learning_rate=0.06 if fast else 0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30 if fast else 50,
        random_state=SEED,
    )


def threshold_sweep(prob, y):
    grid = np.arange(0.3, 0.6 + 1e-9, 0.005)
    accs = [accuracy_score(y, (prob >= t).astype(int)) for t in grid]
    idx = int(np.argmax(accs))
    return float(grid[idx]), float(accs[idx])


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir
    out_dir = ensure_directory(args.output_dir)

    x_train, x_test, y, cat_cols = load_data(package_dir)
    x_tr_ord, x_te_ord = make_ordinal_views(x_train, x_test, cat_cols)
    cat_idx = [x_train.columns.get_loc(c) for c in cat_cols]

    n_splits = 3 if args.fast else 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof = {k: np.zeros(len(x_train)) for k in ["cb", "xgb", "hgb"]}
    test_pred = {k: np.zeros(len(x_test)) for k in ["cb", "xgb", "hgb"]}
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        start = time.time()
        y_tr, y_va = y[tr_idx], y[va_idx]

        cb = make_catboost(args.fast)
        train_pool = Pool(x_train.iloc[tr_idx], y_tr, cat_features=cat_idx)
        val_pool = Pool(x_train.iloc[va_idx], y_va, cat_features=cat_idx)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        p_va_cb = cb.predict_proba(val_pool)[:, 1]
        oof["cb"][va_idx] = p_va_cb
        test_pred["cb"] += cb.predict_proba(Pool(x_test, cat_features=cat_idx))[:, 1] / n_splits

        xgb = make_xgb(args.fast)
        xgb.fit(x_tr_ord.iloc[tr_idx], y_tr, eval_set=[(x_tr_ord.iloc[va_idx], y_va)], verbose=False)
        p_va_xgb = xgb.predict_proba(x_tr_ord.iloc[va_idx])[:, 1]
        oof["xgb"][va_idx] = p_va_xgb
        test_pred["xgb"] += xgb.predict_proba(x_te_ord)[:, 1] / n_splits

        hgb = make_hgb(args.fast)
        hgb.fit(x_tr_ord.iloc[tr_idx], y_tr)
        p_va_hgb = hgb.predict_proba(x_tr_ord.iloc[va_idx])[:, 1]
        oof["hgb"][va_idx] = p_va_hgb
        test_pred["hgb"] += hgb.predict_proba(x_te_ord)[:, 1] / n_splits

        fold_metrics.append({
            "fold": fold,
            "catboost_acc@0.5": float(accuracy_score(y_va, (p_va_cb >= 0.5).astype(int))),
            "xgboost_acc@0.5": float(accuracy_score(y_va, (p_va_xgb >= 0.5).astype(int))),
            "histgb_acc@0.5": float(accuracy_score(y_va, (p_va_hgb >= 0.5).astype(int))),
            "seconds": round(time.time() - start, 2),
        })

    oof_ensemble = (oof["cb"] + oof["xgb"] + oof["hgb"]) / 3
    test_ensemble = (test_pred["cb"] + test_pred["xgb"] + test_pred["hgb"]) / 3

    thr_global, acc_global = threshold_sweep(oof_ensemble, y)

    side_train = x_train["Side"].astype(str).to_numpy() if "Side" in x_train.columns else None
    side_thresholds = {}
    if side_train is not None:
        for side in sorted(set(side_train)):
            mask = side_train == side
            if mask.sum() > 0:
                thr, acc = threshold_sweep(oof_ensemble[mask], y[mask])
                side_thresholds[side] = {"threshold": thr, "oof_acc": acc, "n": int(mask.sum())}

    np.savez(out_dir / "oof_probas.npz", y=y, cb=oof["cb"], xgb=oof["xgb"], hgb=oof["hgb"], ensemble=oof_ensemble)
    np.savez(out_dir / "test_probas.npz", cb=test_pred["cb"], xgb=test_pred["xgb"], hgb=test_pred["hgb"], ensemble=test_ensemble)
    (out_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "threshold_analysis.json").write_text(
        json.dumps({"ensemble_best_threshold": thr_global, "ensemble_oof_acc": acc_global, "per_side": side_thresholds}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved fusion outputs to: {out_dir}")


if __name__ == "__main__":
    main()
