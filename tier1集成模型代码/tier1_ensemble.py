"""
Tier 1 - Independent improvement over CatBoost baseline.

Pipeline:
    1. Load preprocessed 102-feature CatBoost package.
    2. 5-fold stratified CV with CatBoost, XGBoost, HistGradientBoosting.
    3. Save OOF probabilities for every model.
    4. Average the three OOF probabilities (soft voting).
    5. Sweep classification threshold on OOF accuracy.
    6. Also compute per-Side optimal thresholds as a diagnostic for the
       Side=P bias surfaced by the reference-diff analysis.
    7. Retrain-on-fold predictions are averaged for the test set, then the
       chosen threshold is applied to produce submission CSVs.

Reproducibility:
    SEED = 42 everywhere. All randomized components are seeded.

Configurable data dir:
    Set environment variable SPACETITANIC_DATA, or just put this script
    in the same directory as the CSVs and run it.

Outputs (written next to the data, all under DATA_DIR):
    oof_probas.npz
    test_probas.npz
    fold_metrics.json
    threshold_analysis.json
    submission_ensemble.csv            (global optimal threshold)
    submission_ensemble_perside.csv    (per-Side thresholds)

Tested on:
    catboost 1.2.10, xgboost 3.2.0, scikit-learn 1.8.0, pandas 2.x.
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("SPACETITANIC_DATA", "."))
OUT_DIR = DATA_DIR
SEED = 42
N_SPLITS = 5

FILES = {
    "X_train": DATA_DIR / "X_train_catboost_features.csv",
    "X_test": DATA_DIR / "X_test_catboost_features.csv",
    "y_train": DATA_DIR / "y_train_with_ids.csv",
    "test_ids": DATA_DIR / "test_passenger_ids.csv",
    "meta": DATA_DIR / "catboost_preprocessing_metadata.json",
}


def check_files():
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  " + "\n  ".join(missing)
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    with open(FILES["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    cat_cols = meta["categorical_columns"]

    X_train = pd.read_csv(FILES["X_train"])
    X_test = pd.read_csv(FILES["X_test"])
    y_full = pd.read_csv(FILES["y_train"])
    test_ids = pd.read_csv(FILES["test_ids"])

    # Force categorical columns to string so all three libraries agree.
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("Missing")
        X_test[c] = X_test[c].astype(str).fillna("Missing")

    # Ensure numeric columns are float (some came in as int).
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0.0)

    y = y_full["Transported"].astype(int).values
    return X_train, X_test, y, test_ids, cat_cols, num_cols


# ---------------------------------------------------------------------------
# Ordinal encoding for XGBoost and HistGradientBoosting
# ---------------------------------------------------------------------------
def make_ordinal_views(X_train, X_test, cat_cols):
    """Return ordinal-encoded copies of X_train, X_test for XGB and HGB."""
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    enc.fit(pd.concat([X_train[cat_cols], X_test[cat_cols]], axis=0))
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[cat_cols] = enc.transform(X_train[cat_cols])
    X_te[cat_cols] = enc.transform(X_test[cat_cols])
    # Make sure entire frame is float32 for speed.
    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Model factories - parameters chosen to match prior CatBoost tuning notes.
# ---------------------------------------------------------------------------
def make_catboost():
    return CatBoostClassifier(
        iterations=7000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5.0,
        random_strength=1.2,
        random_seed=SEED,
        eval_metric="Accuracy",
        early_stopping_rounds=150,
        verbose=False,
        allow_writing_files=False,
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.0,
        tree_method="hist",
        eval_metric="error",
        early_stopping_rounds=150,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=2000,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=SEED,
    )


# ---------------------------------------------------------------------------
# Cross-validation training loop
# ---------------------------------------------------------------------------
def run_cv(X_train, X_test, y, cat_cols):
    n_train = len(X_train)
    n_test = len(X_test)

    # Ordinal-encoded copies for XGB and HGB.
    X_tr_ord, X_te_ord = make_ordinal_views(X_train, X_test, cat_cols)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    oof = {
        "cb": np.zeros(n_train),
        "xgb": np.zeros(n_train),
        "hgb": np.zeros(n_train),
    }
    test_pred = {
        "cb": np.zeros(n_test),
        "xgb": np.zeros(n_test),
        "hgb": np.zeros(n_test),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")
        t_fold = time.time()

        y_tr, y_va = y[tr_idx], y[va_idx]

        # -------- CatBoost --------
        t0 = time.time()
        cb = make_catboost()
        train_pool = Pool(X_train.iloc[tr_idx], y_tr, cat_features=cat_idx)
        val_pool = Pool(X_train.iloc[va_idx], y_va, cat_features=cat_idx)
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        p_va_cb = cb.predict_proba(val_pool)[:, 1]
        p_te_cb = cb.predict_proba(
            Pool(X_test, cat_features=cat_idx)
        )[:, 1]
        oof["cb"][va_idx] = p_va_cb
        test_pred["cb"] += p_te_cb / N_SPLITS
        acc_cb = accuracy_score(y_va, (p_va_cb >= 0.5).astype(int))
        print(f"  CatBoost  best_iter={cb.get_best_iteration():>4d}  "
              f"val_acc={acc_cb:.5f}  t={time.time()-t0:.1f}s")

        # -------- XGBoost --------
        t0 = time.time()
        xgb = make_xgb()
        xgb.fit(
            X_tr_ord.iloc[tr_idx],
            y_tr,
            eval_set=[(X_tr_ord.iloc[va_idx], y_va)],
            verbose=False,
        )
        p_va_xgb = xgb.predict_proba(X_tr_ord.iloc[va_idx])[:, 1]
        p_te_xgb = xgb.predict_proba(X_te_ord)[:, 1]
        oof["xgb"][va_idx] = p_va_xgb
        test_pred["xgb"] += p_te_xgb / N_SPLITS
        acc_xgb = accuracy_score(y_va, (p_va_xgb >= 0.5).astype(int))
        print(f"  XGBoost   best_iter={xgb.best_iteration:>4d}  "
              f"val_acc={acc_xgb:.5f}  t={time.time()-t0:.1f}s")

        # -------- HistGradientBoosting --------
        t0 = time.time()
        hgb = make_hgb()
        hgb.fit(X_tr_ord.iloc[tr_idx], y_tr)
        p_va_hgb = hgb.predict_proba(X_tr_ord.iloc[va_idx])[:, 1]
        p_te_hgb = hgb.predict_proba(X_te_ord)[:, 1]
        oof["hgb"][va_idx] = p_va_hgb
        test_pred["hgb"] += p_te_hgb / N_SPLITS
        acc_hgb = accuracy_score(y_va, (p_va_hgb >= 0.5).astype(int))
        print(f"  HistGB    n_iter={hgb.n_iter_:>4d}  "
              f"val_acc={acc_hgb:.5f}  t={time.time()-t0:.1f}s")

        fold_metrics.append({
            "fold": fold,
            "catboost": {"val_acc": float(acc_cb),
                         "best_iter": int(cb.get_best_iteration())},
            "xgboost":  {"val_acc": float(acc_xgb),
                         "best_iter": int(xgb.best_iteration)},
            "hgb":      {"val_acc": float(acc_hgb),
                         "n_iter": int(hgb.n_iter_)},
            "fold_seconds": round(time.time() - t_fold, 2),
        })

    return oof, test_pred, fold_metrics


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------
def threshold_sweep(probas, y, grid=None):
    if grid is None:
        grid = np.arange(0.30, 0.55 + 1e-9, 0.005)
    accs = [accuracy_score(y, (probas >= t).astype(int)) for t in grid]
    accs = np.array(accs)
    best_idx = int(np.argmax(accs))
    return {
        "best_threshold": float(grid[best_idx]),
        "best_accuracy": float(accs[best_idx]),
        "grid": grid.tolist(),
        "accuracies": accs.tolist(),
    }


def per_side_thresholds(probas, y, side_array):
    out = {}
    for side_val in sorted(np.unique(side_array)):
        mask = side_array == side_val
        if mask.sum() < 50:
            continue
        sweep = threshold_sweep(probas[mask], y[mask])
        out[str(side_val)] = {
            "n": int(mask.sum()),
            "best_threshold": sweep["best_threshold"],
            "best_accuracy": sweep["best_accuracy"],
            "acc_at_050": float(accuracy_score(
                y[mask], (probas[mask] >= 0.5).astype(int)
            )),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("DATA_DIR =", DATA_DIR.resolve())
    check_files()

    print("Loading data ...")
    X_train, X_test, y, test_ids, cat_cols, num_cols = load_data()
    print(f"  X_train shape = {X_train.shape}")
    print(f"  X_test  shape = {X_test.shape}")
    print(f"  y mean        = {y.mean():.4f}")
    print(f"  cat cols      = {len(cat_cols)}")

    t_total = time.time()
    oof, test_pred, fold_metrics = run_cv(X_train, X_test, y, cat_cols)
    print(f"\nCV finished in {time.time()-t_total:.1f}s")

    # Soft voting ensemble.
    oof_ensemble = (oof["cb"] + oof["xgb"] + oof["hgb"]) / 3.0
    test_ensemble = (
        test_pred["cb"] + test_pred["xgb"] + test_pred["hgb"]
    ) / 3.0

    print("\n----- OOF accuracy at threshold 0.50 -----")
    for k, p in oof.items():
        acc = accuracy_score(y, (p >= 0.5).astype(int))
        print(f"  {k:<3s}: {acc:.5f}")
    acc_ens_050 = accuracy_score(y, (oof_ensemble >= 0.5).astype(int))
    print(f"  ens: {acc_ens_050:.5f}")

    # Global threshold sweep.
    print("\n----- Threshold sweep on ensemble OOF -----")
    sweep_global = threshold_sweep(oof_ensemble, y)
    print(f"  best threshold = {sweep_global['best_threshold']:.3f}")
    print(f"  best OOF acc   = {sweep_global['best_accuracy']:.5f}")
    print(f"  gain vs 0.50   = "
          f"{sweep_global['best_accuracy'] - acc_ens_050:+.5f}")

    # Per-Side threshold sweep.
    side_train = X_train["Side"].astype(str).values
    side_test = X_test["Side"].astype(str).values
    print("\n----- Per-Side threshold sweep on ensemble OOF -----")
    sweep_per_side = per_side_thresholds(oof_ensemble, y, side_train)
    for side_val, info in sweep_per_side.items():
        print(f"  Side={side_val}: n={info['n']}, "
              f"thr={info['best_threshold']:.3f}, "
              f"best_acc={info['best_accuracy']:.5f}, "
              f"acc@0.50={info['acc_at_050']:.5f}")

    # Save OOF and test probabilities.
    np.savez(
        OUT_DIR / "oof_probas.npz",
        cb=oof["cb"], xgb=oof["xgb"], hgb=oof["hgb"], ensemble=oof_ensemble,
        y=y,
    )
    np.savez(
        OUT_DIR / "test_probas.npz",
        cb=test_pred["cb"], xgb=test_pred["xgb"], hgb=test_pred["hgb"],
        ensemble=test_ensemble,
    )

    # Fold metrics and threshold analysis JSON.
    with open(OUT_DIR / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2, ensure_ascii=False)

    threshold_analysis = {
        "ensemble_acc_at_050": float(acc_ens_050),
        "global_best_threshold": sweep_global["best_threshold"],
        "global_best_oof_acc": sweep_global["best_accuracy"],
        "per_side": sweep_per_side,
        "oof_acc_each_model_at_050": {
            k: float(accuracy_score(y, (p >= 0.5).astype(int)))
            for k, p in oof.items()
        },
    }
    with open(OUT_DIR / "threshold_analysis.json", "w", encoding="utf-8") as f:
        json.dump(threshold_analysis, f, indent=2, ensure_ascii=False)

    # Submission with global optimal threshold.
    thr = sweep_global["best_threshold"]
    sub = pd.DataFrame({
        "PassengerId": test_ids["PassengerId"],
        "Transported": (test_ensemble >= thr).astype(bool),
    })
    sub.to_csv(OUT_DIR / "submission_ensemble.csv", index=False)
    print(f"\nWrote submission_ensemble.csv (threshold={thr:.3f}, "
          f"True count={int(sub['Transported'].sum())})")

    # Per-Side submission.
    perside_thr = {
        side_val: info["best_threshold"]
        for side_val, info in sweep_per_side.items()
    }
    pred_perside = np.zeros(len(X_test), dtype=int)
    for i, s in enumerate(side_test):
        t = perside_thr.get(s, thr)
        pred_perside[i] = int(test_ensemble[i] >= t)
    sub_ps = pd.DataFrame({
        "PassengerId": test_ids["PassengerId"],
        "Transported": pred_perside.astype(bool),
    })
    sub_ps.to_csv(OUT_DIR / "submission_ensemble_perside.csv", index=False)
    print(f"Wrote submission_ensemble_perside.csv "
          f"(thresholds={perside_thr}, "
          f"True count={int(sub_ps['Transported'].sum())})")

    print("\nDone. Files written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
