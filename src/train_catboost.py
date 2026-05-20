"""Train the CatBoost baseline migrated from Untitled11-checkpoint.ipynb.

Run from repository root:
    python src/train_catboost.py

This script reproduces the progress-stage CatBoost baseline from
`Untitled11-checkpoint.ipynb`. To keep reported numbers consistent, it retains
that notebook's train/test combined imputation and second-pass threshold search.
These choices are preserved for baseline reproducibility and are not presented as
strictly leakage-safe pipeline design.

Main reproducible standalone public score for this baseline is about 0.80547.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from metrics import threshold_search
from utils import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, ensure_directory, require_file

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:  # pragma: no cover
    raise ImportError("CatBoost is required. Install dependencies with: pip install -r requirements.txt") from exc

FOLDS = 5
SEEDS = [42, 123]

CATBOOST_PARAMS = {
    "iterations": 7000,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.2,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "early_stopping_rounds": 100,
    "verbose": 200,
    "use_best_model": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train progress-stage CatBoost baseline for Spaceship Titanic.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[["Deck", "Cabin_Num", "Side"]] = data["Cabin"].str.split("/", expand=True)
    data[["Group", "Id"]] = data["PassengerId"].str.split("_", expand=True)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    data["Total_Spending"] = data[spend_cols].sum(axis=1)
    data["Has_Spending"] = (data["Total_Spending"] > 0).astype(int)
    data["Family_Size"] = data.groupby("Group")["Group"].transform("count")
    data["Is_Alone"] = (data["Family_Size"] == 1).astype(int)
    return data


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    ensure_directory(args.figures_dir)

    train_path = require_file(args.data_dir / "train.csv")
    test_path = require_file(args.data_dir / "test.csv")
    sample_sub_path = args.data_dir / "sample_submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_fe = feature_engineering(train)
    test_fe = feature_engineering(test)

    # Keep notebook behavior intentionally for progress-stage baseline reproduction.
    all_data = pd.concat([train_fe, test_fe], ignore_index=True)

    num_cols = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "Cabin_Num", "Total_Spending", "Family_Size",
    ]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Group", "Id"]

    for col in num_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
        all_data[col] = all_data[col].fillna(all_data[col].median())

    for col in cat_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    train_df = all_data[all_data["Transported"].notna()].copy()
    test_df = all_data[all_data["Transported"].isna()].copy()

    drop_cols = ["PassengerId", "Cabin", "Name", "Transported"]
    X = train_df.drop(columns=drop_cols)
    y = train_df["Transported"].astype(int)
    X_test = test_df.drop(columns=drop_cols)

    cat_features = [X.columns.get_loc(col) for col in cat_cols]

    test_pred_final = np.zeros(len(X_test))
    fold_rows: list[dict[str, float | int]] = []
    feature_importances = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        test_pred_probs = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            test_pool = Pool(X_test, cat_features=cat_features)

            model = CatBoostClassifier(**CATBOOST_PARAMS, random_state=seed)
            start_time = time.time()
            model.fit(train_pool, eval_set=val_pool)
            fold_time = time.time() - start_time

            val_pred = model.predict(val_pool)
            acc = accuracy_score(y_val, val_pred)
            best_iteration = int(model.get_best_iteration())

            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "validation_accuracy": float(acc),
                    "best_iteration": best_iteration,
                    "training_time_seconds": float(fold_time),
                }
            )

            test_pred_probs += model.predict_proba(test_pool)[:, 1] / FOLDS
            feature_importances.append(model.get_feature_importance(train_pool))

        test_pred_final += test_pred_probs / len(SEEDS)

    # Second-pass threshold search retained from notebook logic using seed=42 OOF.
    oof_probs = np.zeros(len(X))
    skf_base = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for train_idx, val_idx in skf_base.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**CATBOOST_PARAMS, random_state=42)
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        model.fit(train_pool, eval_set=val_pool)
        oof_probs[val_idx] = model.predict_proba(val_pool)[:, 1]

    best_threshold, best_acc, threshold_df = threshold_search(y, oof_probs, start=0.42, stop=0.58, step=0.001)

    oof_output = pd.DataFrame(
        {
            "PassengerId": train["PassengerId"],
            "y_true": y,
            "oof_probability": oof_probs,
            "prediction_at_best_threshold": (oof_probs >= best_threshold).astype(int),
        }
    )

    if sample_sub_path.exists():
        submission = pd.read_csv(sample_sub_path)
        submission["Transported"] = (test_pred_final >= best_threshold).astype(bool)
    else:
        submission = pd.DataFrame(
            {
                "PassengerId": test["PassengerId"],
                "Transported": (test_pred_final >= best_threshold).astype(bool),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    mean_importance = np.mean(np.vstack(feature_importances), axis=0)
    fi_df = pd.DataFrame({"feature": X.columns, "importance": mean_importance}).sort_values("importance", ascending=False)

    submission.to_csv(output_dir / "submission_catboost_v1.csv", index=False)
    oof_output.to_csv(output_dir / "catboost_oof_predictions.csv", index=False)
    fold_df.to_csv(output_dir / "catboost_fold_results.csv", index=False)
    fi_df.to_csv(output_dir / "catboost_feature_importance.csv", index=False)
    threshold_df.to_csv(output_dir / "catboost_threshold_search.csv", index=False)

    print(f"Best threshold: {best_threshold:.3f}, OOF accuracy: {best_acc:.5f}")


if __name__ == "__main__":
    main()
