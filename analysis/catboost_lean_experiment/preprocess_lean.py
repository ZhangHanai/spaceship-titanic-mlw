"""Lean preprocessing for Spaceship Titanic CatBoost analysis experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import DATA_DIR, OUTPUT_DIR, ensure_directory, require_file

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _group_mode_fill(df: pd.DataFrame, col: str, by: str | list[str]) -> pd.Series:
    def first_mode(series: pd.Series):
        modes = series.dropna().mode()
        return modes.iloc[0] if len(modes) else np.nan

    fill = df.groupby(by, dropna=False)[col].transform(first_mode)
    return df[col].fillna(fill)


def _group_median_fill(df: pd.DataFrame, col: str, by: str | list[str]) -> pd.Series:
    fill = df.groupby(by, dropna=False)[col].transform("median")
    return df[col].fillna(fill)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    y = train["Transported"].astype(int).values
    n_train = len(train)

    df = pd.concat([train.drop(columns=["Transported"]), test], axis=0, ignore_index=True)

    pid = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = pid[0].astype(int)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    df["Surname"] = df["Name"].astype(str).str.split().str[-1]
    df.loc[df["Name"].isna(), "Surname"] = np.nan

    for col in SPEND_COLS:
        df.loc[(df["CryoSleep"] == True) & (df[col].isna()), col] = 0.0

    known_spend = df[SPEND_COLS].fillna(0).sum(axis=1)
    df.loc[(known_spend > 0) & df["CryoSleep"].isna(), "CryoSleep"] = False
    all_known_spend = df[SPEND_COLS].notna().all(axis=1)
    df.loc[df["CryoSleep"].isna() & all_known_spend & (known_spend == 0), "CryoSleep"] = True

    df["HomePlanet"] = _group_mode_fill(df, "HomePlanet", "Group")
    df["HomePlanet"] = _group_mode_fill(df, "HomePlanet", "Surname")
    df["HomePlanet"] = _group_mode_fill(df, "HomePlanet", "Deck")
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode().iloc[0])

    df["Destination"] = _group_mode_fill(df, "Destination", "Group")
    df["Destination"] = _group_mode_fill(df, "Destination", "Surname")
    df["Destination"] = df["Destination"].fillna(df["Destination"].mode().iloc[0])

    df["Side"] = _group_mode_fill(df, "Side", "Group")
    df["Side"] = df["Side"].fillna(df["Side"].mode().iloc[0])

    df["Deck"] = _group_mode_fill(df, "Deck", "Group")
    df["Deck"] = _group_mode_fill(df, "Deck", ["HomePlanet", "Side"])
    df["Deck"] = df["Deck"].fillna(df["Deck"].mode().iloc[0])

    df["CabinNum"] = _group_median_fill(df, "CabinNum", ["Deck", "Side"])
    df["CabinNum"] = _group_median_fill(df, "CabinNum", "Deck")
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())

    for col in SPEND_COLS:
        df[col] = _group_median_fill(df, col, ["HomePlanet", "CryoSleep"])
        df[col] = _group_median_fill(df, col, "CryoSleep")
        df[col] = df[col].fillna(df[col].median())

    df["CryoSleep"] = df["CryoSleep"].astype("object").where(~df["CryoSleep"].isna(), "Unknown")
    df["CryoSleep"] = df["CryoSleep"].astype(str)
    df["VIP"] = df["VIP"].fillna(False).astype(str)

    df["Age"] = _group_median_fill(df, "Age", ["HomePlanet", "CryoSleep"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["IsChild"] = (df["Age"] < 13).astype(int)

    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["LeisureSpend"] = df["FoodCourt"] + df["ShoppingMall"]

    df["GroupTotalSpend"] = df.groupby("Group")["TotalSpend"].transform("sum")
    df["GroupChildrenCount"] = df.groupby("Group")["IsChild"].transform("sum")

    feature_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Age", "IsChild", "Deck", "CabinNum", "Side", "Group", "GroupSize", "IsAlone", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpend", "NoSpend", "LogTotalSpend", "LuxurySpend", "LeisureSpend", "GroupTotalSpend", "GroupChildrenCount"]
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("Missing").replace({"nan": "Missing"})

    x_full = df[feature_cols].copy()
    x_train = x_full.iloc[:n_train].reset_index(drop=True)
    x_test = x_full.iloc[n_train:].reset_index(drop=True)
    ids_test = df["PassengerId"].iloc[n_train:].reset_index(drop=True)
    return x_train, x_test, y, ids_test, feature_cols, cat_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lean CatBoost features.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing train.csv/test.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "catboost_lean", help="Output directory for lean features")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = ensure_directory(args.output_dir)

    train_path = require_file(data_dir / "train.csv")
    test_path = require_file(data_dir / "test.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    x_train, x_test, y, test_ids, feature_cols, cat_cols = build_features(train, test)

    x_train.to_csv(output_dir / "X_train_lean.csv", index=False)
    x_test.to_csv(output_dir / "X_test_lean.csv", index=False)
    pd.DataFrame({"Transported": y}).to_csv(output_dir / "y_train_lean.csv", index=False)
    pd.DataFrame({"PassengerId": test_ids}).to_csv(output_dir / "test_passenger_ids.csv", index=False)
    (output_dir / "lean_feature_metadata.json").write_text(
        json.dumps({"features": feature_cols, "categorical": cat_cols}, indent=2), encoding="utf-8"
    )
    print(f"Saved lean features to: {output_dir}")


if __name__ == "__main__":
    main()
