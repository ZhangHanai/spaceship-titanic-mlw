"""Preprocessing functions for Spaceship Titanic model scripts."""

from __future__ import annotations

import pandas as pd
import numpy as np

SVM_MONETARY_FEATURES = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SVM_CATEGORICAL_FEATURES = ["HomePlanet", "Destination", "CryoSleep", "VIP"]
SVM_NUMERIC_FEATURES = [*SVM_MONETARY_FEATURES, "AgeGroup"]
SVM_DROP_FEATURES = ["Name", "PassengerId", "Cabin"]


def preprocess_svm_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply the SVM notebook preprocessing strategy to raw Kaggle CSV files.

    Steps are intentionally kept equivalent to the original notebook:
    monetary missing values are set to zero, monetary features are log1p
    transformed, selected categorical values use the training mode, Age uses
    the training median and is binned into four ordinal age groups, and Name,
    PassengerId, Cabin, and original Age are removed from model features.
    """
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    test_passenger_id = test_processed["PassengerId"].copy()
    y = train_processed["Transported"].astype(int)
    train_processed = train_processed.drop(columns=["Transported"])

    for col in SVM_MONETARY_FEATURES:
        train_processed[col] = train_processed[col].fillna(0)
        test_processed[col] = test_processed[col].fillna(0)
        train_processed[col] = np.log1p(train_processed[col])
        test_processed[col] = np.log1p(test_processed[col])

    for col in SVM_CATEGORICAL_FEATURES:
        mode_value = train_processed[col].mode()[0]
        train_processed[col] = train_processed[col].fillna(mode_value).infer_objects(copy=False)
        test_processed[col] = test_processed[col].fillna(mode_value).infer_objects(copy=False)

    age_median = train_processed["Age"].median()
    train_processed["Age"] = train_processed["Age"].fillna(age_median)
    test_processed["Age"] = test_processed["Age"].fillna(age_median)

    age_bins = [-1, 12, 25, 60, np.inf]
    age_labels = [0, 1, 2, 3]
    train_processed["AgeGroup"] = pd.cut(
        train_processed["Age"], bins=age_bins, labels=age_labels
    ).astype(int)
    test_processed["AgeGroup"] = pd.cut(
        test_processed["Age"], bins=age_bins, labels=age_labels
    ).astype(int)

    train_processed = train_processed.drop(columns=["Age", *SVM_DROP_FEATURES])
    test_processed = test_processed.drop(columns=["Age", *SVM_DROP_FEATURES])

    return train_processed, test_processed, y, test_passenger_id


def align_categorical_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align train/test categorical dtypes for LightGBM native categories."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        train_df[col] = train_df[col].astype("string").fillna("Missing")
        test_df[col] = test_df[col].astype("string").fillna("Missing")

        all_categories = pd.Index(pd.concat([train_df[col], test_df[col]], axis=0).unique())
        train_df[col] = pd.Categorical(train_df[col], categories=all_categories)
        test_df[col] = pd.Categorical(test_df[col], categories=all_categories)

    return train_df, test_df
