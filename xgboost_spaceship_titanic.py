"""
XGBoost solution for AI3023 Spaceship Titanic project.

What this script does:
1. Reads Kaggle Spaceship Titanic train/test CSV files.
2. Performs preprocessing and feature engineering.
3. Tunes an XGBoost model with RandomizedSearchCV.
4. Evaluates with Stratified K-Fold cross-validation.
5. Trains the final model and creates a Kaggle submission CSV.

Run:
    pip install pandas numpy scikit-learn xgboost
    python xgboost_spaceship_titanic.py --train train.csv --test test.csv

Fast debug run:
    python xgboost_spaceship_titanic.py --train train.csv --test test.csv --fast
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint, uniform

from xgboost import XGBClassifier


RANDOM_STATE = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def resolve_path(input_path: str, candidates: list[str]) -> str:
    """Use the given path if it exists; otherwise try common fallback paths."""
    if input_path and os.path.exists(input_path):
        return input_path

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find file: {input_path}. Tried fallbacks: {candidates}"
    )


def split_passenger_id(df: pd.DataFrame) -> pd.DataFrame:
    """Extract group-level features from PassengerId."""
    out = df.copy()
    out["Group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["PassengerNo"] = out["PassengerId"].astype(str).str.split("_").str[1].astype(int)
    out["GroupSize"] = out.groupby("Group")["PassengerId"].transform("count")
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    return out


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin into deck, number and side."""
    out = df.copy()
    cabin_split = out["Cabin"].astype("object").str.split("/", expand=True)
    out["CabinDeck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce").astype(float)
    out["CabinSide"] = cabin_split[2]

    out["CabinNumBin"] = pd.cut(
        out["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 1500, 1800, 2400],
        labels=False,
    ).astype(float)
    return out


def split_name(df: pd.DataFrame) -> pd.DataFrame:
    """Extract surname and family size from Name."""
    out = df.copy()
    name = out["Name"].fillna("Unknown Unknown").astype(str)
    out["Surname"] = name.str.split().str[-1]
    out.loc[out["Name"].isna(), "Surname"] = "Unknown"

    out["FamilySize"] = out.groupby("Surname")["PassengerId"].transform("count")
    out.loc[out["Surname"].eq("Unknown"), "FamilySize"] = 1
    out["HasFamily"] = (out["FamilySize"] > 1).astype(int)

    # Reduce high-cardinality names.
    surname_counts = out["Surname"].value_counts()
    common_surnames = set(surname_counts[surname_counts >= 3].index)
    out["SurnameGroup"] = out["Surname"].where(
        out["Surname"].isin(common_surnames), "RareSurname"
    )
    return out


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing-value indicator features before imputation."""
    out = df.copy()
    raw_cols = [
        "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name"
    ]
    for col in raw_cols:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def rule_based_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple domain rules:
    - If a passenger spent money, they were probably not in CryoSleep.
    - If CryoSleep is True, missing spending values are likely zero.
    """
    out = df.copy()

    spend_sum = out[SPEND_COLS].fillna(0).sum(axis=1)
    out.loc[out["CryoSleep"].isna() & (spend_sum > 0), "CryoSleep"] = False

    for col in SPEND_COLS:
        out.loc[(out["CryoSleep"] == True) & out[col].isna(), col] = 0

    return out


def add_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create spending-related numerical features."""
    out = df.copy()
    spend = out[SPEND_COLS].fillna(0)

    out["TotalSpend"] = spend.sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["AnySpend"] = (out["TotalSpend"] > 0).astype(int)
    out["SpendCount"] = (spend > 0).sum(axis=1)

    out["LuxurySpend"] = spend[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    out["SocialSpend"] = spend[["FoodCourt", "ShoppingMall"]].sum(axis=1)
    out["MaxSpend"] = spend.max(axis=1)

    out["LuxuryRatio"] = out["LuxurySpend"] / (out["TotalSpend"] + 1)
    out["SocialRatio"] = out["SocialSpend"] / (out["TotalSpend"] + 1)

    for col in SPEND_COLS + ["TotalSpend", "LuxurySpend", "SocialSpend", "MaxSpend"]:
        out[f"Log_{col}"] = np.log1p(out[col].fillna(0))

    return out


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age-group features."""
    out = df.copy()
    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[-1, 5, 12, 18, 25, 35, 50, 65, 100],
        labels=False,
    ).astype(float)
    out["IsChild"] = (out["Age"] < 13).astype(int)
    out["IsTeen"] = ((out["Age"] >= 13) & (out["Age"] < 20)).astype(int)
    out["IsAdult"] = (out["Age"] >= 18).astype(int)
    return out


def add_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features for important categorical variables."""
    out = df.copy()

    def combine(a: str, b: str) -> pd.Series:
        return out[a].astype(str).fillna("Missing") + "_" + out[b].astype(str).fillna("Missing")

    out["Route"] = combine("HomePlanet", "Destination")
    out["DeckSide"] = combine("CabinDeck", "CabinSide")
    out["PlanetDeck"] = combine("HomePlanet", "CabinDeck")
    out["CryoNoSpend"] = out["CryoSleep"].astype(str) + "_" + out["NoSpend"].astype(str)
    return out


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add category frequency features using only non-target structure."""
    out = df.copy()
    freq_cols = [
        "Group", "HomePlanet", "Destination", "CabinDeck", "CabinSide",
        "CabinNumBin", "SurnameGroup", "Route", "DeckSide", "PlanetDeck"
    ]

    for col in freq_cols:
        vc = out[col].astype(str).value_counts(dropna=False)
        out[f"{col}_freq"] = out[col].astype(str).map(vc).astype(float)

    return out


def create_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Combine train and test for non-target feature engineering, then split back.
    This is common in Kaggle preprocessing because no target values from test are used.
    """
    train = train_df.drop(columns=["Transported"]).copy()
    test = test_df.copy()
    n_train = len(train)

    full = pd.concat([train, test], axis=0, ignore_index=True)

    full = split_passenger_id(full)
    full = split_cabin(full)
    full = split_name(full)
    full = add_missing_indicators(full)
    full = rule_based_imputation(full)
    full = add_spending_features(full)
    full = add_age_features(full)
    full = add_combination_features(full)
    full = add_frequency_features(full)

    # Drop raw columns that are too detailed or not directly model-friendly.
    drop_cols = ["PassengerId", "Cabin", "Name", "Surname", "Group"]
    full = full.drop(columns=[c for c in drop_cols if c in full.columns])

    # Convert pandas nullable missing values to np.nan so scikit-learn imputers work.
    # Keep numeric columns numeric; keep categorical columns as object.
    for col in full.columns:
        if pd.api.types.is_bool_dtype(full[col]):
            full[col] = full[col].astype(object).where(pd.notna(full[col]), np.nan)
            full[col] = full[col].map({True: "True", False: "False"}).fillna(np.nan)
        elif pd.api.types.is_numeric_dtype(full[col]):
            full[col] = pd.to_numeric(full[col], errors="coerce").astype(float)
        else:
            full[col] = full[col].astype(object).where(pd.notna(full[col]), np.nan)

    X = full.iloc[:n_train].reset_index(drop=True)
    X_test = full.iloc[n_train:].reset_index(drop=True)
    y = train_df["Transported"].astype(int).values

    return X, X_test, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing + XGBoost pipeline."""
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,  # avoid nested parallelism during RandomizedSearchCV
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", xgb_model),
        ]
    )

    return pipeline


def get_param_distributions(fast: bool = False) -> dict:
    """Hyperparameter search space for XGBoost."""
    if fast:
        return {
            "clf__n_estimators": [80, 120],
            "clf__max_depth": [3, 4],
            "clf__learning_rate": [0.03, 0.05],
            "clf__subsample": [0.85, 0.95],
            "clf__colsample_bytree": [0.80, 0.90],
            "clf__min_child_weight": [2, 4],
            "clf__reg_lambda": [3.0, 6.0],
            "clf__reg_alpha": [0.0, 0.1],
        }

    return {
        "clf__n_estimators": randint(300, 900),
        "clf__max_depth": randint(3, 7),
        "clf__learning_rate": uniform(0.015, 0.055),
        "clf__subsample": uniform(0.75, 0.25),
        "clf__colsample_bytree": uniform(0.70, 0.30),
        "clf__min_child_weight": randint(1, 8),
        "clf__gamma": uniform(0.0, 0.20),
        "clf__reg_lambda": uniform(3.0, 8.0),
        "clf__reg_alpha": uniform(0.0, 0.50),
    }


def find_best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Search the best classification threshold based on validation accuracy."""
    best_acc = -1.0
    best_threshold = 0.50

    for threshold in np.arange(0.35, 0.651, 0.001):
        pred = (prob >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--output", type=str, default="submission_xgboost.csv", help="Output submission CSV")
    parser.add_argument("--cv-results", type=str, default="xgboost_cv_results.csv", help="Output CV results CSV")
    parser.add_argument("--fast", action="store_true", help="Use fewer iterations for quick debugging")
    args = parser.parse_args()

    train_path = resolve_path(
        args.train,
        [
            "/mnt/data/train(3).csv",
            "/mnt/data/train(1).csv",
            "/mnt/data/train.csv",
            "train(3).csv",
            "train(1).csv",
            "train.csv",
        ],
    )
    test_path = resolve_path(
        args.test,
        [
            "/mnt/data/test(3).csv",
            "/mnt/data/test(1).csv",
            "/mnt/data/test.csv",
            "test(3).csv",
            "test(1).csv",
            "test.csv",
        ],
    )

    print(f"Reading training data from: {train_path}")
    print(f"Reading test data from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target positive rate: {train_df['Transported'].mean():.4f}")

    print("\nCreating features...")
    X, X_test, y = create_features(train_df, test_df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")

    pipeline = build_pipeline(X)

    cv_splits = 3 if args.fast else 5
    n_iter = 3 if args.fast else 15

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nRunning hyperparameter tuning...")
    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_distributions(fast=args.fast),
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=True,
    )

    search.fit(X, y)
    elapsed = time.time() - start_time

    print("\nBest CV accuracy from RandomizedSearchCV:")
    print(f"{search.best_score_:.5f}")
    print("\nBest parameters:")
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")
    print(f"\nTuning time: {elapsed:.2f} seconds")

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(args.cv_results, index=False)
    print(f"Saved CV results to: {args.cv_results}")

    print("\nCreating out-of-fold predictions with best model...")
    best_model = search.best_estimator_
    oof_prob = cross_val_predict(
        best_model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold, oof_acc = find_best_threshold(y, oof_prob)
    oof_pred = (oof_prob >= threshold).astype(int)

    print(f"\nOOF accuracy at threshold 0.500: {accuracy_score(y, oof_prob >= 0.5):.5f}")
    print(f"Best OOF threshold: {threshold:.3f}")
    print(f"Best OOF accuracy: {oof_acc:.5f}")

    print("\nClassification report:")
    print(classification_report(y, oof_pred, target_names=["Not transported", "Transported"]))

    print("Confusion matrix:")
    print(confusion_matrix(y, oof_pred))

    print("\nTraining final model on all training data...")
    best_model.set_params(clf__n_jobs=1)
    best_model.fit(X, y)

    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(bool)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Transported": test_pred,
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"\nSaved Kaggle submission to: {output_path}")
    print("Prediction distribution:")
    print(submission["Transported"].value_counts())
    print(f"Predicted transported rate: {submission['Transported'].mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
