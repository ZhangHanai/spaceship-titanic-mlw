"""
XGBoost training script for AI3023 Spaceship Titanic project.

What this script does:
1. Reads Kaggle Spaceship Titanic train/test CSV files.
2. Performs preprocessing and feature engineering.
3. Tunes an XGBoost model with RandomizedSearchCV.
4. Evaluates with Stratified K-Fold cross-validation.
5. Trains the final model and creates a Kaggle submission CSV.

Run:
    pip install pandas numpy scikit-learn xgboost
    python src/train_xgboost.py

Fast debug run:
    python src/train_xgboost.py --fast
"""

import argparse
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

from utils import DATA_DIR, OUTPUT_DIR, require_file, ensure_directory

RANDOM_STATE = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


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
    """Extract non-leaky name features."""
    out = df.copy()
    name = out["Name"].fillna("Unknown Unknown").astype(str)
    out["Surname"] = name.str.split().str[-1]
    out.loc[out["Name"].isna(), "Surname"] = "Unknown"
    out["NameWordCount"] = name.str.split().str.len().astype(float)
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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe engineered features for one dataset only."""
    out = df.copy()
    out = split_passenger_id(out)
    out = split_cabin(out)
    out = split_name(out)
    out = add_missing_indicators(out)
    out = rule_based_imputation(out)
    out = add_spending_features(out)
    out = add_age_features(out)
    out = add_combination_features(out)

    drop_cols = ["PassengerId", "Cabin", "Name", "Surname", "Group"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(object).where(pd.notna(out[col]), np.nan)
            out[col] = out[col].map({True: "True", False: "False"}).fillna(np.nan)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        else:
            out[col] = out[col].astype(object).where(pd.notna(out[col]), np.nan)

    return out


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
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="Directory containing train.csv/test.csv.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path, help="Directory for saved outputs.")
    parser.add_argument("--fast", action="store_true", help="Use fewer iterations for quick debugging")
    parser.add_argument("--make-submission", action="store_true", help="Train on full train.csv and write Kaggle submission.")
    args = parser.parse_args()

    output_dir = ensure_directory(args.output_dir)

    train_path = require_file(args.data_dir / "train.csv", "Download the Kaggle train.csv into data/.")
    test_path = require_file(args.data_dir / "test.csv", "Download the Kaggle test.csv into data/.")

    print(f"Reading training data from: {train_path}")
    print(f"Reading test data from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target positive rate: {train_df['Transported'].mean():.4f}")

    print("\nCreating leakage-safe training features...")
    X = create_features(train_df.drop(columns=["Transported"]))
    y = train_df["Transported"].astype(int).values
    print(f"Feature matrix shape: {X.shape}")

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
    cv_results_path = output_dir / "xgboost_cv_results.csv"
    cv_results.to_csv(cv_results_path, index=False)
    print(f"Saved CV results to: {cv_results_path}")

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

    submission_path = pd.NA
    if args.make_submission:
        print("\nTraining final model on all training data for Kaggle submission...")
        best_model.set_params(clf__n_jobs=1)
        best_model.fit(X, y)

        X_test = create_features(test_df)
        print(f"Test feature matrix shape: {X_test.shape}")
        test_prob = best_model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(bool)

        submission = pd.DataFrame(
            {
                "PassengerId": test_df["PassengerId"],
                "Transported": test_pred,
            }
        )

        output_path = output_dir / "xgboost_submission.csv"
        submission.to_csv(output_path, index=False)
        submission_path = "outputs/xgboost_submission.csv"

        print(f"\nSaved Kaggle submission to: {output_path}")
        print("Prediction distribution:")
        print(submission["Transported"].value_counts())
        print(f"Predicted transported rate: {submission['Transported'].mean():.4f}")

    metrics_summary = pd.DataFrame(
        [
            {
                "validation_accuracy": float(oof_acc),
                "best_threshold": float(threshold),
                "search_best_cv_accuracy": float(search.best_score_),
                "training_time_seconds": float(elapsed),
                "kaggle_public_score": pd.NA,
                "submission_file": submission_path,
            }
        ]
    )
    metrics_path = output_dir / "xgboost_validation_summary.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"Saved XGBoost summary to: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
