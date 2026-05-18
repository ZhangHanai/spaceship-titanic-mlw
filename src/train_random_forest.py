"""Train the Random Forest model for the Kaggle Spaceship Titanic competition.

Run from the repository root:
    python src/train_random_forest.py

This script integrates the teammate Random Forest implementation into the
shared project layout while preserving its original feature engineering,
model hyperparameters, cross-validation, validation evaluation, feature
importance reporting, and Kaggle submission generation strategy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import DATA_DIR, FIGURES_DIR, OUTPUT_DIR, ensure_directory, require_file

RANDOM_STATE = 42
SPENDING_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLUMNS = ["HomePlanet", "Destination", "CryoSleep", "VIP"]
DROP_COLUMNS = ["PassengerId", "Name", "Cabin"]
TARGET_COLUMN = "Transported"


def preprocess_random_forest_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Apply the teammate Random Forest preprocessing logic to one data frame.

    The original implementation processed train and test frames independently,
    including Age median and categorical mode imputation. That behavior is kept
    here to preserve the uploaded model logic as closely as possible.
    """
    data = df.copy()

    # Economic consumption features: missing means not used, fill with 0.
    for col in SPENDING_COLUMNS:
        data[col] = data[col].fillna(0)

    # Create TotalSpending before log transformation.
    data["TotalSpending"] = data[SPENDING_COLUMNS].sum(axis=1)

    # Log1p transformation to reduce right skewness.
    for col in [*SPENDING_COLUMNS, "TotalSpending"]:
        data[col] = np.log1p(data[col])

    # Age: fill missing with median, then discretize into four ordered groups.
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-1, 12, 18, 60, 100],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Drop original Age after creating AgeGroup.
    data = data.drop(columns=["Age"])

    # Categorical variables: mode imputation.
    for col in CATEGORICAL_COLUMNS:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Drop irrelevant features.
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])

    # Convert target variable.
    if is_train:
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    return data


def build_random_forest_pipeline() -> Pipeline:
    """Build the teammate Random Forest model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ],
        remainder="passthrough",
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        criterion="gini",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_model),
        ]
    )


def get_feature_importance(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return feature importances with post-OneHotEncoder feature names."""
    encoded_cat_features = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_COLUMNS)
    )
    numeric_features = [col for col in X.columns if col not in CATEGORICAL_COLUMNS]
    feature_names = list(encoded_cat_features) + numeric_features
    importances = model.named_steps["classifier"].feature_importances_

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> Path:
    """Save the validation confusion matrix figure."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Transported", "Transported"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def save_feature_importance_plot(feature_importance_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> Path:
    """Save a horizontal bar chart of the top feature importances."""
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance Ranking Chart of Random Forest Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the Random Forest Spaceship Titanic model.")
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path, help="Directory containing train.csv/test.csv.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path, help="Directory for saved CSV outputs.")
    parser.add_argument("--figures-dir", default=FIGURES_DIR, type=Path, help="Directory for saved plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    figures_dir = ensure_directory(args.figures_dir)

    train_path = require_file(args.data_dir / "train.csv", "Download the Kaggle train.csv into data/.")
    test_path = require_file(args.data_dir / "test.csv", "Download the Kaggle test.csv into data/.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_passenger_id = test_df["PassengerId"].copy()

    print("Training set shape:", train_df.shape)
    print("Test set shape:", test_df.shape)

    train_processed = preprocess_random_forest_data(train_df, is_train=True)
    test_processed = preprocess_random_forest_data(test_df, is_train=False)

    X = train_processed.drop(columns=[TARGET_COLUMN])
    y = train_processed[TARGET_COLUMN]
    X_test = test_processed

    print("Processed training features shape:", X.shape)
    print("Processed test features shape:", X_test.shape)
    print("Target shape:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_random_forest_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    start_time = time.time()
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("\n5-Fold CV Scores:", cv_scores)
    print(f"Mean CV Accuracy: {cv_scores.mean():.5f}")
    print(f"CV Accuracy Std: {cv_scores.std():.5f}")

    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    report = classification_report(y_val, val_pred, target_names=["Not Transported", "Transported"])
    cm = confusion_matrix(y_val, val_pred)

    print(f"\nValidation Accuracy: {val_accuracy:.5f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"Training time: {training_time:.2f} seconds")

    summary_path = output_dir / "random_forest_validation_summary.csv"
    cv_columns = {f"cv_fold_{idx + 1}_accuracy": score for idx, score in enumerate(cv_scores)}
    summary_df = pd.DataFrame(
        [
            {
                **cv_columns,
                "mean_cv_accuracy": cv_scores.mean(),
                "std_cv_accuracy": cv_scores.std(),
                "validation_accuracy": val_accuracy,
                "training_time_seconds": training_time,
            }
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    confusion_matrix_path = figures_dir / "random_forest_confusion_matrix.png"
    save_confusion_matrix_plot(cm, confusion_matrix_path)

    feature_importance_df = get_feature_importance(model, X)
    feature_importance_path = output_dir / "random_forest_feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)

    print("\nTop 15 Feature Importances:")
    print(feature_importance_df.head(15))

    feature_importance_plot_path = figures_dir / "random_forest_feature_importance.png"
    save_feature_importance_plot(feature_importance_df, feature_importance_plot_path)

    # Train on full dataset and predict the Kaggle test set, matching the original script.
    model.fit(X, y)
    test_pred = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "PassengerId": test_passenger_id,
            "Transported": test_pred.astype(bool),
        }
    )
    submission_path = output_dir / "random_forest_submission.csv"
    submission.to_csv(submission_path, index=False)

    print("\nOutput files saved:")
    print(f"- Validation summary: {summary_path}")
    print(f"- Feature importance CSV: {feature_importance_path}")
    print(f"- Confusion matrix plot: {confusion_matrix_path}")
    print(f"- Feature importance plot: {feature_importance_plot_path}")
    print(f"- Kaggle submission: {submission_path}")


if __name__ == "__main__":
    main()
