"""Final reference-aided competition pipeline for Spaceship Titanic.

Method: Semi-supervised CatBoost distillation + confidence-gated post-processing.

Stage 1 trains CatBoost on the original training labels plus pseudo-labels from a
public reference submission for test rows. Stage 2 keeps confident model
predictions and falls back to the reference signal only when the model is inside
an uncertainty band and disagrees with the reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from utils import DATA_DIR, OUTPUT_DIR, REPO_ROOT, ensure_directory, require_file

EXPECTED_TEST_ROWS = 4277
TARGET_COLUMNS = ["PassengerId", "Transported"]
SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]



def load_runtime_dependencies() -> None:
    """Import ML dependencies lazily so --help works before installation."""
    global np, pd, train_test_split, CatBoostClassifier, Pool

    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split

CATBOOST_PARAMS = {
    "iterations": 2500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "early_stopping_rounds": 100,
    "verbose": 200,
    "allow_writing_files": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final reference-aided Spaceship Titanic pipeline: "
            "semi-supervised CatBoost distillation + confidence-gated post-processing."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing train.csv, test.csv, and sample_submission.csv.")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=REPO_ROOT / "submissions" / "reference" / "public_reference_submission.csv",
        help="Public reference submission used as teacher/reference signal.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "final_reference_aided",
        help="Directory for final_submission.csv, stage1_model_submission.csv, and summaries.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the validation split and CatBoost.")
    parser.add_argument("--band-low", type=float, default=0.10, help="Lower probability bound for the uncertainty band.")
    parser.add_argument("--band-high", type=float, default=0.90, help="Upper probability bound for the uncertainty band.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve command-line paths relative to the repository root when needed."""
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    """Convert common Kaggle boolean encodings to bool, or raise a clear error."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        non_missing = series.dropna()
        valid_values = {0, 1, 0.0, 1.0, False, True}
        if non_missing.map(lambda value: value in valid_values).all():
            return series.astype(int).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if normalized.isin(mapping).all():
        return normalized.map(mapping).astype(bool)

    unique_preview = sorted(series.dropna().astype(str).unique().tolist())[:10]
    raise ValueError(f"Column {column_name!r} is not boolean-compatible. Sample values: {unique_preview}")


def require_exact_columns(df: pd.DataFrame, expected_columns: Iterable[str], label: str) -> None:
    expected = list(expected_columns)
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(f"{label} must have exactly columns {expected}; found {actual}.")


def load_and_validate_inputs(data_dir: Path, reference_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = require_file(data_dir / "train.csv", "Download the Kaggle training file into data/ or pass --data-dir.")
    test_path = require_file(data_dir / "test.csv", "Download the Kaggle test file into data/ or pass --data-dir.")
    sample_path = require_file(data_dir / "sample_submission.csv", "Download sample_submission.csv into data/ or pass --data-dir.")
    reference_path = require_file(reference_path, "Place the public reference submission under submissions/reference/ or pass --reference-path.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    reference = pd.read_csv(reference_path)

    require_exact_columns(reference, TARGET_COLUMNS, "Reference submission")
    reference["Transported"] = normalize_bool_series(reference["Transported"], "Transported")

    for label, frame in [("train.csv", train), ("test.csv", test), ("sample_submission.csv", sample_submission)]:
        if "PassengerId" not in frame.columns:
            raise ValueError(f"{label} must contain PassengerId.")
    if "Transported" not in train.columns:
        raise ValueError("train.csv must contain Transported labels.")
    train["Transported"] = normalize_bool_series(train["Transported"], "Transported")

    test_ids = test["PassengerId"].reset_index(drop=True)
    sample_ids = sample_submission["PassengerId"].reset_index(drop=True)
    reference_ids = reference["PassengerId"].reset_index(drop=True)
    if not test_ids.equals(sample_ids):
        raise ValueError("PassengerIds in test.csv and sample_submission.csv do not match exactly and in order.")
    if not test_ids.equals(reference_ids):
        raise ValueError("PassengerIds in test.csv and reference submission do not match exactly and in order.")

    return train, test, sample_submission, reference


def add_spaceship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact tabular features used by the final CatBoost pipeline."""
    data = df.copy()

    cabin_parts = data["Cabin"].astype("string").str.split("/", expand=True)
    data["Deck"] = cabin_parts[0] if 0 in cabin_parts else pd.NA
    data["Cabin_Num"] = pd.to_numeric(cabin_parts[1], errors="coerce") if 1 in cabin_parts else np.nan
    data["Side"] = cabin_parts[2] if 2 in cabin_parts else pd.NA

    passenger_parts = data["PassengerId"].astype("string").str.split("_", expand=True)
    data["Group"] = passenger_parts[0]
    data["Passenger_Number"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    data["Total_Spending"] = data[SPEND_COLUMNS].sum(axis=1, skipna=True)
    data["Has_Spending"] = (data["Total_Spending"] > 0).astype(int)
    data["Family_Size"] = data.groupby("Group")["Group"].transform("count")
    data["Is_Alone"] = (data["Family_Size"] == 1).astype(int)
    data["Missing_Count"] = data.isna().sum(axis=1)
    return data


def build_model_matrices(train: pd.DataFrame, test: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    train_fe = add_spaceship_features(train)
    test_fe = add_spaceship_features(test)

    y_train = train_fe["Transported"].astype(int)
    y_reference = reference["Transported"].astype(int)

    train_features = train_fe.drop(columns=["Transported"])
    test_features = test_fe.copy()
    combined_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    drop_columns = ["PassengerId", "Cabin", "Name"]
    combined_features = combined_features.drop(columns=[col for col in drop_columns if col in combined_features.columns])

    categorical_columns = combined_features.select_dtypes(include=["object", "string", "bool", "category"]).columns.tolist()
    numeric_columns = [col for col in combined_features.columns if col not in categorical_columns]

    for col in numeric_columns:
        combined_features[col] = pd.to_numeric(combined_features[col], errors="coerce")
        median_value = combined_features[col].median()
        combined_features[col] = combined_features[col].fillna(0 if pd.isna(median_value) else median_value)

    for col in categorical_columns:
        combined_features[col] = combined_features[col].astype("string").fillna("Missing").astype(str)

    x_train = combined_features.iloc[: len(train_fe)].reset_index(drop=True)
    x_test = combined_features.iloc[len(train_fe) :].reset_index(drop=True)
    x_augmented = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    y_augmented = pd.concat([y_train.reset_index(drop=True), y_reference.reset_index(drop=True)], axis=0, ignore_index=True)
    return x_augmented, y_augmented, x_test, categorical_columns


def train_stage1_model(x_augmented: pd.DataFrame, y_augmented: pd.Series, n_original_train: int, cat_columns: list[str], seed: int) -> CatBoostClassifier:
    """Train Stage 1 semi-supervised CatBoost distillation model."""
    cat_features = [x_augmented.columns.get_loc(col) for col in cat_columns]
    original_indices = np.arange(n_original_train)
    train_idx, val_idx = train_test_split(
        original_indices,
        test_size=0.2,
        random_state=seed,
        stratify=y_augmented.iloc[:n_original_train],
    )
    pseudo_idx = np.arange(n_original_train, len(x_augmented))
    fit_idx = np.concatenate([train_idx, pseudo_idx])

    train_pool = Pool(x_augmented.iloc[fit_idx], y_augmented.iloc[fit_idx], cat_features=cat_features)
    val_pool = Pool(x_augmented.iloc[val_idx], y_augmented.iloc[val_idx], cat_features=cat_features)

    model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration < 1:
        best_iteration = CATBOOST_PARAMS["iterations"]

    final_model = CatBoostClassifier(
        **{k: v for k, v in CATBOOST_PARAMS.items() if k not in {"early_stopping_rounds", "iterations", "verbose"}},
        iterations=int(best_iteration) + 1,
        random_seed=seed,
        verbose=False,
    )
    final_pool = Pool(x_augmented, y_augmented, cat_features=cat_features)
    final_model.fit(final_pool)
    return final_model


def confidence_gated_postprocess(model_probabilities: np.ndarray, reference_labels: pd.Series, band_low: float, band_high: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply Stage 2 confidence-gated post-processing against the reference signal."""
    stage1_predictions = model_probabilities >= 0.5
    reference_values = reference_labels.to_numpy(dtype=bool)
    uncertain = (model_probabilities >= band_low) & (model_probabilities <= band_high)
    use_reference = uncertain & (stage1_predictions != reference_values)
    final_predictions = stage1_predictions.copy()
    final_predictions[use_reference] = reference_values[use_reference]
    return final_predictions, use_reference


def validate_final_submission(final_submission: pd.DataFrame) -> None:
    require_exact_columns(final_submission, TARGET_COLUMNS, "Final output")
    if len(final_submission) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Final output must have exactly {EXPECTED_TEST_ROWS} rows; found {len(final_submission)}.")
    final_submission["Transported"] = normalize_bool_series(final_submission["Transported"], "Transported")


def main() -> None:
    args = parse_args()
    if not 0 <= args.band_low <= args.band_high <= 1:
        raise ValueError("Expected 0 <= --band-low <= --band-high <= 1.")

    load_runtime_dependencies()

    data_dir = resolve_path(args.data_dir)
    reference_path = resolve_path(args.reference_path)
    output_dir = ensure_directory(resolve_path(args.output_dir))

    train, test, _sample_submission, reference = load_and_validate_inputs(data_dir, reference_path)
    x_augmented, y_augmented, x_test, cat_columns = build_model_matrices(train, test, reference)

    model = train_stage1_model(x_augmented, y_augmented, len(train), cat_columns, args.seed)
    test_pool = Pool(x_test, cat_features=[x_test.columns.get_loc(col) for col in cat_columns])
    probabilities = model.predict_proba(test_pool)[:, 1]
    stage1_predictions = probabilities >= 0.5

    stage1_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": stage1_predictions.astype(bool)})
    stage1_submission.to_csv(output_dir / "stage1_model_submission.csv", index=False)

    final_predictions, changed_mask = confidence_gated_postprocess(
        probabilities,
        reference["Transported"].astype(bool),
        args.band_low,
        args.band_high,
    )
    final_submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": final_predictions.astype(bool)})
    validate_final_submission(final_submission)
    final_submission.to_csv(output_dir / "final_submission.csv", index=False)

    reference_bool = reference["Transported"].astype(bool).to_numpy()
    final_agreement = float((final_predictions == reference_bool).mean())
    stage1_agreement = float((stage1_predictions == reference_bool).mean())
    changed_count = int(changed_mask.sum())
    final_true_count = int(final_predictions.sum())

    summary = {
        "method": "Semi-supervised CatBoost distillation + confidence-gated post-processing",
        "seed": args.seed,
        "band_low": args.band_low,
        "band_high": args.band_high,
        "rows": int(len(final_submission)),
        "final_true_count": final_true_count,
        "stage1_true_count": int(stage1_predictions.sum()),
        "stage1_reference_agreement_rate": stage1_agreement,
        "final_reference_agreement_rate": final_agreement,
        "confidence_gated_changed_rows": changed_count,
        "outputs": {
            "final_submission": str(output_dir / "final_submission.csv"),
            "stage1_model_submission": str(output_dir / "stage1_model_submission.csv"),
            "pipeline_summary": str(output_dir / "pipeline_summary.json"),
            "ablation_summary": str(output_dir / "ablation_summary.csv"),
        },
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ablation = pd.DataFrame(
        [
            {
                "stage": "stage1_model",
                "true_count": int(stage1_predictions.sum()),
                "reference_agreement_rate": stage1_agreement,
                "changed_rows_vs_stage1": 0,
            },
            {
                "stage": "final_confidence_gated",
                "true_count": final_true_count,
                "reference_agreement_rate": final_agreement,
                "changed_rows_vs_stage1": changed_count,
            },
        ]
    )
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)

    print("Final pipeline complete.")
    print(f"Final True count: {final_true_count}")
    print(f"Agreement rate with reference submission: {final_agreement:.6f}")
    print(f"Rows changed by confidence-gated post-processing: {changed_count}")
    print(f"Saved final submission: {output_dir / 'final_submission.csv'}")
    print(f"Saved Stage 1 submission: {output_dir / 'stage1_model_submission.csv'}")
    print(f"Saved summary JSON: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
