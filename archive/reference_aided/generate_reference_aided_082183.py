from pathlib import Path

import pandas as pd


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _validate_submission_schema(df: pd.DataFrame, name: str) -> None:
    expected_columns = ["PassengerId", "Transported"]
    actual_columns = list(df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"{name} must have columns exactly {expected_columns}, got {actual_columns}"
        )


def _to_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False}
    if not lowered.isin(mapping.keys()).all():
        bad_values = sorted(series[~lowered.isin(mapping.keys())].astype(str).unique())
        raise ValueError(
            f"{name} contains non-boolean values that cannot be parsed: {bad_values}"
        )
    return lowered.map(mapping).astype(bool)


def main() -> None:
    test_path = Path("data/test.csv")
    baseline_path = Path(
        "submissions/reference_aided_inputs/submission_catboost_threshold_050.csv"
    )
    reference_path = Path(
        "submissions/reference_aided_inputs/reference_submission_082137.csv"
    )
    output_path = Path(
        "submissions/reference_aided_outputs/best_kaggle_submission_082183.csv"
    )

    for input_path in (test_path, baseline_path, reference_path):
        _ensure_exists(input_path)

    test_df = pd.read_csv(test_path)
    baseline_df = pd.read_csv(baseline_path)
    reference_df = pd.read_csv(reference_path)

    _validate_submission_schema(baseline_df, "Baseline submission")
    _validate_submission_schema(reference_df, "Reference submission")

    if "PassengerId" not in test_df.columns:
        raise ValueError("data/test.csv must contain PassengerId column")
    if "Cabin" not in test_df.columns:
        raise ValueError("data/test.csv must contain Cabin column")

    test_ids = test_df["PassengerId"]
    baseline_ids = baseline_df["PassengerId"]
    reference_ids = reference_df["PassengerId"]

    if not test_ids.equals(baseline_ids):
        raise ValueError("PassengerId order mismatch between data/test.csv and baseline")
    if not test_ids.equals(reference_ids):
        raise ValueError("PassengerId order mismatch between data/test.csv and reference")

    baseline_bool = _to_bool(baseline_df["Transported"], "Baseline submission Transported")
    reference_bool = _to_bool(reference_df["Transported"], "Reference submission Transported")

    cabin_parts = test_df["Cabin"].fillna("//").astype(str).str.split("/", n=2, expand=True)
    while cabin_parts.shape[1] < 3:
        cabin_parts[cabin_parts.shape[1]] = ""
    cabin_parts = cabin_parts.iloc[:, :3]
    cabin_parts.columns = ["Deck", "Cabin_Num", "Side"]

    false_to_true_mask = (~baseline_bool) & reference_bool
    true_to_false_side_p_mask = baseline_bool & (~reference_bool) & (cabin_parts["Side"] == "P")

    output_df = baseline_df.copy()
    output_bool = baseline_bool.copy()
    output_bool.loc[false_to_true_mask] = True
    output_bool.loc[true_to_false_side_p_mask] = False
    output_df["Transported"] = output_bool

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"baseline True count: {int(baseline_bool.sum())}")
    print(f"reference True count: {int(reference_bool.sum())}")
    print(f"number of False -> True corrections: {int(false_to_true_mask.sum())}")
    print(
        "number of True -> False Side=P corrections: "
        f"{int(true_to_false_side_p_mask.sum())}"
    )
    print(f"final True count: {int(output_bool.sum())}")
    print(f"output path: {output_path}")
    print(
        "Important: this is a reference-aided post-processing audit, "
        "not a standalone ML training pipeline."
    )


if __name__ == "__main__":
    main()
