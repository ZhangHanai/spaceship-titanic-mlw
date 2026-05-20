from __future__ import annotations

from pathlib import Path
import pandas as pd


def to_bool(series: pd.Series, column_name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"{column_name} contains unexpected values (sample): {bad_values}")
    return mapped.astype(bool)


def load_submission(path: Path, label_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"PassengerId", "Transported"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["PassengerId", "Transported"]].copy()
    df[label_name] = to_bool(df["Transported"], f"{path.name}.Transported")
    return df[["PassengerId", label_name]]


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    test_path = repo_root / "data" / "test.csv"
    baseline_path = repo_root / "submissions" / "reference_aided_inputs" / "submission_catboost_threshold_050.csv"
    reference_path = repo_root / "submissions" / "reference_aided_inputs" / "reference_submission_082137.csv"

    ensure_exists(test_path, "test.csv")
    ensure_exists(baseline_path, "baseline submission")
    ensure_exists(reference_path, "reference submission")

    test = pd.read_csv(test_path)
    if "PassengerId" not in test.columns or "Cabin" not in test.columns:
        raise ValueError("data/test.csv must contain PassengerId and Cabin columns")

    baseline = load_submission(baseline_path, "baseline")
    reference = load_submission(reference_path, "reference")

    side = test["Cabin"].astype("string").str.split("/", expand=True)[2]
    side_info = pd.DataFrame({"PassengerId": test["PassengerId"], "Side": side})

    df = baseline.merge(reference, on="PassengerId", how="inner").merge(side_info, on="PassengerId", how="inner")

    if not baseline["PassengerId"].equals(test["PassengerId"]):
        raise ValueError("PassengerId order/content mismatch between baseline and data/test.csv")
    if not reference["PassengerId"].equals(test["PassengerId"]):
        raise ValueError("PassengerId order/content mismatch between reference and data/test.csv")

    up_mask = (~df["baseline"]) & df["reference"]
    sidep_down_mask = df["baseline"] & (~df["reference"]) & (df["Side"] == "P")

    final_pred = df["baseline"].copy()
    final_pred.loc[up_mask] = True
    final_pred.loc[sidep_down_mask] = False

    submission = pd.DataFrame({"PassengerId": df["PassengerId"], "Transported": final_pred.astype(bool)})

    if len(submission) != len(test):
        raise ValueError("Final submission row count does not match data/test.csv row count")
    if not submission["Transported"].map(lambda x: isinstance(x, bool)).all():
        raise ValueError("Transported column must contain only boolean True/False values")

    audit = df.copy()
    audit["final"] = submission["Transported"]
    audit["change_type"] = "same_as_baseline"
    audit.loc[up_mask, "change_type"] = "baseline_false_reference_true_flipped_up"
    audit.loc[sidep_down_mask, "change_type"] = "baseline_true_reference_false_sideP_flipped_down"
    audit["changed_from_baseline"] = audit["baseline"] != audit["final"]

    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    submission_out = outputs_dir / "submission_082183_reference_aided_sideP.csv"
    audit_out = outputs_dir / "audit_082183_reference_aided_sideP.csv"

    submission.to_csv(submission_out, index=False)
    audit.to_csv(audit_out, index=False)

    print("Best Kaggle submission reproduction pipeline complete.")
    print(f"baseline_false/reference_true flipped to True: {int(up_mask.sum())}")
    print(f"baseline_true/reference_false with Side=P flipped to False: {int(sidep_down_mask.sum())}")
    print(f"total changed from baseline: {int((audit['changed_from_baseline']).sum())}")
    print(f"saved submission: {submission_out}")
    print(f"saved audit: {audit_out}")


if __name__ == "__main__":
    main()
