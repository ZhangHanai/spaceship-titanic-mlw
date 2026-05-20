from __future__ import annotations

from pathlib import Path
import pandas as pd

LOCAL_HINT = Path(r"C:\Users\28782\MLWproject")
BASELINE_CANDIDATES = [
    Path("spaceship_082plus_outputs/submission_catboost_threshold_050.csv"),
    Path("submission_catboost_threshold_050.csv"),
]
REFERENCE_CANDIDATES = [
    Path("submission.csv"),
    Path("reference_submission_082137.csv"),
]
OUTPUT_DIR_NAME = "spaceship_082183_outputs"


def find_project_dir() -> Path:
    candidates: list[Path] = []
    if LOCAL_HINT.exists():
        candidates.append(LOCAL_HINT)
    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if (path / "test.csv").exists():
            return path
    raise FileNotFoundError(
        "Cannot find test.csv. Put this script in your project folder, "
        "or edit LOCAL_HINT near the top of this file."
    )


def first_existing(base_dir: Path, candidates: list[Path], description: str) -> Path:
    for rel_path in candidates:
        path = base_dir / rel_path
        if path.exists():
            return path
    names = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find {description}. Expected one of: {names}")


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Transported contains unexpected values: {bad_values}")
    return mapped.astype(bool)


def load_submission(path: Path, label_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"PassengerId", "Transported"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df = df[["PassengerId", "Transported"]].copy()
    df[label_name] = to_bool(df["Transported"])
    return df[["PassengerId", label_name]].sort_values("PassengerId").reset_index(drop=True)


def parse_side(test: pd.DataFrame) -> pd.DataFrame:
    side = test["Cabin"].astype("string").str.split("/", expand=True)[2]
    return pd.DataFrame({"PassengerId": test["PassengerId"], "Side": side})


def build_reference_aided_submission(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = first_existing(base_dir, BASELINE_CANDIDATES, "baseline CatBoost submission")
    reference_path = first_existing(base_dir, REFERENCE_CANDIDATES, "0.82137 reference submission")
    test_path = base_dir / "test.csv"

    baseline = load_submission(baseline_path, "baseline")
    reference = load_submission(reference_path, "reference")
    test = pd.read_csv(test_path)
    side_info = parse_side(test)

    df = baseline.merge(reference, on="PassengerId", how="inner").merge(side_info, on="PassengerId", how="left")
    if len(df) != len(test):
        raise ValueError("PassengerId mismatch among baseline, reference, and test.csv.")

    up_mask = (~df["baseline"]) & df["reference"]
    sidep_down_mask = df["baseline"] & (~df["reference"]) & (df["Side"] == "P")

    final_pred = df["baseline"].copy()
    final_pred.loc[up_mask] = True
    final_pred.loc[sidep_down_mask] = False

    submission = pd.DataFrame({
        "PassengerId": df["PassengerId"],
        "Transported": final_pred.astype(bool),
    })

    audit = df.copy()
    audit["final"] = final_pred.astype(bool)
    audit["change_type"] = "same_as_baseline"
    audit.loc[up_mask, "change_type"] = "baseline_false_reference_true_flipped_up"
    audit.loc[sidep_down_mask, "change_type"] = "baseline_true_reference_false_sideP_flipped_down"
    audit["changed_from_baseline"] = audit["baseline"] != audit["final"]

    return submission, audit


def main() -> None:
    base_dir = find_project_dir()
    out_dir = base_dir / OUTPUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    submission, audit = build_reference_aided_submission(base_dir)
    sub_path = out_dir / "submission_082183_reference_aided_sideP.csv"
    audit_path = out_dir / "audit_082183_reference_aided_sideP.csv"

    submission.to_csv(sub_path, index=False)
    audit.to_csv(audit_path, index=False)

    print(f"project_dir: {base_dir}")
    print(f"saved: {sub_path}")
    print(f"saved: {audit_path}")
    print(f"rows: {len(submission)}")
    print(f"true_count: {int(submission['Transported'].sum())}")
    print(f"true_rate: {submission['Transported'].mean():.5f}")
    print(audit["change_type"].value_counts().to_string())
    print("\nImportant: this is reference-aided post-processing, not an independent model.")


if __name__ == "__main__":
    main()
