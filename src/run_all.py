"""Run one or more Spaceship Titanic model training scripts and build a unified summary."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from utils import OUTPUT_DIR, REPO_ROOT, ensure_directory

@dataclass(frozen=True)
class ModelCommand:
    name: str
    script: Path
    heavy: bool = False
    optional: bool = False

MODEL_COMMANDS: dict[str, ModelCommand] = {
    "logistic_regression": ModelCommand("logistic_regression", REPO_ROOT / "src" / "train_logistic_regression.py"),
    "random_forest": ModelCommand("random_forest", REPO_ROOT / "src" / "train_random_forest.py"),
    "svm": ModelCommand("svm", REPO_ROOT / "src" / "train_svm.py"),
    "xgboost": ModelCommand("xgboost", REPO_ROOT / "src" / "train_xgboost.py", heavy=True),
    "lightgbm": ModelCommand("lightgbm", REPO_ROOT / "src" / "train_lightgbm.py", heavy=True, optional=True),
    "catboost": ModelCommand("catboost", REPO_ROOT / "src" / "train_catboost.py", heavy=True, optional=False),
}
DEFAULT_MODELS = ["logistic_regression", "random_forest", "svm", "xgboost", "lightgbm", "catboost"]

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Run selected Spaceship Titanic model scripts.")
    p.add_argument("--models", nargs="+", choices=sorted(MODEL_COMMANDS), default=DEFAULT_MODELS)
    p.add_argument("--skip-heavy", action="store_true")
    return p.parse_args()

def command_for_model(model: ModelCommand, skip_heavy: bool) -> list[str]:
    cmd=[sys.executable, str(model.script)]
    if skip_heavy and model.name=="svm": cmd.append("--skip-tuning")
    if skip_heavy and model.name=="logistic_regression": cmd.extend(["--optuna-trials","5"])
    if model.name=="xgboost": cmd.append("--make-submission")
    if skip_heavy and model.name=="xgboost": cmd.append("--fast")
    return cmd

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def summarize_model_outputs(model_name: str, output_dir: Path) -> dict[str, object]:
    row={"model_name":model_name,"validation_accuracy":pd.NA,"cv_accuracy":pd.NA,"kaggle_score":pd.NA,"training_time_seconds":pd.NA,"best_params":pd.NA,"submission_file":pd.NA,"notes":""}
    if model_name=="svm" and (output_dir/"svm_validation_metrics.json").exists():
        m=_read_json(output_dir/"svm_validation_metrics.json"); row.update({"validation_accuracy":m.get("best_validation_accuracy"),"cv_accuracy":m.get("best_cv_accuracy"),"training_time_seconds":m.get("best_training_time_seconds"),"best_params":json.dumps(m.get("best_params",{})),"submission_file":"outputs/svm_submission.csv","notes":"SVM metrics from svm_validation_metrics.json"})
    elif model_name=="random_forest" and (output_dir/"random_forest_validation_summary.csv").exists():
        s=pd.read_csv(output_dir/"random_forest_validation_summary.csv").iloc[0]; row.update({"validation_accuracy":s.get("validation_accuracy"),"cv_accuracy":s.get("mean_cv_accuracy"),"training_time_seconds":s.get("training_time_seconds"),"best_params":"fixed teammate params","submission_file":"outputs/random_forest_submission.csv","notes":"Random Forest summary"})
    elif model_name=="logistic_regression" and (output_dir/"logistic_regression_validation_summary.csv").exists():
        s=pd.read_csv(output_dir/"logistic_regression_validation_summary.csv").iloc[0]; row.update({"validation_accuracy":s.get("validation_accuracy"),"cv_accuracy":s.get("cv_roc_auc",pd.NA),"training_time_seconds":s.get("training_time_seconds"),"best_params":s.get("best_params",pd.NA),"submission_file":"outputs/logistic_regression_submission.csv","notes":"Logistic Regression summary"})
    elif model_name=="xgboost" and (output_dir/"xgboost_validation_summary.csv").exists():
        s=pd.read_csv(output_dir/"xgboost_validation_summary.csv").iloc[0]; row.update({"validation_accuracy":s.get("validation_accuracy"),"cv_accuracy":s.get("cv_accuracy",pd.NA),"training_time_seconds":s.get("training_time_seconds"),"best_params":s.get("best_params",pd.NA),"submission_file":"outputs/xgboost_submission.csv","notes":"XGBoost summary"})
    elif model_name=="lightgbm" and (output_dir/"lgbm_fold_results.csv").exists():
        s=pd.read_csv(output_dir/"lgbm_fold_results.csv"); row.update({"validation_accuracy":s["accuracy_threshold_0.5"].mean() if "accuracy_threshold_0.5" in s.columns else pd.NA,"training_time_seconds":s["training_time_seconds"].sum() if "training_time_seconds" in s.columns else pd.NA,"best_params":"fixed teammate params","submission_file":"outputs/submission_lgbm_v1.csv","notes":"LightGBM fold summary"})
    elif model_name=="catboost" and (output_dir/"catboost_fold_results.csv").exists():
        s=pd.read_csv(output_dir/"catboost_fold_results.csv"); mean_acc=s["validation_accuracy"].mean(); total_time=s["training_time_seconds"].sum()
        row.update({"validation_accuracy":mean_acc,"cv_accuracy":mean_acc,"training_time_seconds":total_time,"best_params":"iterations=7000, learning_rate=0.02, depth=8, l2_leaf_reg=5.0, random_strength=1.2","submission_file":"outputs/submission_catboost_v1.csv","notes":"CatBoost progress-stage baseline summary"})
    return row

def main() -> int:
    args=parse_args(); requested=[m for m in dict.fromkeys(args.models) if (not args.skip_heavy or not MODEL_COMMANDS[m].heavy)]; output_dir=ensure_directory(OUTPUT_DIR)
    run_rows=[]
    for model_name in requested:
        model=MODEL_COMMANDS[model_name]
        if not model.script.exists():
            note="Optional model script not found; skipped." if model.optional else "Model script not found."; row=summarize_model_outputs(model_name, output_dir); row["notes"]=f"{row['notes']} {note}".strip(); run_rows.append(row); continue
        command=command_for_model(model,args.skip_heavy); start=time.time(); result=subprocess.run(command,cwd=REPO_ROOT,check=False); elapsed=time.time()-start
        row=summarize_model_outputs(model_name, output_dir)
        if pd.isna(row["training_time_seconds"]): row["training_time_seconds"]=round(elapsed,3)
        if result.returncode!=0: row["notes"]=f"{row['notes']} run failed with code {result.returncode}".strip()
        run_rows.append(row)
    df=pd.DataFrame(run_rows); df.to_csv(output_dir/"results_summary.csv",index=False); print(df); return 0

if __name__=="__main__":
    raise SystemExit(main())
