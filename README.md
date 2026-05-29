# AI3023 Machine Learning Workshop — Spaceship Titanic

## Project overview

This repository contains our AI3023 Machine Learning Workshop project for the Kaggle **Spaceship Titanic** binary classification competition. The task is to predict each passenger's `Transported` value from the Kaggle training and test CSV files.

The repository keeps two kinds of work clearly separated:

1. **Standard model experiments** for reproducible baseline and model-comparison work.
2. **Final competition-side pipeline** for the submitted/reference-aided competition solution.

The final competition-side method is documented transparently as **Semi-supervised CatBoost distillation + confidence-gated post-processing**. It uses a public reference submission as a teacher/reference signal and should not be described as a purely standalone CatBoost baseline.

## Standard model experiments: LR, RF, SVM, XGBoost, LightGBM, CatBoost

The standard experiment scripts are preserved under `src/` and remain runnable independently:

```text
src/train_logistic_regression.py
src/train_random_forest.py
src/train_svm.py
src/train_xgboost.py
src/train_lightgbm.py
src/train_catboost.py
src/run_all.py
src/compare_models.py
src/analyze_fusion_vs_catboost.py
```

These scripts cover Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, and CatBoost experiments, plus comparison utilities. They are retained as standard model experiments and are not replaced by the final competition-side pipeline.

Useful standard commands:

```bash
# Fast inspection run; skips heavier models.
python src/run_all.py --skip-heavy
python src/compare_models.py
python src/analyze_fusion_vs_catboost.py

# Full model-comparison run.
python src/run_all.py
python src/compare_models.py
python src/analyze_fusion_vs_catboost.py

# Individual model scripts.
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_svm.py
python src/train_xgboost.py --make-submission
python src/train_lightgbm.py
python src/train_catboost.py
```

LightGBM additionally expects either `data/spaceship_catboost_preprocessed_package.zip` or an extracted `data/spaceship_catboost_preprocessed_package/` directory, as described in `data/README.md`.

## Final competition-side pipeline: semi-supervised distillation + confidence-gated post-processing

The final competition-side script is:

```text
src/run_final_reference_aided_pipeline.py
```

Method name: **Semi-supervised CatBoost distillation + confidence-gated post-processing**.

The final submitted version has known Kaggle public score: **0.82160**.

This final pipeline uses a public reference submission as a teacher/reference signal. It is therefore documented separately from the standalone model experiments.

### Stage 1 — semi-supervised pseudo-label distillation

The script trains CatBoost using:

- original training rows with original Kaggle labels; and
- test rows with pseudo-labels taken from the public reference submission.

This stage is semi-supervised pseudo-label distillation: CatBoost learns from both the original labeled training data and the reference-labeled test rows.

### Stage 2 — confidence-gated post-processing

The script then compares Stage 1 model probabilities with the reference signal. If the model prediction is uncertain and disagrees with the reference signal, the final output uses the reference signal. If the model is highly confident, the final output keeps the model prediction.

The uncertainty band is controlled by:

- `--band-low`
- `--band-high`

With the default competition command below, probabilities between `0.10` and `0.90` are treated as the uncertainty band for disagreement-based reference fallback.

The script validates that:

- `train.csv`, `test.csv`, `sample_submission.csv`, and the reference submission exist;
- the reference submission has exactly `PassengerId` and `Transported` columns;
- PassengerIds match between `test.csv`, `sample_submission.csv`, and the reference submission;
- the final output has exactly 4,277 rows;
- the final output has exactly `PassengerId` and `Transported` columns;
- `Transported` values are boolean-compatible; and
- final true count, reference agreement rate, and changed row count are printed.

## How to run

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Required files

Place the Kaggle competition files in `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

Place the public reference submission used as teacher/reference signal here:

```text
submissions/reference/public_reference_submission.csv
```

The reference submission must contain exactly:

```text
PassengerId,Transported
```

Generated outputs are written under `outputs/` and are ignored by Git.

### Run the final reference-aided pipeline

```bash
python src/run_final_reference_aided_pipeline.py \
  --data-dir data \
  --reference-path submissions/reference/public_reference_submission.csv \
  --output-dir outputs/final_reference_aided \
  --band-low 0.10 \
  --band-high 0.90
```

Expected generated files:

```text
outputs/final_reference_aided/final_submission.csv
outputs/final_reference_aided/stage1_model_submission.csv
outputs/final_reference_aided/pipeline_summary.json
outputs/final_reference_aided/ablation_summary.csv
```

### Repository structure

```text
spaceship-titanic-mlw/
├── analysis/
├── archive/
├── data/
├── docs/
│   └── experiment_summary.md
├── figures/
├── outputs/
├── src/
│   ├── run_final_reference_aided_pipeline.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_svm.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_catboost.py
│   ├── run_all.py
│   ├── compare_models.py
│   └── analyze_fusion_vs_catboost.py
├── submissions/
│   └── reference/
│       └── README.md
├── README.md
└── requirements.txt
```
