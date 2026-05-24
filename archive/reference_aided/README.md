# Reference-aided archive

This folder contains an **archived reference-aided audit artifact** related to a post-hoc Kaggle submission analysis.

## What is included

- `generate_reference_aided_submission.py`: script to reproduce the archived reference-aided post-hoc submission and an audit table.
- `reference_aided_inputs/`: fixed input submissions used by the archived script.
- `best_kaggle_submission_082183.csv`: archived auxiliary submission artifact.

## Scope and status

This archive is **not** the standard reproducible training pipeline.

Standard training/demo pipeline uses:
- `src/train_*.py`
- `src/compare_models.py`
- `src/analyze_fusion_vs_catboost.py`

## Run (if needed)

```bash
python archive/reference_aided/generate_reference_aided_submission.py
```
