# Reference-aided 0.82183 audit reproduction (archive only)

This folder is **not** part of the standard training pipeline.

It provides a clearly separated, reference-aided audit/post-processing reproduction line for the 0.82183 Kaggle submission. It requires an external reference submission and should **not** be interpreted as a standalone ML model.

The clean main pipeline remains based on:
- `src/train_*.py`
- `src/compare_models.py`
- `src/analyze_fusion_vs_catboost.py`

## Required local input files

- `data/test.csv`
- `submissions/reference_aided_inputs/submission_catboost_threshold_050.csv`
- `submissions/reference_aided_inputs/reference_submission_082137.csv`

## Command

```bash
python archive/reference_aided/generate_reference_aided_082183.py
```

## Output

- `submissions/reference_aided_outputs/best_kaggle_submission_082183.csv`
