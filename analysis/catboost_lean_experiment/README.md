# Lean CatBoost experiment (optional analysis)

This folder contains optional analysis scripts that reproduce our stronger **lean CatBoost** experiment used as the best clean CatBoost comparison point.

- `preprocess_lean.py`
- `train_catboost_lean.py`

## Important scope

- This is **optional analysis code** and **not a replacement** for teammate progress-stage baseline `src/train_catboost.py`.
- The progress-stage CatBoost baseline remains in `src/train_catboost.py`.
- In our submitted experiments, the lean CatBoost per-Side submission achieved **Kaggle public score 0.80851**.
- These scripts support final analysis comparing best clean CatBoost vs fusion/ensemble.

## Main demo path (unchanged)

```bash
python src/run_all.py --skip-heavy
python src/compare_models.py
python src/analyze_fusion_vs_catboost.py
```

## Optional rerun commands

```bash
python analysis/catboost_lean_experiment/preprocess_lean.py
python analysis/catboost_lean_experiment/train_catboost_lean.py
```

Full training can be slow. Requires Kaggle data files:

- `data/train.csv`
- `data/test.csv`
