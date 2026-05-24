# Tier 1 Fusion Experiments (Optional Analysis)

These scripts reproduce the Tier 1 fusion experiment used for experimental analysis.

- Fusion combines **CatBoost**, **XGBoost**, and **HistGradientBoosting** using soft voting, weighted averaging, and per-Side threshold calibration.
- This is **optional analysis code**, not the final model pipeline.
- The main demo path remains:
  - `python src/run_all.py --skip-heavy`
  - `python src/compare_models.py`
  - `python src/analyze_fusion_vs_catboost.py`

## Re-run fusion experiments

```bash
python analysis/fusion_experiments/tier1_ensemble.py
python analysis/fusion_experiments/generate_candidates.py
```

Notes:
- Fusion training may be slow.
- It requires `data/spaceship_catboost_preprocessed_package/`.
- The final presentation uses `src/analyze_fusion_vs_catboost.py` to generate the clean comparison figure.
