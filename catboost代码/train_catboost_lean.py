"""
CatBoost multi-seed CV on the lean 24-feature set.

For each seed:
  - 5-fold stratified split
  - native categorical handling (no one-hot)
  - early stopping at 100 rounds
  - keep OOF probas + test fold probas

Final test proba = mean across (seeds x folds).
Outputs:
  - oof_probas_lean.npy
  - test_probas_lean.npy
  - submission_lean_catboost.csv  (threshold = 0.5)
  - lean_run_summary.json
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

DATA_DIR = Path(os.environ.get('SPACETITANIC_DATA', '.'))

SEEDS = [42, 123, 2024]
N_SPLITS = 5

# Params: similar to user's prior best (depth=8, lr=0.02, l2=4.0).
# Iterations capped high; early stopping decides actual count.
PARAMS = dict(
    iterations=4000,        # capped; early stop usually fires <800
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=4.0,
    random_strength=1.0,
    bootstrap_type='Bayesian',
    bagging_temperature=1.0,
    border_count=128,
    loss_function='Logloss',
    eval_metric='Accuracy',
    od_type='Iter',
    od_wait=100,
    use_best_model=True,
    verbose=False,
    allow_writing_files=False,
)


def run_one_seed(X, y, X_test, cat_idx, seed, params):
    """Return (oof_proba, test_proba_mean_over_folds, fold_accs)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=np.float64)
    test_acc = np.zeros(len(X_test), dtype=np.float64)
    fold_accs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = Pool(X_va, y_va, cat_features=cat_idx)
        test_pool = Pool(X_test, cat_features=cat_idx)

        p = dict(params)
        p['random_seed'] = seed * 1000 + fold_idx
        model = CatBoostClassifier(**p)
        model.fit(train_pool, eval_set=val_pool)

        oof[va_idx] = model.predict_proba(val_pool)[:, 1]
        test_acc += model.predict_proba(test_pool)[:, 1] / N_SPLITS

        fa = float((oof[va_idx] > 0.5).astype(int) == y_va).mean() if False else (
            ((oof[va_idx] > 0.5).astype(int) == y_va).mean()
        )
        fold_accs.append(fa)
        print(f'  seed {seed} fold {fold_idx}: acc={fa:.5f} '
              f'best_iter={model.get_best_iteration()}')

    return oof, test_acc, fold_accs


def main():
    X_train = pd.read_csv(DATA_DIR / 'X_train_lean.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test_lean.csv')
    y_df = pd.read_csv(DATA_DIR / 'y_train_lean.csv')
    test_ids = pd.read_csv(DATA_DIR / 'test_ids_lean.csv')
    meta = json.loads((DATA_DIR / 'lean_meta.json').read_text())

    y = y_df['Transported'].astype(int).values
    cat_cols = meta['categorical']
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    print(f'cat columns -> indices: {dict(zip(cat_cols, cat_idx))}')

    oof_all = np.zeros((len(SEEDS), len(X_train)))
    test_all = np.zeros((len(SEEDS), len(X_test)))
    seed_accs = {}

    t0 = time.time()
    for i, s in enumerate(SEEDS):
        print(f'\n=== seed {s} ===')
        oof, test_proba, fold_accs = run_one_seed(
            X_train, y, X_test, cat_idx, s, PARAMS
        )
        oof_all[i] = oof
        test_all[i] = test_proba
        seed_acc = float(((oof > 0.5).astype(int) == y).mean())
        seed_accs[s] = {
            'cv_acc_at_0.5': seed_acc,
            'fold_accs': fold_accs,
        }
        print(f'  seed {s} CV acc @0.5 = {seed_acc:.5f}')

    oof_mean = oof_all.mean(axis=0)
    test_mean = test_all.mean(axis=0)
    final_oof_acc = float(((oof_mean > 0.5).astype(int) == y).mean())
    elapsed = time.time() - t0

    # Find best threshold on OOF
    best_t = 0.5
    best_a = final_oof_acc
    for t in np.arange(0.30, 0.70, 0.005):
        a = float(((oof_mean > t).astype(int) == y).mean())
        if a > best_a:
            best_a = a
            best_t = float(t)

    print(f'\n==== FINAL ====')
    print(f'  OOF acc @0.5    = {final_oof_acc:.5f}')
    print(f'  OOF acc @best={best_t:.3f} -> {best_a:.5f}')
    print(f'  elapsed = {elapsed:.1f}s')

    # Save
    np.save(DATA_DIR / 'oof_probas_lean.npy', oof_mean)
    np.save(DATA_DIR / 'test_probas_lean.npy', test_mean)

    # Submission @0.5 (matches old baseline 0.80547 protocol)
    sub05 = pd.DataFrame({
        'PassengerId': test_ids['PassengerId'],
        'Transported': (test_mean > 0.5).astype(bool),
    })
    sub05.to_csv(DATA_DIR / 'submission_lean_catboost_t050.csv', index=False)

    sub_best = pd.DataFrame({
        'PassengerId': test_ids['PassengerId'],
        'Transported': (test_mean > best_t).astype(bool),
    })
    sub_best.to_csv(DATA_DIR / 'submission_lean_catboost_tbest.csv', index=False)

    summary = {
        'n_features': X_train.shape[1],
        'seeds': SEEDS,
        'oof_acc_at_0.5': final_oof_acc,
        'oof_best_threshold': best_t,
        'oof_acc_at_best': best_a,
        'per_seed': seed_accs,
        'true_count_sub_at_0.5': int(sub05['Transported'].sum()),
        'true_count_sub_at_best': int(sub_best['Transported'].sum()),
        'elapsed_seconds': elapsed,
        'params': {k: v for k, v in PARAMS.items() if not callable(v)},
    }
    with open(DATA_DIR / 'lean_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nTrue count @0.5: {summary["true_count_sub_at_0.5"]}')
    print(f'True count @best({best_t:.3f}): {summary["true_count_sub_at_best"]}')
    print('Wrote submission_lean_catboost_t050.csv and ..._tbest.csv')


if __name__ == '__main__':
    main()
