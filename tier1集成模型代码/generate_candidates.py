"""
Generate candidate submissions from already-trained OOF / test probas.

Run AFTER tier1_ensemble.py has populated oof_probas.npz and test_probas.npz.

Outputs (all in DATA_DIR):
    submission_cb_only_050.csv
    submission_cb_only_045.csv
    submission_cb_only_042.csv
    submission_4cb_xgb_hgb_050.csv
    submission_4cb_xgb_hgb_optimal.csv
    submission_ensemble_perside.csv      (already exists; rewritten)
    candidates_summary.csv               (one-row-per-submission comparison)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


DATA_DIR = Path(os.environ.get("SPACETITANIC_DATA", "."))


def best_thr(p, y, grid=None):
    if grid is None:
        grid = np.arange(0.30, 0.60 + 1e-9, 0.005)
    accs = [accuracy_score(y, (p >= t).astype(int)) for t in grid]
    i = int(np.argmax(accs))
    return float(grid[i]), float(accs[i])


def main():
    oof = np.load(DATA_DIR / "oof_probas.npz")
    test = np.load(DATA_DIR / "test_probas.npz")
    y = oof["y"]
    test_ids = pd.read_csv(DATA_DIR / "test_passenger_ids.csv")
    side_train = pd.read_csv(
        DATA_DIR / "X_train_catboost_features.csv"
    )["Side"].astype(str).values
    side_test = pd.read_csv(
        DATA_DIR / "X_test_catboost_features.csv"
    )["Side"].astype(str).values

    # Weighted ensembles to compare.
    candidate_probas = {
        "cb_only":       (oof["cb"], test["cb"]),
        "xgb_only":      (oof["xgb"], test["xgb"]),
        "hgb_only":      (oof["hgb"], test["hgb"]),
        "ens_111":       (oof["ensemble"], test["ensemble"]),
        "ens_211":       (
            (2 * oof["cb"] + oof["xgb"] + oof["hgb"]) / 4,
            (2 * test["cb"] + test["xgb"] + test["hgb"]) / 4,
        ),
        "ens_311":       (
            (3 * oof["cb"] + oof["xgb"] + oof["hgb"]) / 5,
            (3 * test["cb"] + test["xgb"] + test["hgb"]) / 5,
        ),
        "ens_411":       (
            (4 * oof["cb"] + oof["xgb"] + oof["hgb"]) / 6,
            (4 * test["cb"] + test["xgb"] + test["hgb"]) / 6,
        ),
        "cb_xgb_only":   (
            (oof["cb"] + oof["xgb"]) / 2,
            (test["cb"] + test["xgb"]) / 2,
        ),
    }

    summary_rows = []
    for name, (oof_p, test_p) in candidate_probas.items():
        a050 = accuracy_score(y, (oof_p >= 0.5).astype(int))
        thr, abest = best_thr(oof_p, y)
        n_true_050 = int((test_p >= 0.5).sum())
        n_true_opt = int((test_p >= thr).sum())
        summary_rows.append({
            "blend": name,
            "oof_acc_050": round(a050, 5),
            "oof_best_thr": round(thr, 3),
            "oof_acc_best_thr": round(abest, 5),
            "test_True_at_050": n_true_050,
            "test_True_at_opt": n_true_opt,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "oof_acc_best_thr", ascending=False
    ).reset_index(drop=True)
    summary.to_csv(DATA_DIR / "candidates_summary.csv", index=False)
    print("\n========== Candidate summary ==========")
    print(summary.to_string(index=False))

    # Pick the concrete submissions to write to disk.
    def write_sub(name, test_p, thr):
        pred = (test_p >= thr).astype(bool)
        out = pd.DataFrame({
            "PassengerId": test_ids["PassengerId"],
            "Transported": pred,
        })
        path = DATA_DIR / f"submission_{name}.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {path.name}: thr={thr:.3f}, True={int(pred.sum())}")

    print("\n========== Writing candidate submissions ==========")
    write_sub("cb_only_050", test["cb"], 0.50)
    write_sub("cb_only_045", test["cb"], 0.45)
    write_sub("cb_only_042", test["cb"], 0.42)

    # Best CatBoost-heavy ensemble at its optimal threshold.
    best_row = summary.iloc[0]
    name = best_row["blend"]
    oof_p, test_p = candidate_probas[name]
    thr = float(best_row["oof_best_thr"])
    write_sub(f"best_blend_{name}_thr{thr:.3f}".replace(".", "_"),
              test_p, thr)

    # Per-Side ensemble (using best blend).
    print("\n========== Per-Side thresholds on best blend ==========")
    perside = {}
    for s in ["P", "S"]:
        mask = side_train == s
        if mask.sum() < 50:
            continue
        thr_s, acc_s = best_thr(oof_p[mask], y[mask])
        perside[s] = thr_s
        print(f"  Side={s}: thr={thr_s:.3f}, OOF acc={acc_s:.5f}, "
              f"n={int(mask.sum())}")
    pred = np.zeros(len(test_p), dtype=int)
    for i, s in enumerate(side_test):
        t = perside.get(s, 0.5)
        pred[i] = int(test_p[i] >= t)
    out = pd.DataFrame({
        "PassengerId": test_ids["PassengerId"],
        "Transported": pred.astype(bool),
    })
    out.to_csv(DATA_DIR / "submission_perside_bestblend.csv", index=False)
    print(f"Wrote submission_perside_bestblend.csv: "
          f"True={int(pred.sum())}, thresholds={perside}")

    # Save final analysis json.
    out_json = {
        "best_blend_on_oof": str(name),
        "best_blend_oof_acc": float(best_row["oof_acc_best_thr"]),
        "best_blend_thr": thr,
        "perside_thresholds": {k: float(v) for k, v in perside.items()},
        "all_blends": summary_rows,
    }
    with open(DATA_DIR / "candidate_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
