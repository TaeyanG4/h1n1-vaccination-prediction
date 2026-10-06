from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
man = pd.read_csv(ROOT / "artifacts/baseline_v1/split_manifest.csv")
dev = man.loc[man.partition.eq("development"), "row_id"].to_numpy(int)
folds = man.loc[dev, "dev_fold"].to_numpy(int)
y = pd.read_csv(ROOT / "data/raw/train_labels.csv").vacc_h1n1_f.to_numpy(int)[dev]
a = pd.read_csv(ROOT / "artifacts/v12_exact_v5_fullstack/selected_oof.csv").set_index("row_id").loc[dev].probability.to_numpy()
b = np.load(ROOT / "artifacts/v17_xgb_optuna/round_selection/m0p85_1011_predictions.npz")["oof"]
grid = np.linspace(.15, .85, 141)


def choose_threshold(yy, p):
    scores = np.array([f1_score(yy, p >= t, zero_division=0) for t in grid])
    ix = np.flatnonzero(np.isclose(scores, scores.max()))
    j = ix[len(ix) // 2]
    return float(grid[j]), float(scores[j])


def nested(p):
    hard = np.zeros(len(y), dtype=bool)
    thresholds = []
    for f in range(5):
        tune = folds != f
        valid = folds == f
        t, _ = choose_threshold(y[tune], p[tune])
        thresholds.append(t)
        hard[valid] = p[valid] >= t
    return float(f1_score(y, hard, zero_division=0)), thresholds


def rank01(p):
    return pd.Series(p).rank(method="average", pct=True).to_numpy()


rows = []
for w in np.linspace(0, 1, 11):
    for kind, p in [
        ("prob", w * a + (1 - w) * b),
        ("rank", w * rank01(a) + (1 - w) * rank01(b)),
    ]:
        nf1, ts = nested(p)
        t, tf1 = choose_threshold(y, p)
        rows.append({
            "v12_weight": float(w),
            "v17_weight": float(1 - w),
            "kind": kind,
            "nested_f1": nf1,
            "tuned_f1": tf1,
            "threshold": t,
            "auc": float(roc_auc_score(y, p)),
            "nested_thresholds": ts,
        })

rows.sort(key=lambda r: (r["nested_f1"], r["tuned_f1"], r["auc"]), reverse=True)
out = ROOT / "artifacts/v17_xgb_optuna/v12_v17_simple_ensemble_scan.json"
out.write_text(json.dumps({"top": rows[:20], "all": rows}, indent=2), encoding="utf-8")
for row in rows[:12]:
    print(row)
