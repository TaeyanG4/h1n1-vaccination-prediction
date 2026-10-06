"""Equal-weight ensemble-first diagnostic after v6 TabICLv2 completes."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "artifacts" / "v5_automl"
V5E = ROOT / "artifacts" / "v5_automl_ensemble"
V6 = ROOT / "artifacts" / "v6_tabicl"
OUT = ROOT / "artifacts" / "v6_ensemble"
PARENT = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.15, 0.60, 91)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def subsets(items: list[str]):
    for size in range(1, len(items) + 1):
        yield from itertools.combinations(items, size)


def mean_prob(frame: pd.DataFrame, members) -> np.ndarray:
    return frame.loc[:, list(members)].mean(axis=1).to_numpy(dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    shortlist = pd.read_csv(V5E / "candidate_shortlist.csv")
    names = shortlist["model"].tolist()
    v5_oof = pd.read_csv(V5 / "oof_probabilities.csv")
    v5_audit = pd.read_csv(V5 / "audit_probabilities.csv")
    v5_test = pd.read_csv(V5 / "test_probabilities.csv")
    v6_oof = pd.read_csv(V6 / "oof.csv")
    v6_audit = pd.read_csv(V6 / "audit_probabilities.csv")
    v6_test = pd.read_csv(V6 / "test_probabilities.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv")
    labels = pd.read_csv(ROOT / "data" / "raw" / "train_labels.csv")["vacc_h1n1_f"].to_numpy(dtype=int)

    if not np.array_equal(v5_oof["row_id"].to_numpy(), v6_oof["row_id"].to_numpy()):
        raise ValueError("v5/v6 OOF row order mismatch")
    if not np.array_equal(v5_audit["row_id"].to_numpy(), v6_audit["row_id"].to_numpy()):
        raise ValueError("v5/v6 audit row order mismatch")
    if not np.array_equal(v5_test["Id"].to_numpy(), v6_test["Id"].to_numpy()):
        raise ValueError("v5/v6 test order mismatch")

    oof = v5_oof[["row_id", "target"] + names].copy()
    oof["TabICLv2"] = v6_oof["probability"].to_numpy()
    audit = v5_audit[["row_id"] + names].copy()
    audit["TabICLv2"] = v6_audit["probability"].to_numpy()
    test = v5_test[["Id"] + names].copy()
    test["TabICLv2"] = v6_test["probability"].to_numpy()
    names = names + ["TabICLv2"]
    combos = list(subsets(names))

    row_ids = oof["row_id"].to_numpy(dtype=int)
    y = oof["target"].to_numpy(dtype=int)
    folds = manifest.set_index("row_id").loc[row_ids, "dev_fold"].to_numpy(dtype=int)
    oof[names].corr().to_csv(OUT / "candidate_correlation.csv")

    standalone = []
    for name in names:
        t, f1 = select_threshold(y, oof[name].to_numpy())
        standalone.append({"model": name, "threshold": t, "dev_oof_f1": f1, "roc_auc": roc_auc_score(y, oof[name])})
    standalone = pd.DataFrame(standalone).sort_values(["dev_oof_f1", "roc_auc"], ascending=False)
    standalone.to_csv(OUT / "standalone.csv", index=False)

    full_rows = []
    for members in combos:
        p = mean_prob(oof, members)
        t, score = select_threshold(y, p)
        fs = [f1_score(y[folds == f], p[folds == f] >= t, zero_division=0) for f in range(5)]
        full_rows.append({
            "members": "|".join(members), "n_members": len(members), "threshold": t,
            "dev_oof_f1": score, "roc_auc": roc_auc_score(y, p),
            "fold_f1_std": float(np.std(fs, ddof=1)), "fold_f1": json.dumps([float(x) for x in fs]),
        })
    full = pd.DataFrame(full_rows).sort_values(["dev_oof_f1", "fold_f1_std", "n_members"], ascending=[False, True, True])
    full.to_csv(OUT / "equal_weight_all_subsets.csv", index=False)

    nested_rows = []
    ens_pred = np.zeros(len(y), dtype=np.int8)
    single_pred = np.zeros(len(y), dtype=np.int8)
    selected_sets = []
    selected_singles = []
    for outer in range(5):
        tr = folds != outer
        va = folds == outer
        best_ens = None
        for members in combos:
            t, score = select_threshold(y[tr], mean_prob(oof.loc[tr], members))
            key = (score, -len(members), members, t)
            if best_ens is None or key[:2] > best_ens[:2]:
                best_ens = key
        assert best_ens is not None
        inner_f1, _, members, t = best_ens
        ens_pred[va] = (mean_prob(oof.loc[va], members) >= t).astype(np.int8)
        selected_sets.append(members)

        best_single = None
        for name in names:
            st, ss = select_threshold(y[tr], oof.loc[tr, name].to_numpy())
            key = (ss, name, st)
            if best_single is None or key[0] > best_single[0]:
                best_single = key
        assert best_single is not None
        single_inner, single_name, single_t = best_single
        single_pred[va] = (oof.loc[va, single_name].to_numpy() >= single_t).astype(np.int8)
        selected_singles.append(single_name)
        nested_rows.append({
            "outer_fold": outer,
            "ensemble_members": "|".join(members),
            "ensemble_inner_f1": inner_f1,
            "ensemble_threshold": t,
            "ensemble_outer_f1": f1_score(y[va], ens_pred[va], zero_division=0),
            "single_model": single_name,
            "single_inner_f1": single_inner,
            "single_threshold": single_t,
            "single_outer_f1": f1_score(y[va], single_pred[va], zero_division=0),
        })
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(OUT / "nested_selection.csv", index=False)

    counts = {name: sum(name in chosen for chosen in selected_sets) for name in names}
    stable = tuple(name for name in names if counts[name] >= 3)
    if not stable:
        stable = tuple(full.iloc[0]["members"].split("|"))
    stable_p = mean_prob(oof, stable)
    stable_t, stable_f1 = select_threshold(y, stable_p)
    audit_ids = audit["row_id"].to_numpy(dtype=int)
    audit_y = labels[audit_ids]
    stable_audit_p = mean_prob(audit, stable)
    stable_audit_f1 = f1_score(audit_y, stable_audit_p >= stable_t, zero_division=0)

    v2 = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[row_ids, "probability"].to_numpy(dtype=float)
    v2_pred = np.zeros(len(y), dtype=np.int8)
    for outer in range(5):
        tr = folds != outer
        va = folds == outer
        vt, _ = select_threshold(y[tr], v2[tr])
        v2_pred[va] = (v2[va] >= vt).astype(np.int8)

    pd.DataFrame({"Id": test["Id"], "probability": mean_prob(test, stable)}).to_csv(OUT / "stable_ensemble_test_probability.csv", index=False)
    result = {
        "candidate_count": len(names),
        "candidates": names,
        "full_dev_best_equal_weight": {
            "members": full.iloc[0]["members"].split("|"),
            "dev_oof_f1": float(full.iloc[0]["dev_oof_f1"]),
            "threshold": float(full.iloc[0]["threshold"]),
            "note": "Selection-biased diagnostic; nested result is primary.",
        },
        "nested": {
            "ensemble_pooled_f1": float(f1_score(y, ens_pred, zero_division=0)),
            "single_selection_pooled_f1": float(f1_score(y, single_pred, zero_division=0)),
            "v2_pooled_f1": float(f1_score(y, v2_pred, zero_division=0)),
            "ensemble_fold_f1": [float(x) for x in nested["ensemble_outer_f1"]],
            "single_fold_f1": [float(x) for x in nested["single_outer_f1"]],
            "selected_sets": [list(x) for x in selected_sets],
            "member_selection_counts": counts,
            "selected_singles": selected_singles,
        },
        "stable_recipe": {
            "members": list(stable),
            "dev_oof_f1": float(stable_f1),
            "threshold": float(stable_t),
            "audit_f1": float(stable_audit_f1),
        },
        "submitted": False,
        "limitations": [
            "Full-development subset ranking is selection-biased; nested selection is the main evidence.",
            "The audit split is reused and diagnostic only.",
            "No optimized weights or stacking are used; this remains the ensemble-first stage.",
            "No Kaggle submission is created or sent."
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": "v6_ensemble", "id": "equal_weight_with_tabiclv2", "parent": "v6_tabicl",
            "hypothesis": "Adding TabICLv2 to the diverse AutoML family pool improves nested equal-weight ensemble F1.",
            "validation": "8 candidates; exhaustive equal-weight subsets; 5-fold nested membership and threshold selection on frozen folds.",
            "result": result["nested"], "decision": "diagnostic", "artifact": "artifacts/v6_ensemble", "submitted": False,
        }, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
