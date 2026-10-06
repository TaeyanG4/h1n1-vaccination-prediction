from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "artifacts" / "v5_automl"
OUT = ROOT / "artifacts" / "v5_automl_ensemble"
PARENT = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.15, 0.60, 91)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def family(name: str) -> str | None:
    if name.startswith("CatBoost"):
        return "CatBoost"
    if name.startswith("ExtraTrees"):
        return "ExtraTrees"
    if name.startswith("RandomForest"):
        return "RandomForest"
    if name.startswith("NeuralNetFastAI"):
        return "FastAI"
    if name.startswith("NeuralNetTorch"):
        return "TorchNN"
    if name.startswith("XGBoost"):
        return "XGBoost"
    if name.startswith("LightGBM"):
        return "LightGBM"
    return None


def all_subsets(items: list[str]):
    for size in range(1, len(items) + 1):
        for combo in itertools.combinations(items, size):
            yield combo


def mean_prob(frame: pd.DataFrame, members: tuple[str, ...] | list[str]) -> np.ndarray:
    return frame.loc[:, list(members)].mean(axis=1).to_numpy(dtype=float)


def bootstrap_delta(y: np.ndarray, a: np.ndarray, ta: float, b: np.ndarray, tb: float, n: int = 2000) -> dict:
    rng = np.random.default_rng(20261005)
    pa = (a >= ta).astype(np.int8)
    pb = (b >= tb).astype(np.int8)
    deltas = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        deltas[i] = f1_score(y[idx], pa[idx], zero_division=0) - f1_score(y[idx], pb[idx], zero_division=0)
    return {
        "mean": float(deltas.mean()),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)

    scores = pd.read_csv(AUTO / "model_scores.csv")
    oof = pd.read_csv(AUTO / "oof_probabilities.csv")
    audit = pd.read_csv(AUTO / "audit_probabilities.csv")
    test = pd.read_csv(AUTO / "test_probabilities.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv")
    labels = pd.read_csv(ROOT / "data" / "raw" / "train_labels.csv")["vacc_h1n1_f"].to_numpy()

    dev_ids = oof["row_id"].to_numpy(dtype=int)
    y = oof["target"].to_numpy(dtype=int)
    folds = manifest.set_index("row_id").loc[dev_ids, "dev_fold"].to_numpy(dtype=int)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Frozen folds are not 0..4")

    # One representative per genuinely different AutoML family. Selection is
    # based only on each model's standalone development OOF score.
    scores = scores.copy()
    scores["family"] = scores["model"].map(family)
    candidate_rows = []
    for fam in ["CatBoost", "ExtraTrees", "RandomForest", "FastAI", "TorchNN", "XGBoost", "LightGBM"]:
        sub = scores[scores.family == fam].sort_values(["dev_oof_tuned_f1", "roc_auc"], ascending=False)
        if len(sub):
            candidate_rows.append(sub.iloc[0])
    candidates = pd.DataFrame(candidate_rows).reset_index(drop=True)
    names = candidates["model"].tolist()
    candidates.to_csv(OUT / "candidate_shortlist.csv", index=False)
    oof[names].corr().to_csv(OUT / "candidate_correlation.csv")

    subsets = list(all_subsets(names))

    # Full-dev diagnostic ranking. This is intentionally labelled optimistic;
    # the nested outer-fold result below is the primary evidence surface.
    full_rows = []
    for members in subsets:
        p = mean_prob(oof, members)
        threshold, score = select_threshold(y, p)
        fold_scores = [
            f1_score(y[folds == f], p[folds == f] >= threshold, zero_division=0)
            for f in range(5)
        ]
        full_rows.append({
            "members": "|".join(members),
            "n_members": len(members),
            "threshold": threshold,
            "dev_oof_f1": score,
            "roc_auc": roc_auc_score(y, p),
            "fold_f1_std": float(np.std(fold_scores, ddof=1)),
            "fold_f1": json.dumps([float(x) for x in fold_scores]),
        })
    full = pd.DataFrame(full_rows).sort_values(
        ["dev_oof_f1", "fold_f1_std", "n_members"], ascending=[False, True, True]
    )
    full.to_csv(OUT / "equal_weight_all_subsets.csv", index=False)

    # Nested membership selection: for each outer frozen fold, choose the best
    # equal-weight subset and threshold using only the other four folds.
    nested_rows = []
    nested_pred = np.zeros(len(y), dtype=np.int8)
    single_pred = np.zeros(len(y), dtype=np.int8)
    selected_sets: list[tuple[str, ...]] = []
    best_single = str(candidates.sort_values("dev_oof_tuned_f1", ascending=False).iloc[0]["model"])

    for outer in range(5):
        tr = folds != outer
        va = folds == outer
        best = None
        for members in subsets:
            p_tr = mean_prob(oof.loc[tr], members)
            threshold, inner_f1 = select_threshold(y[tr], p_tr)
            row = (inner_f1, -len(members), members, threshold)
            if best is None or row[:2] > best[:2]:
                best = row
        assert best is not None
        inner_f1, _, members, threshold = best
        selected_sets.append(members)
        p_va = mean_prob(oof.loc[va], members)
        pred_va = (p_va >= threshold).astype(np.int8)
        nested_pred[va] = pred_va

        single_threshold, single_inner_f1 = select_threshold(y[tr], oof.loc[tr, best_single].to_numpy())
        single_va = oof.loc[va, best_single].to_numpy()
        single_pred[va] = (single_va >= single_threshold).astype(np.int8)

        nested_rows.append({
            "outer_fold": outer,
            "selected_members": "|".join(members),
            "n_members": len(members),
            "inner_f1": inner_f1,
            "threshold": threshold,
            "outer_f1": f1_score(y[va], pred_va, zero_division=0),
            "best_single": best_single,
            "best_single_inner_f1": single_inner_f1,
            "best_single_threshold": single_threshold,
            "best_single_outer_f1": f1_score(y[va], single_pred[va], zero_division=0),
        })
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(OUT / "nested_selection.csv", index=False)

    # Stable recipe: members chosen in at least 3/5 outer-fold searches.
    counts = {name: sum(name in subset for subset in selected_sets) for name in names}
    stable = [name for name in names if counts[name] >= 3]
    if not stable:
        stable = full.iloc[0]["members"].split("|")
    stable = tuple(stable)

    stable_p = mean_prob(oof, stable)
    stable_threshold, stable_dev_f1 = select_threshold(y, stable_p)
    audit_ids = audit["row_id"].to_numpy(dtype=int)
    audit_y = labels[audit_ids]
    stable_audit_p = mean_prob(audit, stable)
    stable_audit_f1 = f1_score(audit_y, stable_audit_p >= stable_threshold, zero_division=0)

    best_single_threshold, best_single_dev_f1 = select_threshold(y, oof[best_single].to_numpy())
    best_single_audit_p = audit[best_single].to_numpy(dtype=float)
    best_single_audit_f1 = f1_score(audit_y, best_single_audit_p >= best_single_threshold, zero_division=0)

    # v2 nested baseline for a familiar project anchor.
    v2 = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[dev_ids, "probability"].to_numpy(dtype=float)
    v2_nested_pred = np.zeros(len(y), dtype=np.int8)
    v2_fold_f1 = []
    for outer in range(5):
        tr = folds != outer
        va = folds == outer
        t, _ = select_threshold(y[tr], v2[tr])
        v2_nested_pred[va] = (v2[va] >= t).astype(np.int8)
        v2_fold_f1.append(float(f1_score(y[va], v2_nested_pred[va], zero_division=0)))

    pd.DataFrame({
        "Id": test["Id"],
        "probability": mean_prob(test, stable),
    }).to_csv(OUT / "stable_ensemble_test_probability.csv", index=False)

    result = {
        "candidate_families": {row["family"]: row["model"] for _, row in candidates.iterrows()},
        "candidate_count": len(names),
        "selection_method": "One standalone-best model per family, then equal-weight exhaustive subset selection. Primary evidence is 5-fold nested membership/threshold selection using the frozen folds.",
        "best_single": {
            "model": best_single,
            "dev_oof_f1": float(best_single_dev_f1),
            "threshold": float(best_single_threshold),
            "audit_f1": float(best_single_audit_f1),
        },
        "full_dev_best_equal_weight": {
            "members": full.iloc[0]["members"].split("|"),
            "dev_oof_f1": float(full.iloc[0]["dev_oof_f1"]),
            "threshold": float(full.iloc[0]["threshold"]),
            "fold_f1_std": float(full.iloc[0]["fold_f1_std"]),
            "note": "Optimistic diagnostic because subset and threshold are selected on the same full development OOF.",
        },
        "nested": {
            "pooled_f1": float(f1_score(y, nested_pred, zero_division=0)),
            "fold_f1": [float(x) for x in nested["outer_f1"]],
            "best_single_pooled_f1": float(f1_score(y, single_pred, zero_division=0)),
            "best_single_fold_f1": [float(x) for x in nested["best_single_outer_f1"]],
            "v2_pooled_f1": float(f1_score(y, v2_nested_pred, zero_division=0)),
            "v2_fold_f1": v2_fold_f1,
            "selected_sets": [list(x) for x in selected_sets],
            "member_selection_counts": counts,
        },
        "stable_recipe": {
            "members": list(stable),
            "dev_oof_f1": float(stable_dev_f1),
            "threshold": float(stable_threshold),
            "audit_f1": float(stable_audit_f1),
            "audit_delta_vs_best_automl_single": float(stable_audit_f1 - best_single_audit_f1),
            "audit_bootstrap_delta_vs_best_single": bootstrap_delta(
                audit_y,
                stable_audit_p,
                stable_threshold,
                best_single_audit_p,
                best_single_threshold,
            ),
        },
        "submitted": False,
        "limitations": [
            "The full-development subset leaderboard is selection-biased; use the nested result as the main evidence.",
            "The audit split has been inspected in earlier versions and is diagnostic only.",
            "No optimized blend weights or stacking are used here; this is the ensemble-first stage.",
            "No Kaggle submission is created or sent.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
