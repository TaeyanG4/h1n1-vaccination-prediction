"""Post-hoc robustness diagnostic for the v3 CatBoost-v2/XGBoost blend.

This does not change the frozen v3 selection. For each held-out development
fold, blend weight and threshold are chosen only on the other four OOF folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.15, 0.60, 91)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def main() -> None:
    manifest = pd.read_csv(
        ROOT / "artifacts" / "baseline_v1" / "split_manifest.csv",
        dtype={"group_hash": str},
    )
    labels = pd.read_csv(ROOT / "data" / "raw" / "train_labels.csv")[TARGET].to_numpy()
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy()
    folds = manifest.loc[dev, "dev_fold"].to_numpy()

    v2 = (
        pd.read_csv(ROOT / "artifacts" / "v2" / "selected_oof.csv")
        .set_index("row_id")
        .loc[dev]
    )
    xgb = (
        pd.read_csv(
            ROOT
            / "artifacts"
            / "v3"
            / "candidates"
            / "xgb_d4_survey"
            / "oof.csv"
        )
        .set_index("row_id")
        .loc[dev]
    )
    y = labels[dev]
    if not np.array_equal(v2.target.to_numpy(), y) or not np.array_equal(
        xgb.target.to_numpy(), y
    ):
        raise ValueError("OOF target alignment failure")
    p_v2 = v2.probability.to_numpy()
    p_xgb = xgb.probability.to_numpy()

    weights = np.round(np.arange(0.05, 0.51, 0.05), 2)
    fixed_weight = 0.25
    rows = []
    nested_pred = np.zeros(len(dev), dtype=np.int8)
    baseline_pred = np.zeros(len(dev), dtype=np.int8)
    fixed_pred = np.zeros(len(dev), dtype=np.int8)

    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold

        v2_t, _ = best_threshold(y[train], p_v2[train])
        baseline_pred[valid] = (p_v2[valid] >= v2_t).astype(np.int8)
        baseline_f1 = float(f1_score(y[valid], baseline_pred[valid], zero_division=0))

        best = None
        for w in weights:
            p_train = (1.0 - w) * p_v2[train] + w * p_xgb[train]
            threshold, score = best_threshold(y[train], p_train)
            candidate = (score, -abs(w - fixed_weight), -w, w, threshold)
            if best is None or candidate > best:
                best = candidate
        _, _, _, selected_weight, selected_threshold = best
        p_valid = (1.0 - selected_weight) * p_v2[valid] + selected_weight * p_xgb[valid]
        nested_pred[valid] = (p_valid >= selected_threshold).astype(np.int8)
        nested_f1 = float(f1_score(y[valid], nested_pred[valid], zero_division=0))

        fixed_train = (1.0 - fixed_weight) * p_v2[train] + fixed_weight * p_xgb[train]
        fixed_threshold, _ = best_threshold(y[train], fixed_train)
        fixed_valid = (1.0 - fixed_weight) * p_v2[valid] + fixed_weight * p_xgb[valid]
        fixed_pred[valid] = (fixed_valid >= fixed_threshold).astype(np.int8)
        fixed_f1 = float(f1_score(y[valid], fixed_pred[valid], zero_division=0))

        rows.append(
            {
                "fold": int(fold),
                "v2_threshold_train4": v2_t,
                "v2_f1": baseline_f1,
                "selected_xgb_weight_train4": float(selected_weight),
                "selected_threshold_train4": float(selected_threshold),
                "nested_blend_f1": nested_f1,
                "nested_delta_vs_v2": nested_f1 - baseline_f1,
                "fixed_xgb_weight": fixed_weight,
                "fixed_threshold_train4": fixed_threshold,
                "fixed_blend_f1": fixed_f1,
                "fixed_delta_vs_v2": fixed_f1 - baseline_f1,
            }
        )

    frame = pd.DataFrame(rows)
    out = ROOT / "artifacts" / "v3" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "blend_nested_cv.csv", index=False)
    result = {
        "diagnostic": "leave-one-development-fold-out blend/threshold selection",
        "post_hoc_warning": "Weight grid was introduced after observing v3; use as robustness evidence, not unbiased promotion evidence.",
        "weight_grid": weights.tolist(),
        "fixed_weight": fixed_weight,
        "folds": rows,
        "nested_pooled_f1": float(f1_score(y, nested_pred, zero_division=0)),
        "baseline_v2_nested_pooled_f1": float(
            f1_score(y, baseline_pred, zero_division=0)
        ),
        "nested_pooled_delta": float(
            f1_score(y, nested_pred, zero_division=0)
            - f1_score(y, baseline_pred, zero_division=0)
        ),
        "fixed25_pooled_f1": float(f1_score(y, fixed_pred, zero_division=0)),
        "fixed25_pooled_delta": float(
            f1_score(y, fixed_pred, zero_division=0)
            - f1_score(y, baseline_pred, zero_division=0)
        ),
        "selected_weight_mean": float(frame.selected_xgb_weight_train4.mean()),
        "selected_weight_min": float(frame.selected_xgb_weight_train4.min()),
        "selected_weight_max": float(frame.selected_xgb_weight_train4.max()),
        "positive_folds_nested": int((frame.nested_delta_vs_v2 > 0).sum()),
        "positive_folds_fixed25": int((frame.fixed_delta_vs_v2 > 0).sum()),
    }
    (out / "blend_nested_cv.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
