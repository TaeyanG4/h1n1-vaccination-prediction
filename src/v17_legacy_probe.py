"""Cheap grouped-CV probe of recovered 2023 XGBoost recipes.

This is diagnostic only. It does not alter the Optuna study and does not submit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score

import v17_xgb_optuna as v

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v17_legacy_probe"


def recipe(**kw):
    base = dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=0.2,
        colsample_bylevel=0.9,
        reg_lambda=1.0,
        reg_alpha=1e-5,
        gamma=1.0,
        scale_pos_weight=2.5714285714,
        max_delta_step=0,
        max_bin=256,
        max_cat_to_onehot=8,
        max_cat_threshold=64,
        grow_policy="depthwise",
        max_leaves=0,
    )
    base.update(kw)
    return base


RECIPES = [
    ("legacy_final_484", "legacy_ordinal", 484, recipe()),
    ("legacy_gridbest_276", "legacy_ordinal", 276, recipe(subsample=0.5, colsample_bytree=0.5, colsample_bylevel=0.5)),
    ("legacy_alt_338", "legacy_ordinal", 338, recipe(learning_rate=0.1, max_depth=9, colsample_bytree=0.7, colsample_bylevel=0.5, gamma=0.0, scale_pos_weight=4.0)),
    ("historical_fe_484", "historical", 484, recipe()),
]


def main():
    cfg = v.read_json(ROOT / "configs" / "v17_xgb_optuna.json")
    train, test, y, sample, manifest, dev, audit, fold_ids = v.load_data()
    cache = v.feature_cache(train, test, cfg["feature_modes"])
    ydev = y[dev]
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)

    for name, mode, rounds, params in RECIPES:
        oof = np.full(len(dev), np.nan, dtype=float)
        audit_parts = []
        for fold in range(5):
            tr_ids = dev[fold_ids != fold]
            va_pos = np.flatnonzero(fold_ids == fold)
            va_ids = dev[va_pos]
            model, transformer, p, _ = v.fit_fold(
                params, cfg, cache[mode][0], y, tr_ids, va_ids,
                seed=20230114 + fold, n_estimators=rounds,
                early_stopping=False, mode=mode,
            )
            oof[va_pos] = p
            audit_parts.append(model.predict_proba(v.transform_representation(cache[mode][0].iloc[audit], transformer))[:, 1])

        nested = v.nested_threshold_metrics(ydev, oof, fold_ids, list(range(5)), cfg)
        tuned_t, tuned_f1 = v.select_threshold(ydev, oof, cfg)
        audit_p = np.mean(audit_parts, axis=0)
        row = {
            "name": name,
            "mode": mode,
            "rounds": rounds,
            "params": params,
            "f1_at_0_5": float(f1_score(ydev, oof >= 0.5, zero_division=0)),
            "positive_rate_at_0_5": float(np.mean(oof >= 0.5)),
            "tuned_threshold": tuned_t,
            "tuned_f1": tuned_f1,
            "nested_f1": nested["nested_f1"],
            "nested_thresholds": nested["thresholds"],
            "nested_fold_std": nested["fold_std"],
            "auc": float(roc_auc_score(ydev, oof)),
            "logloss": float(log_loss(ydev, oof, labels=[0, 1])),
            "audit_f1_at_nested_median_threshold": float(f1_score(y[audit], audit_p >= nested["median_threshold"], zero_division=0)),
        }
        rows.append(row)
        np.savez_compressed(OUT / f"{name}.npz", oof=oof, audit=audit_p)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rows.sort(key=lambda r: (r["nested_f1"], r["tuned_f1"], r["auc"]), reverse=True)
    (OUT / "results.json").write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
