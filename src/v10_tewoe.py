"""Compact leakage-safe target-encoding + WOE experiment.

The representation deliberately avoids one-hot expansion. Eight selected categorical
columns receive two supervised numeric features each (TE and WOE), while the proven
64-column v2 survey representation and its native categorical columns are retained.
Total width is therefore 80 columns, not hundreds of dummy columns.

All supervised encoders are fitted only on each fold's training labels. Validation,
audit and test rows are transformed from those frozen mappings. No train+test fitting,
no target leakage and no Kaggle submission occur in this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from v2_features import build_features, categorical_columns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v10_tewoe"
V1 = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"
V5 = ROOT / "artifacts" / "v5_automl"
TARGET = "vacc_h1n1_f"
V5_MODEL = "CatBoost_BAG_L2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    expected = read_json(V1 / "metrics.json")
    if sha(V1 / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen split changed")
    return {"passed": True, "protected_files": len(protected)}


def load_data():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    manifest = pd.read_csv(V1 / "split_manifest.csv", dtype={"group_hash": str})
    if len(train) != len(labels) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(train))):
        raise ValueError("Row identity mismatch")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    folds = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    return train, test, labels[TARGET].to_numpy(dtype=int), manifest, dev, audit, folds


def key_series(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "__MISSING__" if pd.isna(v) else str(v))


def fit_supervised_encoder(frame: pd.DataFrame, y: np.ndarray, columns: list[str], smoothing: float, alpha: float) -> dict:
    global_mean = float(np.mean(y))
    global_odds = (float(np.sum(y)) + alpha) / (float(len(y) - np.sum(y)) + alpha)
    enc = {"global_mean": global_mean, "global_log_odds": float(np.log(global_odds)), "columns": {}}
    for col in columns:
        tmp = pd.DataFrame({"key": key_series(frame[col]), "y": y})
        stats = tmp.groupby("key", sort=False)["y"].agg(["sum", "count"])
        te = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
        neg = stats["count"] - stats["sum"]
        woe = np.log((stats["sum"] + alpha) / (neg + alpha)) - enc["global_log_odds"]
        enc["columns"][col] = {
            "te": {str(k): float(v) for k, v in te.items()},
            "woe": {str(k): float(v) for k, v in woe.items()},
        }
    return enc


def add_supervised_features(frame: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    out = frame.copy()
    for col, maps in encoder["columns"].items():
        keys = key_series(frame[col])
        out[f"v10_{col}_te"] = keys.map(maps["te"]).fillna(float(encoder["global_mean"])).astype(float)
        out[f"v10_{col}_woe"] = keys.map(maps["woe"]).fillna(0.0).astype(float)
    return out


def fit_native_schema(frame: pd.DataFrame) -> dict:
    cats = categorical_columns(frame)
    levels = {c: sorted(frame[c].dropna().astype(str).unique().tolist()) for c in cats}
    return {"columns": list(frame.columns), "categorical": cats, "levels": levels}


def transform_native(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if list(frame.columns) != schema["columns"]:
        raise ValueError("Feature schema/order mismatch")
    out = frame.copy()
    cats = set(schema["categorical"])
    for col in out.columns:
        if col in cats:
            vals = out[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
            out[col] = pd.Categorical(vals, categories=schema["levels"][col])
        else:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    return out


def thresholds(config: dict) -> np.ndarray:
    return np.linspace(float(config["threshold_min"]), float(config["threshold_max"]), int(config["threshold_steps"]))


def select_threshold(y: np.ndarray, p: np.ndarray, config: dict) -> tuple[float, float]:
    grid = thresholds(config)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def metric_bundle(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "positive_prediction_rate": float(pred.mean()),
    }


def summarize(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray, config: dict) -> dict:
    t, score = select_threshold(y, p, config)
    fold_f1 = [float(f1_score(y[fold_ids == f], p[fold_ids == f] >= t, zero_division=0)) for f in range(5)]
    return {
        "threshold": t, "dev_oof_f1": score, "metrics": metric_bundle(y, p, t),
        "fold_f1": fold_f1, "fold_f1_std": float(np.std(fold_f1, ddof=1)),
    }


def nested_threshold(y: np.ndarray, p: np.ndarray, folds: np.ndarray, config: dict) -> dict:
    pred = np.zeros(len(y), dtype=bool)
    rows = []
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        t, train_f1 = select_threshold(y[tr], p[tr], config)
        pred[va] = p[va] >= t
        rows.append({"fold": fold, "threshold": t, "train_f1": train_f1,
                     "heldout_f1": float(f1_score(y[va], pred[va], zero_division=0))})
    return {"pooled_f1": float(f1_score(y, pred, zero_division=0)), "folds": rows}


def make_model(family: str, config: dict, seed: int):
    if family == "xgboost":
        p = dict(config["xgboost"])
        return XGBClassifier(
            objective="binary:logistic", enable_categorical=True,
            n_estimators=int(p["n_estimators"]), early_stopping_rounds=int(p["early_stopping_rounds"]),
            tree_method=p["tree_method"], eval_metric=p["eval_metric"], max_cat_to_onehot=int(p["max_cat_to_onehot"]),
            max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]),
            min_child_weight=float(p["min_child_weight"]), subsample=float(p["subsample"]),
            colsample_bytree=float(p["colsample_bytree"]), reg_lambda=float(p["reg_lambda"]),
            reg_alpha=float(p["reg_alpha"]), gamma=float(p["gamma"]), random_state=seed,
            n_jobs=int(config["threads"]), verbosity=0,
        )
    p = dict(config["lightgbm"])
    return LGBMClassifier(
        objective="binary", n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]),
        num_leaves=int(p["num_leaves"]), max_depth=int(p["max_depth"]), min_child_samples=int(p["min_child_samples"]),
        reg_lambda=float(p["reg_lambda"]), reg_alpha=float(p["reg_alpha"]), colsample_bytree=float(p["colsample_bytree"]),
        subsample=float(p["subsample"]), subsample_freq=int(p["subsample_freq"]), verbosity=int(p["verbosity"]),
        deterministic=bool(p["deterministic"]), force_col_wise=bool(p["force_col_wise"]),
        random_state=seed, n_jobs=int(config["threads"]),
    )


def fold_paths(family: str, fold: int):
    d = OUT / "candidates" / family
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fold{fold}.joblib", d / f"fold{fold}_oof.csv", d / f"fold{fold}_predictions.npz", d / f"fold{fold}_metrics.json"


def fit_fold(family: str, config: dict, fold: int, train: pd.DataFrame, test: pd.DataFrame,
             y: np.ndarray, manifest: pd.DataFrame, dev: np.ndarray, audit: np.ndarray,
             folds: np.ndarray, resume: bool) -> None:
    model_path, oof_path, pred_path, receipt_path = fold_paths(family, fold)
    train_ids, valid_ids = dev[folds != fold], dev[folds == fold]
    if set(manifest.loc[train_ids, "group_hash"]) & set(manifest.loc[valid_ids, "group_hash"]):
        raise ValueError("Duplicate group crosses development fold")
    if resume and all(p.is_file() for p in [model_path, oof_path, pred_path, receipt_path]):
        saved = pd.read_csv(oof_path)
        if not np.array_equal(saved.row_id.to_numpy(), valid_ids):
            raise ValueError(f"Resume alignment failure {family} fold {fold}")
        print(f"REUSE {family} fold={fold}", flush=True)
        return
    if any(p.exists() for p in [model_path, oof_path, pred_path, receipt_path]):
        raise FileExistsError(f"Partial artifact exists for {family} fold {fold}")

    started = time.perf_counter()
    base = build_features(train, config["feature_mode"])
    test_base = build_features(test, config["feature_mode"])
    enc = fit_supervised_encoder(
        base.iloc[train_ids], y[train_ids], list(config["encoded_columns"]),
        float(config["target_encoding_smoothing"]), float(config["woe_alpha"]),
    )
    train_aug = add_supervised_features(base.iloc[train_ids], enc)
    valid_aug = add_supervised_features(base.iloc[valid_ids], enc)
    audit_aug = add_supervised_features(base.iloc[audit], enc)
    test_aug = add_supervised_features(test_base, enc)
    schema = fit_native_schema(train_aug)
    train_x = transform_native(train_aug, schema)
    valid_x = transform_native(valid_aug, schema)
    audit_x = transform_native(audit_aug, schema)
    test_x = transform_native(test_aug, schema)
    model = make_model(family, config, int(config["seed"]) + fold)
    if family == "xgboost":
        model.fit(train_x, y[train_ids], eval_set=[(valid_x, y[valid_ids])], verbose=False)
        best_iteration = int(model.best_iteration)
    else:
        model.fit(
            train_x, y[train_ids], eval_set=[(valid_x, y[valid_ids])], eval_metric="binary_logloss",
            categorical_feature=schema["categorical"],
            callbacks=[early_stopping(int(config["lightgbm"]["early_stopping_rounds"]), verbose=False)],
        )
        best_iteration = int(model.best_iteration_)
    val_p = model.predict_proba(valid_x)[:, 1]
    audit_p = model.predict_proba(audit_x)[:, 1]
    test_p = model.predict_proba(test_x)[:, 1]
    if not all(np.isfinite(v).all() for v in [val_p, audit_p, test_p]):
        raise ValueError("Nonfinite probabilities")
    joblib.dump({"model": model, "encoder": enc, "schema": schema, "family": family}, model_path, compress=3)
    pd.DataFrame({"row_id": valid_ids, "target": y[valid_ids], "probability": val_p}).to_csv(oof_path, index=False)
    np.savez_compressed(pred_path, audit=audit_p, test=test_p)
    receipt = {
        "family": family, "fold": fold, "train_rows": len(train_ids), "valid_rows": len(valid_ids),
        "base_features": int(base.shape[1]), "encoded_columns": len(config["encoded_columns"]),
        "final_features": int(train_x.shape[1]), "best_iteration": best_iteration,
        "seconds": float(time.perf_counter() - started), "completed_utc": now(),
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def assemble_family(family: str, y_dev: np.ndarray, dev: np.ndarray, folds: np.ndarray, config: dict):
    d = OUT / "candidates" / family
    oof = pd.concat([pd.read_csv(d / f"fold{f}_oof.csv") for f in range(5)]).set_index("row_id").loc[dev].reset_index()
    if not np.array_equal(oof.target.to_numpy(), y_dev):
        raise ValueError(f"OOF target mismatch {family}")
    audit_parts, test_parts, seconds, iterations = [], [], [], []
    for f in range(5):
        with np.load(d / f"fold{f}_predictions.npz") as z:
            audit_parts.append(z["audit"]); test_parts.append(z["test"])
        rec = read_json(d / f"fold{f}_metrics.json")
        seconds.append(float(rec["seconds"])); iterations.append(int(rec["best_iteration"]))
    p = oof.probability.to_numpy()
    ap = np.mean(np.vstack(audit_parts), axis=0)
    tp = np.mean(np.vstack(test_parts), axis=0)
    result = summarize(y_dev, p, folds, config)
    result.update({"nested": nested_threshold(y_dev, p, folds, config), "training_seconds": sum(seconds),
                   "best_iterations": iterations, "median_best_iteration": int(np.median(iterations))})
    oof.to_csv(d / "oof.csv", index=False)
    np.savez_compressed(d / "ensemble_predictions.npz", audit=ap, test=tp)
    write_json(d / "development_metrics.json", result)
    return {"oof": p, "audit": ap, "test": tp, "result": result}


def load_references(y_dev: np.ndarray, dev: np.ndarray, audit: np.ndarray, folds: np.ndarray, config: dict):
    v2_frame = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[dev]
    if not np.array_equal(v2_frame.target.to_numpy(), y_dev):
        raise ValueError("v2 reference alignment failure")
    v2_audit = pd.read_csv(V2 / "selected_audit.csv").set_index("row_id").loc[audit, "candidate_probability"].to_numpy()

    v5_oof_frame = pd.read_csv(V5 / "oof_probabilities.csv", usecols=["row_id", "target", V5_MODEL]).set_index("row_id").loc[dev]
    if not np.array_equal(v5_oof_frame.target.to_numpy(), y_dev):
        raise ValueError("v5 OOF alignment failure")
    v5_audit = pd.read_csv(V5 / "audit_probabilities.csv", usecols=["row_id", V5_MODEL]).set_index("row_id").loc[audit, V5_MODEL].to_numpy()
    v5_test = pd.read_csv(V5 / "test_probabilities.csv", usecols=["Id", V5_MODEL])[V5_MODEL].to_numpy()
    return {
        "v2": {"oof": v2_frame.probability.to_numpy(), "audit": v2_audit,
               "result": {**summarize(y_dev, v2_frame.probability.to_numpy(), folds, config),
                          "nested": nested_threshold(y_dev, v2_frame.probability.to_numpy(), folds, config)}},
        "v5": {"oof": v5_oof_frame[V5_MODEL].to_numpy(), "audit": v5_audit, "test": v5_test,
               "result": {**summarize(y_dev, v5_oof_frame[V5_MODEL].to_numpy(), folds, config),
                          "nested": nested_threshold(y_dev, v5_oof_frame[V5_MODEL].to_numpy(), folds, config)}},
    }


def error_overlap(y: np.ndarray, p1: np.ndarray, t1: float, p2: np.ndarray, t2: float) -> dict:
    e1 = (p1 >= t1) != y.astype(bool)
    e2 = (p2 >= t2) != y.astype(bool)
    union = e1 | e2
    return {"both_wrong": int(np.sum(e1 & e2)), "either_wrong": int(np.sum(union)),
            "jaccard": float(np.sum(e1 & e2) / max(1, np.sum(union))),
            "disagree_predictions": int(np.sum((p1 >= t1) != (p2 >= t2)))}


def check(config: dict) -> None:
    train, test, y, _, dev, audit, folds = load_data()
    base = build_features(train.iloc[:200], config["feature_mode"])
    test_base = build_features(test.iloc[:200], config["feature_mode"])
    missing_cols = [c for c in config["encoded_columns"] if c not in base.columns]
    if missing_cols:
        raise ValueError(f"Encoded columns missing: {missing_cols}")
    enc = fit_supervised_encoder(base, y[:len(base)], list(config["encoded_columns"]),
                                 float(config["target_encoding_smoothing"]), float(config["woe_alpha"]))
    aug = add_supervised_features(base, enc)
    aug_test = add_supervised_features(test_base, enc)
    if list(aug.columns) != list(aug_test.columns):
        raise ValueError("Train/test augmented schema mismatch")
    refs = load_references(y[dev], dev, audit, folds, config)
    print(json.dumps({
        "check_passed": True, "integrity": verify_inputs(), "dev_rows": len(dev), "audit_rows": len(audit),
        "base_features": base.shape[1], "encoded_columns": len(config["encoded_columns"]),
        "added_features": 2 * len(config["encoded_columns"]), "final_features": aug.shape[1],
        "encoded_column_cardinality": {c: int(train[c].nunique(dropna=False)) for c in config["encoded_columns"]},
        "v2_reference_f1": refs["v2"]["result"]["dev_oof_f1"],
        "v5_reference_f1": refs["v5"]["result"]["dev_oof_f1"],
        "remote_submission_authorized": bool(config["remote_submission_authorized"]),
    }, indent=2), flush=True)


def smoke(config: dict) -> None:
    train, test, y, manifest, dev, audit, folds = load_data()
    tr = dev[folds != 0][:5000]
    va = dev[folds == 0][:1500]
    base = build_features(train, config["feature_mode"])
    enc = fit_supervised_encoder(base.iloc[tr], y[tr], list(config["encoded_columns"]),
                                 float(config["target_encoding_smoothing"]), float(config["woe_alpha"]))
    tr_aug = add_supervised_features(base.iloc[tr], enc)
    va_aug = add_supervised_features(base.iloc[va], enc)
    schema = fit_native_schema(tr_aug)
    tr_x, va_x = transform_native(tr_aug, schema), transform_native(va_aug, schema)
    rows = []
    for family in ["xgboost", "lightgbm"]:
        local = json.loads(json.dumps(config))
        if family == "xgboost":
            local["xgboost"]["n_estimators"] = 120
            local["xgboost"]["early_stopping_rounds"] = 25
        else:
            local["lightgbm"]["n_estimators"] = 150
            local["lightgbm"]["early_stopping_rounds"] = 25
        model = make_model(family, local, int(config["seed"]))
        started = time.perf_counter()
        if family == "xgboost":
            model.fit(tr_x, y[tr], eval_set=[(va_x, y[va])], verbose=False)
            it = int(model.best_iteration)
        else:
            model.fit(tr_x, y[tr], eval_set=[(va_x, y[va])], eval_metric="binary_logloss",
                      categorical_feature=schema["categorical"], callbacks=[early_stopping(25, verbose=False)])
            it = int(model.best_iteration_)
        p = model.predict_proba(va_x)[:, 1]
        rows.append({"family": family, "features": tr_x.shape[1], "auc": float(roc_auc_score(y[va], p)),
                     "best_iteration": it, "seconds": float(time.perf_counter() - started)})
    report = {"smoke_passed": True, "train_rows": len(tr), "valid_rows": len(va), "models": rows}
    write_json(ROOT / "reports" / "v10_tewoe_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(config: dict, resume: bool) -> None:
    if OUT.exists() and not resume:
        raise FileExistsError("v10 output exists; use --resume or a new version")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config)
    verify_inputs()
    train, test, y, manifest, dev, audit, folds = load_data()
    y_dev, y_audit = y[dev], y[audit]
    refs = load_references(y_dev, dev, audit, folds, config)

    models = {}
    for family in ["xgboost", "lightgbm"]:
        for fold in range(5):
            fit_fold(family, config, fold, train, test, y, manifest, dev, audit, folds, resume)
        models[family] = assemble_family(family, y_dev, dev, folds, config)

    ensembles = {
        "tewoe_xgb_lgb_equal": {
            "oof": 0.5 * (models["xgboost"]["oof"] + models["lightgbm"]["oof"]),
            "audit": 0.5 * (models["xgboost"]["audit"] + models["lightgbm"]["audit"]),
            "test": 0.5 * (models["xgboost"]["test"] + models["lightgbm"]["test"]),
            "components": {"xgboost": 0.5, "lightgbm": 0.5},
        },
        "v5_tewoe_tri_equal": {
            "oof": (refs["v5"]["oof"] + models["xgboost"]["oof"] + models["lightgbm"]["oof"]) / 3.0,
            "audit": (refs["v5"]["audit"] + models["xgboost"]["audit"] + models["lightgbm"]["audit"]) / 3.0,
            "test": (refs["v5"]["test"] + models["xgboost"]["test"] + models["lightgbm"]["test"]) / 3.0,
            "components": {"v5_catboost_bag_l2": 1/3, "xgboost": 1/3, "lightgbm": 1/3},
        },
    }
    for name, obj in ensembles.items():
        obj["result"] = summarize(y_dev, obj["oof"], folds, config)
        obj["result"]["nested"] = nested_threshold(y_dev, obj["oof"], folds, config)

    candidates = {"xgboost_tewoe": models["xgboost"], "lightgbm_tewoe": models["lightgbm"], **ensembles}
    rows = []
    for name, obj in candidates.items():
        r = obj["result"]
        t = float(r["threshold"])
        audit_metrics = metric_bundle(y_audit, obj["audit"], t)
        corr_v5 = float(np.corrcoef(obj["oof"], refs["v5"]["oof"])[0, 1])
        corr_v2 = float(np.corrcoef(obj["oof"], refs["v2"]["oof"])[0, 1])
        overlap = error_overlap(y_dev, obj["oof"], t, refs["v5"]["oof"], refs["v5"]["result"]["threshold"])
        obj["audit_metrics"] = audit_metrics
        obj["correlation_v5"] = corr_v5
        obj["correlation_v2"] = corr_v2
        obj["error_overlap_v5"] = overlap
        rows.append({
            "candidate": name, "dev_oof_f1": r["dev_oof_f1"], "nested_f1": r["nested"]["pooled_f1"],
            "threshold": t, "fold_std": r["fold_f1_std"], "audit_f1": audit_metrics["f1"],
            "corr_v5": corr_v5, "corr_v2": corr_v2, "error_jaccard_v5": overlap["jaccard"],
        })
        pd.DataFrame({"row_id": dev, "target": y_dev, "probability": obj["oof"]}).to_csv(OUT / f"{name}_oof.csv", index=False)
        pd.DataFrame({"row_id": audit, "target": y_audit, "probability": obj["audit"]}).to_csv(OUT / f"{name}_audit.csv", index=False)
        pd.DataFrame({"Id": pd.read_csv(ROOT / "data" / "raw" / "submission.csv")["Id"], "probability": obj["test"]}).to_csv(OUT / f"{name}_test.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values(["nested_f1", "dev_oof_f1"], ascending=False)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    result = {
        "version": config["version"], "completed_utc": now(), "feature_design": {
            "base": "v2 survey representation", "base_features": 64,
            "encoded_columns": config["encoded_columns"], "added_features": 2 * len(config["encoded_columns"]),
            "final_features": 64 + 2 * len(config["encoded_columns"]),
            "policy": "Fold-train labels only for TE/WOE; native categoricals retained; no one-hot/get_dummies; no train+test encoder fit."
        },
        "references": {"v2": refs["v2"]["result"], "v5": refs["v5"]["result"]},
        "candidates": {name: {k: v for k, v in obj.items() if k not in {"oof", "audit", "test"}} for name, obj in candidates.items()},
        "comparison": rows, "integrity": verify_inputs(), "submitted": False,
        "limitations": [
            "Thresholds are selected on development OOF; nested threshold scores are the stronger F1 evidence.",
            "The audit set is historically reused and remains diagnostic only.",
            "v5 is itself selected from many AutoGluon candidates, so comparisons to v5 inherit selection optimism.",
            "No final 42,154-label refit and no Kaggle submission occur in this development run."
        ]
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": config["version"], "id": "compact_tewoe_xgb_lgb", "parent": "v2_survey",
            "timestamp_utc": now(), "hypothesis": "Compact fold-local TE+WOE on selected high-value categoricals makes XGB/LGBM stronger and more complementary without one-hot feature explosion.",
            "validation": "Exact frozen grouped 5-fold development OOF; fold-local supervised encoders; nested thresholds; reused audit diagnostic; predeclared equal ensembles.",
            "result": rows, "artifact": "artifacts/v10_tewoe", "submitted": False
        }, allow_nan=False) + "\n")
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps({"completed": True, "results": "artifacts/v10_tewoe/results.json", "submitted": False}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v10_tewoe.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = read_json(args.config)
    if args.phase == "check": check(config)
    elif args.phase == "smoke": smoke(config)
    else: run(config, args.resume)


if __name__ == "__main__":
    main()
