"""v3 XGBoost family/diversity experiment.

Preserves v1/v2 artifacts, reuses the exact grouped development folds, and never
calls Kaggle or any other remote submission API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from baseline import metrics, select_threshold
from v2_features import build_features, categorical_columns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v3"
PARENT = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"
TARGET = "vacc_h1n1_f"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_v1() -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [
        p
        for p, digest in protected.items()
        if not (ROOT / p).is_file() or sha(ROOT / p) != digest
    ]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    return {"passed": True, "protected_files": len(protected)}


def verify_inputs() -> dict:
    v1 = verify_v1()
    expected = read_json(PARENT / "metrics.json")
    if sha(PARENT / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen v1 split changed")
    for name, digest in expected["raw_sha256"].items():
        if sha(ROOT / "data" / "raw" / name) != digest:
            raise ValueError(f"Raw data changed: {name}")
    v2_result = read_json(V2 / "results.json")
    if v2_result["selection"]["candidate"] != "cat_survey":
        raise ValueError("v2 selected recipe differs from the expected cat_survey reference")
    return {
        "v1": v1,
        "v2_candidate": v2_result["selection"]["candidate"],
        "v2_dev_f1": v2_result["candidates"]["cat_survey"]["dev_oof_tuned_f1"],
    }


def initialize(config: dict) -> None:
    if OUT.exists():
        raise FileExistsError("v3 already exists; use --resume after inspecting receipts")
    verify_inputs()
    OUT.mkdir(parents=True)
    shutil.copy2(PARENT / "split_manifest.csv", OUT / "split_manifest.csv")
    write_json(OUT / "config.json", config)
    source = OUT / "source"
    source.mkdir()
    for name in ["v3_pipeline.py", "v2_features.py", "baseline.py"]:
        shutil.copy2(ROOT / "src" / name, source / name)
    write_json(
        OUT / "provenance.json",
        {
            "created_utc": now(),
            "code_sha256": {
                name: sha(source / name)
                for name in ["v3_pipeline.py", "v2_features.py", "baseline.py"]
            },
            "config_sha256": sha(ROOT / "configs" / "v3.json"),
            "split_sha256": sha(OUT / "split_manifest.csv"),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]
            },
            "submitted": False,
        },
    )


def verify_resume(config: dict) -> None:
    if read_json(OUT / "config.json") != config:
        raise ValueError("Resume config mismatch")
    provenance = read_json(OUT / "provenance.json")
    for name, digest in provenance["code_sha256"].items():
        if sha(ROOT / "src" / name) != digest:
            raise ValueError(f"Source changed since v3 initialization: {name}")
    if sha(ROOT / "configs" / "v3.json") != provenance["config_sha256"]:
        raise ValueError("v3 config changed since initialization")
    verify_inputs()


def load_data():
    raw = ROOT / "data" / "raw"
    x = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    if len(x) != len(labels) or not np.array_equal(
        manifest.row_id.to_numpy(), np.arange(len(x))
    ):
        raise ValueError("Row identity mismatch")
    if list(x.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")
    y = labels[TARGET].to_numpy()
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy()
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy()
    return x, test, y, manifest, dev, audit


def fit_xgb_schema(frame: pd.DataFrame) -> dict:
    cats = categorical_columns(frame)
    levels = {
        c: sorted(frame[c].dropna().astype(str).unique().tolist())
        for c in cats
    }
    return {"columns": list(frame.columns), "categorical": cats, "levels": levels}


def transform_xgb(frame: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if list(frame.columns) != schema["columns"]:
        raise ValueError("Feature schema/order mismatch")
    result = frame.copy()
    cats = set(schema["categorical"])
    for col in result.columns:
        if col in cats:
            values = result[col].where(result[col].notna(), np.nan)
            values = values.map(lambda v: str(v) if pd.notna(v) else np.nan)
            result[col] = pd.Categorical(values, categories=schema["levels"][col])
        else:
            result[col] = pd.to_numeric(result[col], errors="raise").astype(float)
    return result


def candidate_key(spec: dict, feature_mode: str) -> str:
    return f'{spec["id"]}_{feature_mode}'


def model_for(spec: dict, config: dict, seed: int, early_stopping: bool = True, n_estimators: int | None = None):
    common = config["xgboost_common"]
    kwargs = {
        "objective": "binary:logistic",
        "tree_method": common["tree_method"],
        "enable_categorical": True,
        "n_estimators": int(n_estimators or common["n_estimators"]),
        "learning_rate": float(spec["learning_rate"]),
        "max_depth": int(spec["max_depth"]),
        "min_child_weight": float(spec["min_child_weight"]),
        "subsample": float(spec["subsample"]),
        "colsample_bytree": float(spec["colsample_bytree"]),
        "reg_lambda": float(spec["reg_lambda"]),
        "reg_alpha": float(spec["reg_alpha"]),
        "gamma": float(spec["gamma"]),
        "max_cat_to_onehot": int(common["max_cat_to_onehot"]),
        "eval_metric": common["eval_metric"],
        "random_state": seed,
        "n_jobs": int(config["threads"]),
        "verbosity": 0,
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = int(common["early_stopping_rounds"])
    return XGBClassifier(**kwargs)


def train_fold(
    spec: dict,
    feature_mode: str,
    config: dict,
    fold: int,
    x: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    manifest: pd.DataFrame,
    dev: np.ndarray,
    audit: np.ndarray,
) -> None:
    key = candidate_key(spec, feature_mode)
    directory = OUT / "candidates" / key
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f"fold{fold}_metrics.json"
    if receipt.exists():
        saved = read_json(receipt)
        for name, digest in saved["artifact_sha256"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Fold artifact changed: {directory / name}")
        print(f"REUSE {key} fold={fold}", flush=True)
        return
    partial = directory / f"fold{fold}.joblib"
    if partial.exists():
        raise FileExistsError(
            f"Partial fold exists without receipt: {partial}; inspect before retrying"
        )

    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    train_ids = dev[fold_ids != fold]
    valid_ids = dev[fold_ids == fold]
    if set(manifest.loc[train_ids, "group_hash"]) & set(
        manifest.loc[valid_ids, "group_hash"]
    ):
        raise ValueError("Duplicate group crosses development fold")

    started = time.monotonic()
    features = build_features(x, feature_mode)
    test_features = build_features(test, feature_mode)
    schema = fit_xgb_schema(features.iloc[train_ids])
    train_x = transform_xgb(features.iloc[train_ids], schema)
    valid_x = transform_xgb(features.iloc[valid_ids], schema)
    model = model_for(spec, config, config["seed"] + fold)
    model.fit(train_x, y[train_ids], eval_set=[(valid_x, y[valid_ids])], verbose=False)

    p = model.predict_proba(valid_x)[:, 1]
    audit_p = model.predict_proba(transform_xgb(features.iloc[audit], schema))[:, 1]
    test_p = model.predict_proba(transform_xgb(test_features, schema))[:, 1]
    if not all(np.isfinite(v).all() for v in [p, audit_p, test_p]):
        raise ValueError("Non-finite probabilities")

    bundle = {
        "model": model,
        "schema": schema,
        "spec": spec,
        "feature_mode": feature_mode,
    }
    joblib.dump(bundle, partial, compress=3)
    pd.DataFrame(
        {"row_id": valid_ids, "target": y[valid_ids], "probability": p}
    ).to_csv(directory / f"fold{fold}_oof.csv", index=False)
    np.savez_compressed(
        directory / f"fold{fold}_predictions.npz", audit=audit_p, test=test_p
    )
    result = metrics(y[valid_ids], p, 0.31)
    result.update(
        {
            "fold": fold,
            "candidate": key,
            "features": len(features.columns),
            "best_iteration": int(model.best_iteration),
            "best_score": float(model.best_score),
            "seconds": time.monotonic() - started,
            "audit_labels_used": False,
        }
    )
    names = [
        f"fold{fold}.joblib",
        f"fold{fold}_oof.csv",
        f"fold{fold}_predictions.npz",
    ]
    result["artifact_sha256"] = {name: sha(directory / name) for name in names}
    write_json(receipt, result)
    print(
        f'{key} fold={fold + 1}/5 F1@0.31={result["f1"]:.6f} '
        f'AUC={result["roc_auc"]:.6f} best_iter={result["best_iteration"]} '
        f'seconds={result["seconds"]:.1f}',
        flush=True,
    )


def summarize(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray) -> dict:
    threshold, score = select_threshold(y, p)
    fold_f1 = [
        float(f1_score(y[fold_ids == f], p[fold_ids == f] >= threshold))
        for f in sorted(set(fold_ids))
    ]
    nearby = {
        f"{z:.3f}": float(f1_score(y, p >= z))
        for z in [threshold - 0.01, threshold - 0.005, threshold, threshold + 0.005, threshold + 0.01]
    }
    return {
        "threshold": threshold,
        "dev_oof_tuned_f1": score,
        "dev_metrics": metrics(y, p, threshold),
        "fold_f1": fold_f1,
        "fold_f1_std": float(np.std(fold_f1, ddof=1)),
        "threshold_neighborhood_f1": nearby,
    }


def load_reference_predictions(y: np.ndarray, manifest: pd.DataFrame, dev: np.ndarray, audit: np.ndarray):
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    v1_oof = (
        pd.read_csv(PARENT / "catboost_oof.csv")
        .set_index("row_id")
        .loc[dev, "probability"]
        .to_numpy()
    )
    v1_audit = (
        pd.read_csv(PARENT / "catboost_audit_probabilities.csv")
        .set_index("row_id")
        .loc[audit, "probability"]
        .to_numpy()
    )
    v2_oof_frame = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[dev]
    if not np.array_equal(v2_oof_frame.target.to_numpy(), y[dev]):
        raise ValueError("v2 OOF target alignment failure")
    v2_audit_frame = pd.read_csv(V2 / "selected_audit.csv").set_index("row_id").loc[audit]
    refs = {
        "catboost_v1": {
            "oof": v1_oof,
            "audit": v1_audit,
            "result": summarize(y[dev], v1_oof, fold_ids),
            "components": {"catboost_v1": 1.0},
        },
        "catboost_v2": {
            "oof": v2_oof_frame.probability.to_numpy(),
            "audit": v2_audit_frame.candidate_probability.to_numpy(),
            "result": summarize(y[dev], v2_oof_frame.probability.to_numpy(), fold_ids),
            "components": {"catboost_v2": 1.0},
        },
    }
    return refs


def assemble_xgb(
    spec: dict,
    feature_mode: str,
    y: np.ndarray,
    manifest: pd.DataFrame,
    dev: np.ndarray,
) -> dict:
    key = candidate_key(spec, feature_mode)
    directory = OUT / "candidates" / key
    parts = [pd.read_csv(directory / f"fold{f}_oof.csv") for f in range(5)]
    oof = pd.concat(parts).set_index("row_id").loc[dev].reset_index()
    if not np.array_equal(oof.target.to_numpy(), y[dev]) or oof.probability.isna().any():
        raise ValueError(f"OOF alignment failure: {key}")
    audit_p, test_p, best_iterations, seconds = [], [], [], []
    for fold in range(5):
        with np.load(directory / f"fold{fold}_predictions.npz") as data:
            audit_p.append(data["audit"])
            test_p.append(data["test"])
        receipt = read_json(directory / f"fold{fold}_metrics.json")
        best_iterations.append(int(receipt["best_iteration"]))
        seconds.append(float(receipt["seconds"]))
    ap = np.mean(audit_p, axis=0)
    tp = np.mean(test_p, axis=0)
    oof.to_csv(directory / "oof.csv", index=False)
    np.savez_compressed(directory / "ensemble_predictions.npz", audit=ap, test=tp)
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    result = summarize(y[dev], oof.probability.to_numpy(), fold_ids)
    result.update(
        {
            "candidate": key,
            "spec": spec,
            "feature_mode": feature_mode,
            "training_seconds": float(sum(seconds)),
            "best_iterations": best_iterations,
            "median_best_iteration": int(np.median(best_iterations)),
        }
    )
    write_json(directory / "development_metrics.json", result)
    return {
        "oof": oof.probability.to_numpy(),
        "audit": ap,
        "test": tp,
        "result": result,
        "components": {key: 1.0},
    }


def bootstrap_delta(
    y: np.ndarray,
    parent_pred: np.ndarray,
    candidate_pred: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    _, inverse = np.unique(groups, return_inverse=True)
    n = int(inverse.max()) + 1

    def counts(pred: np.ndarray) -> np.ndarray:
        weights = [
            (y == 1) & pred,
            (y == 0) & pred,
            (y == 1) & ~pred,
        ]
        return np.column_stack(
            [np.bincount(inverse, weights=w, minlength=n) for w in weights]
        )

    base = counts(parent_pred)
    cand = counts(candidate_pred)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ix = rng.integers(0, n, size=n)
        b = base[ix].sum(axis=0)
        c = cand[ix].sum(axis=0)
        bf = 2 * b[0] / max(1.0, 2 * b[0] + b[1] + b[2])
        cf = 2 * c[0] / max(1.0, 2 * c[0] + c[1] + c[2])
        deltas.append(cf - bf)
    values = np.asarray(deltas)
    return {
        "repeats": repeats,
        "groups": n,
        "delta_95pct_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "positive_fraction": float(np.mean(values > 0)),
        "limitation": "Conditional on fixed fitted models, split and selected thresholds.",
    }


def develop(config: dict, resume: bool) -> None:
    if resume:
        verify_resume(config)
    else:
        initialize(config)
    x, test, y, manifest, dev, audit = load_data()
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    candidates = load_reference_predictions(y, manifest, dev, audit)
    v1 = candidates["catboost_v1"]
    v2 = candidates["catboost_v2"]
    print(
        f'REFERENCE v1={v1["result"]["dev_oof_tuned_f1"]:.6f} '
        f'v2={v2["result"]["dev_oof_tuned_f1"]:.6f}; exact grouped folds',
        flush=True,
    )

    screen_mask = np.isin(fold_ids, config["screen_folds"])
    ref_t, ref_f1 = select_threshold(y[dev][screen_mask], v1["oof"][screen_mask])
    ref_auc = float(roc_auc_score(y[dev][screen_mask], v1["oof"][screen_mask]))
    screen_results = {}

    for feature_mode in config["feature_modes"]:
        for spec in config["candidates"]:
            key = candidate_key(spec, feature_mode)
            for fold in config["screen_folds"]:
                train_fold(spec, feature_mode, config, fold, x, test, y, manifest, dev, audit)
            directory = OUT / "candidates" / key
            part = (
                pd.concat(
                    [
                        pd.read_csv(directory / f"fold{f}_oof.csv")
                        for f in config["screen_folds"]
                    ]
                )
                .set_index("row_id")
                .loc[dev[screen_mask]]
            )
            threshold, score = select_threshold(
                part.target.to_numpy(), part.probability.to_numpy()
            )
            auc = float(roc_auc_score(part.target, part.probability))
            corr = float(
                np.corrcoef(
                    v1["oof"][screen_mask], part.probability.to_numpy()
                )[0, 1]
            )
            passed = (
                score >= ref_f1 - config["screen_max_f1_drop"]
                and auc >= ref_auc - config["screen_max_auc_drop"]
            )
            screen_results[key] = {
                "feature_mode": feature_mode,
                "spec_id": spec["id"],
                "f1": score,
                "threshold": threshold,
                "auc": auc,
                "correlation_v1": corr,
                "f1_delta_vs_parent_same_rows": score - ref_f1,
                "passed": bool(passed),
            }
            print(
                f"SCREEN {key} F1={score:.6f} delta={score-ref_f1:+.6f} "
                f"AUC={auc:.6f} corr={corr:.5f} continue={passed}",
                flush=True,
            )
            write_json(
                OUT / "screening.json",
                {
                    "reference_f1": ref_f1,
                    "reference_threshold": ref_t,
                    "reference_auc": ref_auc,
                    "folds": config["screen_folds"],
                    "candidates": screen_results,
                    "limitation": "Two-fold screening is noisy and is not promotion evidence.",
                },
            )

    promoted = []
    for feature_mode in config["feature_modes"]:
        eligible = [
            key
            for key, result in screen_results.items()
            if result["feature_mode"] == feature_mode and result["passed"]
        ]
        eligible.sort(key=lambda key: screen_results[key]["f1"], reverse=True)
        promoted.extend(eligible[: int(config["full_candidates_per_feature_mode"])])

    if not promoted:
        write_json(
            OUT / "results.json",
            {
                "version": "v3",
                "completed_utc": now(),
                "screening": screen_results,
                "promoted": [],
                "decision": "screen_rejected_all",
                "submitted": False,
            },
        )
        print("No XGBoost candidate passed screening; stop before full CV.", flush=True)
        return

    spec_lookup = {
        candidate_key(spec, mode): (spec, mode)
        for mode in config["feature_modes"]
        for spec in config["candidates"]
    }
    for key in promoted:
        spec, feature_mode = spec_lookup[key]
        for fold in range(5):
            if fold not in config["screen_folds"]:
                train_fold(spec, feature_mode, config, fold, x, test, y, manifest, dev, audit)
        candidates[key] = assemble_xgb(spec, feature_mode, y, manifest, dev)
        print(
            f'DEV COMPLETE {key} F1={candidates[key]["result"]["dev_oof_tuned_f1"]:.6f}',
            flush=True,
        )

    xgb_names = [name for name in promoted if name in candidates]
    blend_weights = [float(w) for w in config["blend_xgb_weights"]]
    for xgb_name in xgb_names:
        for base_name in ["catboost_v1", "catboost_v2"]:
            base = candidates[base_name]
            for wx in blend_weights:
                name = f"blend_{base_name}_{xgb_name}_xgb{int(round(wx * 100)):02d}"
                p = (1.0 - wx) * base["oof"] + wx * candidates[xgb_name]["oof"]
                ap = (1.0 - wx) * base["audit"] + wx * candidates[xgb_name]["audit"]
                tp = (1.0 - wx) * np.zeros_like(candidates[xgb_name]["test"]) + wx * candidates[xgb_name]["test"]
                result = summarize(y[dev], p, fold_ids)
                candidates[name] = {
                    "oof": p,
                    "audit": ap,
                    "test": None,
                    "result": result,
                    "components": {base_name: 1.0 - wx, xgb_name: wx},
                }

    best_cat_name = max(
        ["catboost_v1", "catboost_v2"],
        key=lambda name: candidates[name]["result"]["dev_oof_tuned_f1"],
    )
    chosen = max(
        candidates,
        key=lambda name: candidates[name]["result"]["dev_oof_tuned_f1"],
    )
    rows = []
    for name, obj in candidates.items():
        result = obj["result"]
        result["candidate"] = name
        result["components"] = obj["components"]
        result["dev_delta_vs_v1"] = (
            result["dev_oof_tuned_f1"] - v1["result"]["dev_oof_tuned_f1"]
        )
        result["dev_delta_vs_v2"] = (
            result["dev_oof_tuned_f1"] - v2["result"]["dev_oof_tuned_f1"]
        )
        result["oof_correlation_v1"] = float(
            np.corrcoef(v1["oof"], obj["oof"])[0, 1]
        )
        result["oof_correlation_v2"] = float(
            np.corrcoef(v2["oof"], obj["oof"])[0, 1]
        )
        deltas = np.asarray(result["fold_f1"]) - np.asarray(v1["result"]["fold_f1"])
        result["paired_fold_deltas_vs_v1"] = deltas.tolist()
        result["positive_folds_vs_v1"] = int((deltas > 0).sum())
        rows.append(
            {
                "candidate": name,
                "dev_oof_tuned_f1": result["dev_oof_tuned_f1"],
                "threshold": result["threshold"],
                "dev_delta_vs_v1": result["dev_delta_vs_v1"],
                "dev_delta_vs_v2": result["dev_delta_vs_v2"],
                "positive_folds_vs_v1": result["positive_folds_vs_v1"],
                "oof_correlation_v1": result["oof_correlation_v1"],
                "oof_correlation_v2": result["oof_correlation_v2"],
            }
        )
    pd.DataFrame(rows).sort_values("dev_oof_tuned_f1", ascending=False).to_csv(
        OUT / "comparison.csv", index=False
    )

    selection = {
        "candidate": chosen,
        "components": candidates[chosen]["components"],
        "threshold": candidates[chosen]["result"]["threshold"],
        "frozen_utc": now(),
        "selection_source": "Development OOF only; reused audit untouched until after this freeze.",
        "audit_reuse_warning": "The holdout was previously inspected in v1/v2 and is diagnostic only.",
    }
    write_json(OUT / "frozen_selection.json", selection)

    winner = candidates[chosen]
    audit_score = metrics(y[audit], winner["audit"], selection["threshold"])
    v1_audit = metrics(
        y[audit], v1["audit"], v1["result"]["threshold"]
    )
    audit_delta = audit_score["f1"] - v1_audit["f1"]
    bootstrap = bootstrap_delta(
        y[audit],
        v1["audit"] >= v1["result"]["threshold"],
        winner["audit"] >= selection["threshold"],
        manifest.loc[audit, "group_hash"].to_numpy(),
        int(config["bootstrap_repeats"]),
        20261005,
    )
    best_cat_f1 = candidates[best_cat_name]["result"]["dev_oof_tuned_f1"]
    dev_gain_best_cat = winner["result"]["dev_oof_tuned_f1"] - best_cat_f1
    promising = (
        chosen not in {"catboost_v1", "catboost_v2"}
        and dev_gain_best_cat >= config["minimum_dev_gain_vs_best_cat"]
        and audit_delta >= -config["maximum_audit_drop_vs_v1"]
        and winner["result"]["positive_folds_vs_v1"] >= 3
    )
    pd.DataFrame(
        {
            "row_id": dev,
            "target": y[dev],
            "probability": winner["oof"],
        }
    ).to_csv(OUT / "selected_oof.csv", index=False)
    pd.DataFrame(
        {
            "row_id": audit,
            "target": y[audit],
            "candidate_probability": winner["audit"],
            "v1_probability": v1["audit"],
        }
    ).to_csv(OUT / "selected_audit.csv", index=False)
    result = {
        "version": "v3",
        "completed_utc": now(),
        "screening": screen_results,
        "promoted": promoted,
        "selection": selection,
        "best_cat_reference": best_cat_name,
        "dev_gain_vs_best_cat": dev_gain_best_cat,
        "candidates": {name: obj["result"] for name, obj in candidates.items()},
        "selected_audit": audit_score,
        "v1_audit": v1_audit,
        "audit_delta_vs_v1": audit_delta,
        "audit_paired_group_bootstrap": bootstrap,
        "candidate_promising": bool(promising),
        "integrity": verify_inputs(),
        "submitted": False,
        "limitations": [
            "OOF model/threshold/blend selection is performed on the same development OOF and is selection-biased.",
            "The audit holdout was already inspected in v1 and v2; it is diagnostic, not independent confirmation.",
            "Historical Kaggle XGBoost scores motivated the family choice but are not used for local selection.",
            "No remote submission occurs in this script.",
        ],
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        for name in promoted:
            if name not in candidates:
                continue
            handle.write(
                json.dumps(
                    {
                        "version": "v3",
                        "id": name,
                        "parent": "baseline_v1",
                        "timestamp_utc": now(),
                        "validation": "Exact v1 grouped development folds; XGBoost early stopping per fold; OOF F1 threshold selection",
                        "result": candidates[name]["result"],
                        "decision": "development_candidate",
                        "artifact": "artifacts/v3",
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "version": "v3",
                    "id": chosen,
                    "parent": "baseline_v1",
                    "timestamp_utc": now(),
                    "validation": "Exact v1 grouped development folds; frozen before reused-audit diagnostic",
                    "result": winner["result"],
                    "audit_f1": audit_score["f1"],
                    "audit_delta_vs_v1": audit_delta,
                    "decision": "promising" if promising else "diagnostic",
                    "artifact": "artifacts/v3",
                },
                allow_nan=False,
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "chosen": chosen,
                "dev_f1": winner["result"]["dev_oof_tuned_f1"],
                "dev_gain_vs_v1": winner["result"]["dev_delta_vs_v1"],
                "dev_gain_vs_v2": winner["result"]["dev_delta_vs_v2"],
                "correlation_v1": winner["result"]["oof_correlation_v1"],
                "audit_f1": audit_score["f1"],
                "audit_delta_vs_v1": audit_delta,
                "audit_delta_95pct": bootstrap["delta_95pct_interval"],
                "promising": promising,
                "submitted": False,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "v3.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    develop(config, args.resume)


if __name__ == "__main__":
    main()
