"""v4 XGBoost representation recovery using fold-fitted one-hot categoricals.

This version isolates representation from model tuning: it keeps the v3 depth-4
XGBoost hyperparameters fixed and changes only categorical encoding. It never
submits to Kaggle and never modifies v1/v2/v3 artifacts.
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from baseline import metrics, select_threshold
from v2_features import build_features, categorical_columns


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v4"
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
        raise ValueError("Unexpected v2 reference candidate")
    return {
        "v1": v1,
        "v2_candidate": v2_result["selection"]["candidate"],
        "v2_dev_f1": v2_result["candidates"]["cat_survey"]["dev_oof_tuned_f1"],
    }


def initialize(config: dict) -> None:
    if OUT.exists():
        raise FileExistsError("v4 already exists; use --resume after inspecting it")
    verify_inputs()
    OUT.mkdir(parents=True)
    shutil.copy2(PARENT / "split_manifest.csv", OUT / "split_manifest.csv")
    write_json(OUT / "config.json", config)
    source = OUT / "source"
    source.mkdir()
    for name in ["v4_pipeline.py", "v2_features.py", "baseline.py"]:
        shutil.copy2(ROOT / "src" / name, source / name)
    write_json(
        OUT / "provenance.json",
        {
            "created_utc": now(),
            "code_sha256": {
                name: sha(source / name)
                for name in ["v4_pipeline.py", "v2_features.py", "baseline.py"]
            },
            "config_sha256": sha(ROOT / "configs" / "v4.json"),
            "split_sha256": sha(OUT / "split_manifest.csv"),
            "versions": {
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                ]
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
            raise ValueError(f"Source changed since v4 initialization: {name}")
    if sha(ROOT / "configs" / "v4.json") != provenance["config_sha256"]:
        raise ValueError("v4 config changed since initialization")
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


def make_preprocessor(frame: pd.DataFrame, encoder_spec: dict) -> ColumnTransformer:
    categorical = categorical_columns(frame)
    numeric = [c for c in frame.columns if c not in categorical]
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=encoder_spec["min_frequency"],
        sparse_output=True,
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("categorical", encoder, categorical),
            ("numeric", "passthrough", numeric),
        ],
        sparse_threshold=1.0,
    )


def make_model(config: dict, seed: int, smoke: bool = False) -> XGBClassifier:
    p = config["xgboost"]
    return XGBClassifier(
        objective="binary:logistic",
        tree_method=p["tree_method"],
        n_estimators=40 if smoke else int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        max_depth=int(p["max_depth"]),
        min_child_weight=float(p["min_child_weight"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        reg_lambda=float(p["reg_lambda"]),
        reg_alpha=float(p["reg_alpha"]),
        gamma=float(p["gamma"]),
        eval_metric=p["eval_metric"],
        early_stopping_rounds=10 if smoke else int(p["early_stopping_rounds"]),
        random_state=seed,
        n_jobs=int(config["threads"]),
        verbosity=0,
    )


def key(encoder_spec: dict, feature_mode: str) -> str:
    return f'xgb_{encoder_spec["id"]}_{feature_mode}'


def train_fold(
    encoder_spec: dict,
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
    candidate = key(encoder_spec, feature_mode)
    directory = OUT / "candidates" / candidate
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f"fold{fold}_metrics.json"
    if receipt.exists():
        saved = read_json(receipt)
        for name, digest in saved["artifact_sha256"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Fold artifact changed: {directory / name}")
        print(f"REUSE {candidate} fold={fold}", flush=True)
        return
    if (directory / f"fold{fold}.joblib").exists():
        raise FileExistsError(
            f"Partial fold exists without receipt: {directory / f'fold{fold}.joblib'}"
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
    preprocessor = make_preprocessor(features.iloc[train_ids], encoder_spec)
    train_x = preprocessor.fit_transform(features.iloc[train_ids])
    valid_x = preprocessor.transform(features.iloc[valid_ids])
    model = make_model(config, config["seed"] + fold)
    model.fit(train_x, y[train_ids], eval_set=[(valid_x, y[valid_ids])], verbose=False)

    p = model.predict_proba(valid_x)[:, 1]
    audit_p = model.predict_proba(preprocessor.transform(features.iloc[audit]))[:, 1]
    test_p = model.predict_proba(preprocessor.transform(test_features))[:, 1]
    if not all(np.isfinite(v).all() for v in [p, audit_p, test_p]):
        raise ValueError("Non-finite probabilities")

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "encoder_spec": encoder_spec,
            "feature_mode": feature_mode,
        },
        directory / f"fold{fold}.joblib",
        compress=3,
    )
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
            "candidate": candidate,
            "feature_mode": feature_mode,
            "encoder": encoder_spec["id"],
            "input_features": len(features.columns),
            "encoded_features": int(train_x.shape[1]),
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
        f'{candidate} fold={fold + 1}/5 F1@0.31={result["f1"]:.6f} '
        f'AUC={result["roc_auc"]:.6f} cols={result["encoded_features"]} '
        f'best_iter={result["best_iteration"]} seconds={result["seconds"]:.1f}',
        flush=True,
    )


def summarize(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray) -> dict:
    threshold, score = select_threshold(y, p)
    fold_f1 = [
        float(f1_score(y[fold_ids == f], p[fold_ids == f] >= threshold))
        for f in sorted(set(fold_ids))
    ]
    return {
        "threshold": threshold,
        "dev_oof_tuned_f1": score,
        "dev_metrics": metrics(y, p, threshold),
        "fold_f1": fold_f1,
        "fold_f1_std": float(np.std(fold_f1, ddof=1)),
        "threshold_neighborhood_f1": {
            f"{z:.3f}": float(f1_score(y, p >= z))
            for z in [
                threshold - 0.01,
                threshold - 0.005,
                threshold,
                threshold + 0.005,
                threshold + 0.01,
            ]
        },
    }


def references(y: np.ndarray, manifest: pd.DataFrame, dev: np.ndarray, audit: np.ndarray):
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
    v2_audit_frame = pd.read_csv(V2 / "selected_audit.csv").set_index("row_id").loc[audit]
    if not np.array_equal(v2_oof_frame.target.to_numpy(), y[dev]):
        raise ValueError("v2 OOF target alignment failure")
    return {
        "catboost_v1": {
            "oof": v1_oof,
            "audit": v1_audit,
            "result": summarize(y[dev], v1_oof, fold_ids),
            "components": {"catboost_v1": 1.0},
        },
        "catboost_v2": {
            "oof": v2_oof_frame.probability.to_numpy(),
            "audit": v2_audit_frame.candidate_probability.to_numpy(),
            "result": summarize(
                y[dev], v2_oof_frame.probability.to_numpy(), fold_ids
            ),
            "components": {"catboost_v2": 1.0},
        },
    }


def assemble(
    encoder_spec: dict,
    feature_mode: str,
    y: np.ndarray,
    manifest: pd.DataFrame,
    dev: np.ndarray,
) -> dict:
    candidate = key(encoder_spec, feature_mode)
    directory = OUT / "candidates" / candidate
    parts = [pd.read_csv(directory / f"fold{f}_oof.csv") for f in range(5)]
    oof = pd.concat(parts).set_index("row_id").loc[dev].reset_index()
    if not np.array_equal(oof.target.to_numpy(), y[dev]) or oof.probability.isna().any():
        raise ValueError(f"OOF alignment failure: {candidate}")
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
    result = summarize(
        y[dev], oof.probability.to_numpy(), manifest.loc[dev, "dev_fold"].to_numpy()
    )
    result.update(
        {
            "candidate": candidate,
            "feature_mode": feature_mode,
            "encoder": encoder_spec,
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
        "components": {candidate: 1.0},
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
        return np.column_stack(
            [
                np.bincount(inverse, weights=w, minlength=n)
                for w in [
                    (y == 1) & pred,
                    (y == 0) & pred,
                    (y == 1) & ~pred,
                ]
            ]
        )

    base, cand = counts(parent_pred), counts(candidate_pred)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ix = rng.integers(0, n, size=n)
        b, c = base[ix].sum(axis=0), cand[ix].sum(axis=0)
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


def smoke(config: dict) -> None:
    verify_inputs()
    x, _, y, manifest, dev, _ = load_data()
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    train_ids = dev[fold_ids != 0][:6000]
    valid_ids = dev[fold_ids == 0][:1500]
    encoder_spec = config["encoders"][0]
    features = build_features(x, "raw")
    preprocessor = make_preprocessor(features.iloc[train_ids], encoder_spec)
    train_x = preprocessor.fit_transform(features.iloc[train_ids])
    valid_x = preprocessor.transform(features.iloc[valid_ids])
    model = make_model(config, config["seed"], smoke=True)
    model.fit(train_x, y[train_ids], eval_set=[(valid_x, y[valid_ids])], verbose=False)
    p = model.predict_proba(valid_x)[:, 1]
    print(
        json.dumps(
            {
                "smoke_passed": bool(np.isfinite(p).all()),
                "train_rows": len(train_ids),
                "valid_rows": len(valid_ids),
                "encoded_features": int(train_x.shape[1]),
                "best_iteration": int(model.best_iteration),
                "auc": float(roc_auc_score(y[valid_ids], p)),
            },
            indent=2,
        )
    )


def develop(config: dict, resume: bool) -> None:
    if resume:
        verify_resume(config)
    else:
        initialize(config)
    x, test, y, manifest, dev, audit = load_data()
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy()
    candidates = references(y, manifest, dev, audit)
    v1, v2 = candidates["catboost_v1"], candidates["catboost_v2"]
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
        for encoder_spec in config["encoders"]:
            candidate = key(encoder_spec, feature_mode)
            for fold in config["screen_folds"]:
                train_fold(
                    encoder_spec,
                    feature_mode,
                    config,
                    fold,
                    x,
                    test,
                    y,
                    manifest,
                    dev,
                    audit,
                )
            directory = OUT / "candidates" / candidate
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
                np.corrcoef(v1["oof"][screen_mask], part.probability.to_numpy())[0, 1]
            )
            passed = (
                score >= ref_f1 - config["screen_max_f1_drop"]
                and auc >= ref_auc - config["screen_max_auc_drop"]
            )
            screen_results[candidate] = {
                "feature_mode": feature_mode,
                "encoder_id": encoder_spec["id"],
                "f1": score,
                "threshold": threshold,
                "auc": auc,
                "correlation_v1": corr,
                "f1_delta_vs_parent_same_rows": score - ref_f1,
                "passed": bool(passed),
            }
            print(
                f"SCREEN {candidate} F1={score:.6f} delta={score-ref_f1:+.6f} "
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
            name
            for name, result in screen_results.items()
            if result["feature_mode"] == feature_mode and result["passed"]
        ]
        eligible.sort(key=lambda name: screen_results[name]["f1"], reverse=True)
        promoted.extend(eligible[: int(config["full_candidates_per_feature_mode"])])

    if not promoted:
        write_json(
            OUT / "results.json",
            {
                "version": "v4",
                "completed_utc": now(),
                "screening": screen_results,
                "promoted": [],
                "decision": "screen_rejected_all",
                "submitted": False,
            },
        )
        print("No v4 candidate passed screening.", flush=True)
        return

    lookup = {
        key(encoder_spec, mode): (encoder_spec, mode)
        for mode in config["feature_modes"]
        for encoder_spec in config["encoders"]
    }
    for candidate in promoted:
        encoder_spec, feature_mode = lookup[candidate]
        for fold in range(5):
            if fold not in config["screen_folds"]:
                train_fold(
                    encoder_spec,
                    feature_mode,
                    config,
                    fold,
                    x,
                    test,
                    y,
                    manifest,
                    dev,
                    audit,
                )
        candidates[candidate] = assemble(
            encoder_spec, feature_mode, y, manifest, dev
        )
        print(
            f'DEV COMPLETE {candidate} '
            f'F1={candidates[candidate]["result"]["dev_oof_tuned_f1"]:.6f}',
            flush=True,
        )

    for xgb_name in promoted:
        if xgb_name not in candidates:
            continue
        for weight in [float(w) for w in config["blend_xgb_weights"]]:
            name = f"blend_catboost_v2_{xgb_name}_xgb{int(round(weight*100)):02d}"
            p = (1.0 - weight) * v2["oof"] + weight * candidates[xgb_name]["oof"]
            ap = (1.0 - weight) * v2["audit"] + weight * candidates[xgb_name]["audit"]
            candidates[name] = {
                "oof": p,
                "audit": ap,
                "test": None,
                "result": summarize(y[dev], p, fold_ids),
                "components": {"catboost_v2": 1.0 - weight, xgb_name: weight},
            }

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
        result["oof_correlation_v1"] = float(np.corrcoef(v1["oof"], obj["oof"])[0, 1])
        result["oof_correlation_v2"] = float(np.corrcoef(v2["oof"], obj["oof"])[0, 1])
        deltas = np.asarray(result["fold_f1"]) - np.asarray(v2["result"]["fold_f1"])
        result["paired_fold_deltas_vs_v2"] = deltas.tolist()
        result["positive_folds_vs_v2"] = int((deltas > 0).sum())
        rows.append(
            {
                "candidate": name,
                "dev_oof_tuned_f1": result["dev_oof_tuned_f1"],
                "threshold": result["threshold"],
                "dev_delta_vs_v1": result["dev_delta_vs_v1"],
                "dev_delta_vs_v2": result["dev_delta_vs_v2"],
                "positive_folds_vs_v2": result["positive_folds_vs_v2"],
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
        "audit_reuse_warning": "The holdout was previously inspected in v1/v2/v3 and is diagnostic only.",
    }
    write_json(OUT / "frozen_selection.json", selection)
    winner = candidates[chosen]
    audit_score = metrics(y[audit], winner["audit"], selection["threshold"])
    v1_audit = metrics(y[audit], v1["audit"], v1["result"]["threshold"])
    audit_delta = audit_score["f1"] - v1_audit["f1"]
    bootstrap = bootstrap_delta(
        y[audit],
        v1["audit"] >= v1["result"]["threshold"],
        winner["audit"] >= selection["threshold"],
        manifest.loc[audit, "group_hash"].to_numpy(),
        int(config["bootstrap_repeats"]),
        20261005,
    )
    dev_gain_v2 = winner["result"]["dev_delta_vs_v2"]
    promising = (
        chosen != "catboost_v2"
        and dev_gain_v2 >= config["minimum_dev_gain_vs_v2"]
        and audit_delta >= -config["maximum_audit_drop_vs_v1"]
        and winner["result"]["positive_folds_vs_v2"] >= 3
    )
    pd.DataFrame(
        {"row_id": dev, "target": y[dev], "probability": winner["oof"]}
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
        "version": "v4",
        "completed_utc": now(),
        "screening": screen_results,
        "promoted": promoted,
        "selection": selection,
        "candidates": {name: obj["result"] for name, obj in candidates.items()},
        "selected_audit": audit_score,
        "v1_audit": v1_audit,
        "audit_delta_vs_v1": audit_delta,
        "audit_paired_group_bootstrap": bootstrap,
        "candidate_promising": bool(promising),
        "integrity": verify_inputs(),
        "submitted": False,
        "limitations": [
            "OOF encoder/model/threshold/blend selection is performed on development OOF and is selection-biased.",
            "The audit holdout was already inspected in v1/v2/v3; it is diagnostic only.",
            "Historical Kaggle XGBoost scores motivated representation recovery but are not used for selection.",
            "No remote submission or final full-data v4 candidate is created by this script.",
        ],
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "version": "v4",
                    "id": chosen,
                    "parent": "baseline_v1",
                    "timestamp_utc": now(),
                    "validation": "Exact v1 grouped folds; fold-fitted one-hot XGBoost; frozen before reused-audit diagnostic",
                    "result": winner["result"],
                    "audit_f1": audit_score["f1"],
                    "audit_delta_vs_v1": audit_delta,
                    "decision": "promising" if promising else "diagnostic",
                    "artifact": "artifacts/v4",
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
                "dev_gain_vs_v2": dev_gain_v2,
                "correlation_v2": winner["result"]["oof_correlation_v2"],
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
    parser.add_argument("phase", choices=["smoke", "develop"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v4.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == "smoke":
        smoke(config)
    else:
        develop(config, args.resume)


if __name__ == "__main__":
    main()
