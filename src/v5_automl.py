"""v5 AutoML scout using AutoGluon on the exact frozen development folds.

The run is deliberately isolated from the project's primary Python environment.
It trains on development rows only, preserves per-model OOF probabilities for
later ensemble work, freezes selection on development OOF, and only then scores
the selected model on the reused audit holdout. It never submits to Kaggle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from v2_features import build_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v5_automl"
PREDICTOR_PATH = OUT / "predictor"
PARENT = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"
V3_XGB = ROOT / "artifacts" / "v3" / "candidates" / "xgb_d4_survey" / "oof.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def select_threshold(y: np.ndarray, p: np.ndarray, config: dict) -> tuple[float, float]:
    grid = np.linspace(float(config["threshold_min"]), float(config["threshold_max"]), int(config["threshold_steps"]))
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(np.int8)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "positive_prediction_rate": float(pred.mean()),
    }


def verify_v1() -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    expected = read_json(PARENT / "metrics.json")
    if sha(PARENT / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen v1 split changed")
    for name, digest in expected["raw_sha256"].items():
        if sha(ROOT / "data" / "raw" / name) != digest:
            raise ValueError(f"Raw data changed: {name}")
    return {"passed": True, "protected_files": len(protected)}


def load_data(config: dict):
    raw = ROOT / "data" / "raw"
    x = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    if len(x) != len(labels) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(x))):
        raise ValueError("Row identity mismatch")
    if list(x.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")
    y = labels[TARGET].to_numpy()
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy()
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy()
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    if sorted(np.unique(fold_ids).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected exact frozen development folds 0..4")
    if set(manifest.loc[dev, "group_hash"]) & set(manifest.loc[audit, "group_hash"]):
        raise ValueError("Duplicate group crosses development/audit boundary")
    for fold in range(5):
        tr = dev[fold_ids != fold]
        va = dev[fold_ids == fold]
        if set(manifest.loc[tr, "group_hash"]) & set(manifest.loc[va, "group_hash"]):
            raise ValueError(f"Duplicate group crosses development fold {fold}")

    train_features = build_features(x.iloc[dev].copy(), config["feature_mode"])
    audit_features = build_features(x.iloc[audit].copy(), config["feature_mode"])
    test_features = build_features(test.copy(), config["feature_mode"])
    train_data = train_features.copy()
    train_data[TARGET] = y[dev]
    train_data["__fold__"] = fold_ids
    return x, test, y, manifest, dev, audit, fold_ids, train_data, audit_features, test_features


def load_reference_oof(dev: np.ndarray) -> dict[str, np.ndarray]:
    refs = {}
    refs["catboost_v1"] = pd.read_csv(PARENT / "catboost_oof.csv").set_index("row_id").loc[dev, "probability"].to_numpy()
    v2_frame = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[dev]
    refs["catboost_v2"] = v2_frame.probability.to_numpy()
    if V3_XGB.is_file():
        v3 = pd.read_csv(V3_XGB).set_index("row_id").loc[dev]
        refs["xgb_v3_survey"] = v3.probability.to_numpy()
    return refs


def normalize_proba(obj, expected_index: np.ndarray | None = None) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        if 1 in obj.columns:
            series = obj[1]
        elif "1" in obj.columns:
            series = obj["1"]
        else:
            series = obj.iloc[:, -1]
    elif isinstance(obj, pd.Series):
        series = obj
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            arr = arr[:, -1]
        return arr.astype(float)
    if expected_index is not None and len(series) == len(expected_index):
        if set(series.index.tolist()) == set(expected_index.tolist()):
            series = series.reindex(expected_index)
    return series.to_numpy(dtype=float)


def check_phase(config: dict) -> None:
    integrity = verify_v1()
    _, _, y, _, dev, audit, fold_ids, train_data, audit_features, test_features = load_data(config)
    payload = {
        "check_passed": True,
        "integrity": integrity,
        "development_rows": int(len(dev)),
        "audit_rows": int(len(audit)),
        "feature_mode": config["feature_mode"],
        "features": int(len(train_data.columns) - 2),
        "fold_counts": {str(f): int(np.sum(fold_ids == f)) for f in sorted(np.unique(fold_ids))},
        "positive_rate_dev": float(y[dev].mean()),
        "audit_features": int(audit_features.shape[1]),
        "test_rows": int(len(test_features)),
        "remote_submission_authorized": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def fit_or_load_predictor(config: dict, resume: bool):
    from autogluon.tabular import TabularPredictor

    if resume:
        predictor_file = PREDICTOR_PATH / "predictor.pkl"
        if not predictor_file.is_file():
            raise FileNotFoundError("Cannot resume: predictor.pkl is absent; preserve the partial directory for inspection.")
        return TabularPredictor.load(PREDICTOR_PATH)

    if OUT.exists():
        raise FileExistsError(f"{OUT} already exists. Do not overwrite an AutoML run.")
    OUT.mkdir(parents=True)
    write_json(OUT / "config.json", config)
    verify_v1()
    _, _, y, _, dev, _, _, train_data, _, _ = load_data(config)
    predictor = TabularPredictor(
        label=TARGET,
        problem_type="binary",
        eval_metric=config["eval_metric"],
        positive_class=1,
        groups="__fold__",
        path=str(PREDICTOR_PATH),
        verbosity=2,
        log_to_file=True,
    )
    started = time.monotonic()
    predictor.fit(
        train_data=train_data,
        presets=config["preset"],
        time_limit=int(config["time_limit_seconds"]),
        num_cpus=int(config["num_cpus"]),
        num_gpus=int(config["num_gpus"]),
        num_stack_levels=int(config["num_stack_levels"]),
        num_bag_sets=int(config["num_bag_sets"]),
        calibrate_decision_threshold=True,
    )
    write_json(
        OUT / "fit_receipt.json",
        {
            "completed_utc": now(),
            "seconds": time.monotonic() - started,
            "development_rows": int(len(dev)),
            "positive_rate": float(y[dev].mean()),
            "autogluon_version": importlib.metadata.version("autogluon.tabular"),
            "predictor_path": str(PREDICTOR_PATH.relative_to(ROOT)).replace("\\", "/"),
            "predictor_best": predictor.model_best,
            "predictor_decision_threshold": float(predictor.decision_threshold),
            "submitted": False,
        },
    )
    return predictor


def run_phase(config: dict, resume: bool) -> None:
    version = importlib.metadata.version("autogluon.tabular")
    if version != config["autogluon_version"]:
        raise RuntimeError(f'Expected autogluon.tabular {config["autogluon_version"]}, got {version}')
    predictor = fit_or_load_predictor(config, resume)
    _, test, y, _, dev, audit, fold_ids, _, audit_features, test_features = load_data(config)
    refs = load_reference_oof(dev)

    predictor.leaderboard(extra_info=True, silent=True).to_csv(OUT / "leaderboard.csv", index=False)
    eligible: dict[str, np.ndarray] = {}
    failures = {}
    rows = []
    for model in predictor.model_names():
        try:
            raw = predictor.predict_proba_oof(model=model, as_multiclass=False)
            p = normalize_proba(raw, expected_index=dev)
            if len(p) != len(dev) or not np.isfinite(p).all():
                raise ValueError("OOF length/nonfinite failure")
            threshold, tuned_f1 = select_threshold(y[dev], p, config)
            fold_f1 = [
                float(f1_score(y[dev][fold_ids == fold], p[fold_ids == fold] >= threshold, zero_division=0))
                for fold in range(5)
            ]
            row = {
                "model": model,
                "dev_oof_tuned_f1": tuned_f1,
                "threshold": threshold,
                "roc_auc": float(roc_auc_score(y[dev], p)),
                "log_loss": float(log_loss(y[dev], p, labels=[0, 1])),
                "fold_f1_std": float(np.std(fold_f1, ddof=1)),
                "fold_f1": json.dumps(fold_f1),
            }
            for ref_name, ref_p in refs.items():
                row[f"corr_{ref_name}"] = float(np.corrcoef(p, ref_p)[0, 1])
            eligible[model] = p
            rows.append(row)
        except Exception as exc:
            failures[model] = f"{type(exc).__name__}: {exc}"
    if not eligible:
        raise RuntimeError("No AutoGluon model exposed valid OOF probabilities")

    scores = pd.DataFrame(rows).sort_values(["dev_oof_tuned_f1", "roc_auc"], ascending=False)
    scores.to_csv(OUT / "model_scores.csv", index=False)
    best_model = str(scores.iloc[0]["model"])
    best_threshold = float(scores.iloc[0]["threshold"])
    selection = {
        "candidate": best_model,
        "threshold": best_threshold,
        "dev_oof_tuned_f1": float(scores.iloc[0]["dev_oof_tuned_f1"]),
        "roc_auc": float(scores.iloc[0]["roc_auc"]),
        "frozen_utc": now(),
        "selection_source": "Development OOF only; exact frozen dev folds; reused audit labels not consulted.",
        "autogluon_internal_best": predictor.model_best,
        "autogluon_decision_threshold": float(predictor.decision_threshold),
    }
    write_json(OUT / "frozen_selection.json", selection)

    oof_wide = pd.DataFrame({"row_id": dev, "target": y[dev]})
    for model, p in eligible.items():
        oof_wide[model] = p
    oof_wide.to_csv(OUT / "oof_probabilities.csv", index=False)
    oof_wide.drop(columns=["row_id", "target"]).corr().to_csv(OUT / "oof_correlation_matrix.csv")

    eligible_models = list(eligible)
    audit_multi = predictor.predict_proba_multi(audit_features, models=eligible_models, as_multiclass=False)
    test_multi = predictor.predict_proba_multi(test_features, models=eligible_models, as_multiclass=False)
    audit_wide = pd.DataFrame({"row_id": audit})
    test_wide = pd.DataFrame({"Id": pd.read_csv(ROOT / "data" / "raw" / "submission.csv")["Id"]})
    for model in eligible_models:
        audit_wide[model] = normalize_proba(audit_multi[model])
        test_wide[model] = normalize_proba(test_multi[model])
    audit_wide.to_csv(OUT / "audit_probabilities.csv", index=False)
    test_wide.to_csv(OUT / "test_probabilities.csv", index=False)

    selected_audit_p = audit_wide[best_model].to_numpy()
    audit_metrics = classification_metrics(y[audit], selected_audit_p, best_threshold)
    v1_threshold, v1_dev_f1 = select_threshold(y[dev], refs["catboost_v1"], config)
    v2_threshold, v2_dev_f1 = select_threshold(y[dev], refs["catboost_v2"], config)
    v1_audit = pd.read_csv(PARENT / "catboost_audit_probabilities.csv").set_index("row_id").loc[audit, "probability"].to_numpy()
    v1_audit_metrics = classification_metrics(y[audit], v1_audit, v1_threshold)

    result = {
        "version": "v5_automl",
        "completed_utc": now(),
        "engine": "AutoGluon",
        "autogluon_version": version,
        "preset": config["preset"],
        "feature_mode": config["feature_mode"],
        "validation": "Exact frozen v1 development folds supplied to AutoGluon as five group IDs; development rows only.",
        "models_with_valid_oof": len(eligible),
        "model_failures": failures,
        "selection": selection,
        "selected_audit": audit_metrics,
        "references": {
            "v1_dev_oof_f1": v1_dev_f1,
            "v1_threshold": v1_threshold,
            "v1_audit_f1": v1_audit_metrics["f1"],
            "v2_dev_oof_f1": v2_dev_f1,
            "v2_threshold": v2_threshold,
            "dev_gain_vs_v1": selection["dev_oof_tuned_f1"] - v1_dev_f1,
            "dev_gain_vs_v2": selection["dev_oof_tuned_f1"] - v2_dev_f1,
            "audit_delta_vs_v1": audit_metrics["f1"] - v1_audit_metrics["f1"],
        },
        "integrity": verify_v1(),
        "submitted": False,
        "limitations": [
            "AutoGluon model/ensemble selection and external threshold selection both use development OOF, so selected OOF F1 is optimistic.",
            "The audit holdout was already inspected in v1-v3 and remains diagnostic only.",
            "The AutoGluon predictor is trained on development rows only. A later final submission would require a separately frozen full-data refit.",
            "No Kaggle submission or leaderboard probing occurs in this run."
        ],
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": "v5_automl",
            "id": best_model,
            "parent": "baseline_v1",
            "timestamp_utc": now(),
            "hypothesis": "AutoML bagging/stacking/weighted ensembles can discover a stronger and/or more diverse development predictor than the manually tuned tree models.",
            "validation": result["validation"],
            "result": {
                "dev_oof_tuned_f1": selection["dev_oof_tuned_f1"],
                "threshold": selection["threshold"],
                "audit_f1": audit_metrics["f1"],
                "dev_gain_vs_v2": result["references"]["dev_gain_vs_v2"],
            },
            "decision": "automl_candidate",
            "artifact": "artifacts/v5_automl",
        }, allow_nan=False) + "\n")
    print(json.dumps({
        "best_automl_model": best_model,
        "dev_oof_f1": selection["dev_oof_tuned_f1"],
        "threshold": selection["threshold"],
        "dev_gain_vs_v2": result["references"]["dev_gain_vs_v2"],
        "audit_f1": audit_metrics["f1"],
        "audit_delta_vs_v1": result["references"]["audit_delta_vs_v1"],
        "models_with_valid_oof": len(eligible),
        "autogluon_internal_best": predictor.model_best,
        "submitted": False,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["check", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v5_automl.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check_phase(config)
    else:
        run_phase(config, args.resume)


if __name__ == "__main__":
    main()
