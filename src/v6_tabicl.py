"""v6 TabICLv2 scout on the exact frozen grouped development folds.

This pipeline is deliberately isolated from the primary project environment.
It uses the row-local v2 survey features, trains/evaluates TabICLv2 on the same
five frozen development folds, saves resumable per-fold probabilities, freezes
the F1 threshold on development OOF, then scores the reused audit diagnostically.
It never submits to Kaggle.
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
OUT = ROOT / "artifacts" / "v6_tabicl"
FOLDS_OUT = OUT / "folds"
PARENT = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2"
V3_XGB = ROOT / "artifacts" / "v3" / "candidates" / "xgb_d4_survey" / "oof.csv"
V5_OOF = ROOT / "artifacts" / "v5_automl" / "oof_probabilities.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    y = labels[TARGET].to_numpy(dtype=int)
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    fold_ids = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    if sorted(np.unique(fold_ids).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected exact frozen development folds 0..4")
    if set(manifest.loc[dev, "group_hash"]) & set(manifest.loc[audit, "group_hash"]):
        raise ValueError("Duplicate group crosses development/audit boundary")
    for fold in range(5):
        tr_ids = dev[fold_ids != fold]
        va_ids = dev[fold_ids == fold]
        if set(manifest.loc[tr_ids, "group_hash"]) & set(manifest.loc[va_ids, "group_hash"]):
            raise ValueError(f"Duplicate group crosses development fold {fold}")

    dev_x = build_features(x.iloc[dev].copy(), config["feature_mode"])
    audit_x = build_features(x.iloc[audit].copy(), config["feature_mode"])
    test_x = build_features(test.copy(), config["feature_mode"])
    if list(dev_x.columns) != list(audit_x.columns) or list(dev_x.columns) != list(test_x.columns):
        raise ValueError("Feature schema mismatch")
    return x, test, y, manifest, dev, audit, fold_ids, dev_x, audit_x, test_x


def positive_probability(proba) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] == 2:
        arr = arr[:, 1]
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("Invalid probability output")
    return arr


def make_model(config: dict, n_estimators: int, random_state: int):
    from tabicl import TabICLClassifier

    return TabICLClassifier(
        n_estimators=n_estimators,
        batch_size=int(config["batch_size"]),
        kv_cache=bool(config["kv_cache"]),
        checkpoint_version=config["checkpoint_version"],
        device=config["device"],
        use_amp=config["use_amp"],
        use_fa3=bool(config["use_fa3"]),
        offload_mode=config["offload_mode"],
        random_state=random_state,
        n_jobs=int(config["n_jobs"]),
        verbose=True,
    )


def environment_info() -> dict:
    import torch

    return {
        "tabicl": importlib.metadata.version("tabicl"),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_phase(config: dict) -> None:
    _, _, y, _, dev, audit, fold_ids, dev_x, audit_x, test_x = load_data(config)
    info = environment_info()
    if config["device"] == "cuda" and not info["cuda_available"]:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if info["tabicl"] != config["tabicl_version"]:
        raise RuntimeError(f"Expected tabicl {config['tabicl_version']}, got {info['tabicl']}")
    print(json.dumps({
        "check_passed": True,
        "integrity": verify_v1(),
        "environment": info,
        "development_rows": len(dev),
        "audit_rows": len(audit),
        "features": dev_x.shape[1],
        "feature_mode": config["feature_mode"],
        "fold_counts": {str(i): int((fold_ids == i).sum()) for i in range(5)},
        "positive_rate_dev": float(y[dev].mean()),
        "audit_features": audit_x.shape[1],
        "test_rows": len(test_x),
        "remote_submission_authorized": bool(config["remote_submission_authorized"]),
    }, indent=2, ensure_ascii=False))


def smoke_phase(config: dict) -> None:
    import torch

    _, _, y, _, dev, _, fold_ids, dev_x, _, _ = load_data(config)
    tr = np.flatnonzero(fold_ids != 0)[: int(config["smoke_train_rows"])]
    va = np.flatnonzero(fold_ids == 0)[: int(config["smoke_valid_rows"])]
    model = make_model(config, int(config["smoke_n_estimators"]), int(config["random_state"]))
    start = time.perf_counter()
    model.fit(dev_x.iloc[tr], y[dev[tr]])
    p = positive_probability(model.predict_proba(dev_x.iloc[va]))
    elapsed = time.perf_counter() - start
    result = {
        "smoke_passed": True,
        "train_rows": len(tr),
        "valid_rows": len(va),
        "features": dev_x.shape[1],
        "n_estimators": int(config["smoke_n_estimators"]),
        "roc_auc": float(roc_auc_score(y[dev[va]], p)),
        "elapsed_seconds": float(elapsed),
        "environment": environment_info(),
    }
    write_json(ROOT / "reports" / "v6_tabicl_smoke.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_reference_oof(dev: np.ndarray) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    refs["catboost_v1"] = pd.read_csv(PARENT / "catboost_oof.csv").set_index("row_id").loc[dev, "probability"].to_numpy()
    refs["catboost_v2"] = pd.read_csv(V2 / "selected_oof.csv").set_index("row_id").loc[dev, "probability"].to_numpy()
    if V3_XGB.is_file():
        refs["xgb_v3_survey"] = pd.read_csv(V3_XGB).set_index("row_id").loc[dev, "probability"].to_numpy()
    if V5_OOF.is_file():
        v5 = pd.read_csv(V5_OOF).set_index("row_id").loc[dev]
        for name in ["CatBoost_BAG_L2", "ExtraTreesGini_BAG_L2", "NeuralNetFastAI_r102_BAG_L2"]:
            if name in v5:
                refs[f"v5_{name}"] = v5[name].to_numpy()
    return refs


def fold_paths(fold: int) -> dict[str, Path]:
    base = FOLDS_OUT / f"fold_{fold}"
    return {
        "val": base.with_name(base.name + "_val.csv"),
        "audit": base.with_name(base.name + "_audit.npy"),
        "test": base.with_name(base.name + "_test.npy"),
        "meta": base.with_name(base.name + "_meta.json"),
    }


def run_phase(config: dict, resume: bool) -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    FOLDS_OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config)
    env = environment_info()
    if config["device"] == "cuda" and not env["cuda_available"]:
        raise RuntimeError("CUDA unavailable")

    _, test_raw, y, _, dev, audit, fold_ids, dev_x, audit_x, test_x = load_data(config)
    y_dev = y[dev]
    oof = np.full(len(dev), np.nan, dtype=float)
    audit_fold = []
    test_fold = []
    fold_meta = []

    for fold in range(5):
        paths = fold_paths(fold)
        ready = all(paths[key].is_file() for key in ["val", "audit", "test", "meta"])
        if resume and ready:
            frame = pd.read_csv(paths["val"])
            va = np.flatnonzero(fold_ids == fold)
            expected_ids = dev[va]
            if not np.array_equal(frame["row_id"].to_numpy(dtype=int), expected_ids):
                raise ValueError(f"Fold {fold} resume row mismatch")
            oof[va] = frame["probability"].to_numpy(dtype=float)
            audit_fold.append(np.load(paths["audit"]))
            test_fold.append(np.load(paths["test"]))
            fold_meta.append(read_json(paths["meta"]))
            print(f"Resumed fold {fold}", flush=True)
            continue

        tr = np.flatnonzero(fold_ids != fold)
        va = np.flatnonzero(fold_ids == fold)
        model = make_model(config, int(config["n_estimators"]), int(config["random_state"]) + fold)
        started = time.perf_counter()
        model.fit(dev_x.iloc[tr], y_dev[tr])
        p_val = positive_probability(model.predict_proba(dev_x.iloc[va]))
        p_audit = positive_probability(model.predict_proba(audit_x))
        p_test = positive_probability(model.predict_proba(test_x))
        seconds = time.perf_counter() - started

        oof[va] = p_val
        audit_fold.append(p_audit)
        test_fold.append(p_test)
        meta = {
            "fold": fold,
            "train_rows": len(tr),
            "valid_rows": len(va),
            "seconds": float(seconds),
            "n_estimators": int(config["n_estimators"]),
            "seed": int(config["random_state"]) + fold,
            "completed_utc": now(),
        }
        fold_meta.append(meta)
        pd.DataFrame({"row_id": dev[va], "probability": p_val}).to_csv(paths["val"], index=False)
        np.save(paths["audit"], p_audit)
        np.save(paths["test"], p_test)
        write_json(paths["meta"], meta)
        print(json.dumps(meta), flush=True)
        del model
        torch.cuda.empty_cache()

    if not np.isfinite(oof).all():
        raise RuntimeError("OOF contains missing/non-finite probabilities")
    audit_p = np.mean(np.vstack(audit_fold), axis=0)
    test_p = np.mean(np.vstack(test_fold), axis=0)
    threshold, dev_f1 = select_threshold(y_dev, oof, config)
    dev_metrics = classification_metrics(y_dev, oof, threshold)
    audit_metrics = classification_metrics(y[audit], audit_p, threshold)
    fold_f1 = [
        float(f1_score(y_dev[fold_ids == fold], oof[fold_ids == fold] >= threshold, zero_division=0))
        for fold in range(5)
    ]
    refs = load_reference_oof(dev)
    correlations = {name: float(np.corrcoef(oof, p)[0, 1]) for name, p in refs.items()}

    pd.DataFrame({"row_id": dev, "target": y_dev, "dev_fold": fold_ids, "probability": oof}).to_csv(OUT / "oof.csv", index=False)
    pd.DataFrame({"row_id": audit, "probability": audit_p}).to_csv(OUT / "audit_probabilities.csv", index=False)
    ids = pd.read_csv(ROOT / "data" / "raw" / "submission.csv")["Id"]
    if len(ids) != len(test_raw):
        raise ValueError("Submission/test row count mismatch")
    pd.DataFrame({"Id": ids, "probability": test_p}).to_csv(OUT / "test_probabilities.csv", index=False)

    result = {
        "version": config["version"],
        "completed_utc": now(),
        "family": config["family"],
        "tabicl_version": env["tabicl"],
        "checkpoint_version": config["checkpoint_version"],
        "feature_mode": config["feature_mode"],
        "validation": "Exact frozen v1 grouped development folds; one TabICLv2 fit per fold; threshold selected on concatenated development OOF only.",
        "environment": env,
        "dev": {
            "threshold": threshold,
            "metrics": dev_metrics,
            "fold_f1": fold_f1,
            "fold_f1_std": float(np.std(fold_f1, ddof=1)),
        },
        "audit": audit_metrics,
        "correlations": correlations,
        "fold_runtime": fold_meta,
        "integrity": verify_v1(),
        "submitted": False,
        "limitations": [
            "Development threshold selection is optimistic by construction and is used only for local model comparison.",
            "The audit holdout was already inspected in earlier versions and remains diagnostic only.",
            "Audit/test predictions are the mean of five fold-trained TabICLv2 predictors, not a separate full-data refit.",
            "No Kaggle submission or leaderboard probing occurs in this run."
        ],
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "version": config["version"],
            "id": "tabiclv2_zero_shot",
            "parent": "v5_automl",
            "timestamp_utc": now(),
            "hypothesis": "TabICLv2 adds a genuinely different foundation-model error structure and can improve standalone or ensemble F1 on the frozen folds.",
            "validation": result["validation"],
            "result": {
                "dev_oof_f1": dev_f1,
                "threshold": threshold,
                "audit_f1": audit_metrics["f1"],
                "corr_v5_catboost": correlations.get("v5_CatBoost_BAG_L2"),
                "corr_v5_fastai": correlations.get("v5_NeuralNetFastAI_r102_BAG_L2"),
            },
            "decision": "foundation_model_candidate",
            "artifact": "artifacts/v6_tabicl",
            "submitted": False,
        }, allow_nan=False) + "\n")
    print(json.dumps({
        "dev_oof_f1": dev_f1,
        "threshold": threshold,
        "fold_f1": fold_f1,
        "audit_f1": audit_metrics["f1"],
        "correlations": correlations,
        "submitted": False,
    }, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["check", "smoke", "run"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v6_tabicl.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check_phase(config)
    elif args.phase == "smoke":
        smoke_phase(config)
    else:
        run_phase(config, args.resume)


if __name__ == "__main__":
    main()
