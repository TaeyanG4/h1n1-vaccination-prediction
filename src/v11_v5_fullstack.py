"""v11: complete full-label rebuild of the v5 AutoGluon CatBoost_BAG_L2 stack.

This is a finalization experiment, not a new model-selection sweep. The parent v5
selected CatBoost_BAG_L2 at threshold 0.315 on frozen development OOF. v11 keeps
that model identity and threshold fixed, but rebuilds the entire AutoGluon bagged
stack on all 42,154 labeled rows so that L1 OOF meta-features and the L2 CatBoost
both benefit from every available label.

No Kaggle submission occurs in this script.
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
from sklearn.model_selection import StratifiedGroupKFold

from baseline import check_submission
from v2_features import build_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v11_v5_fullstack"
PREDICTOR_PATH = OUT / "predictor"
PARENT = ROOT / "artifacts" / "v5_automl"
V1 = ROOT / "artifacts" / "baseline_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def normalize_proba(obj, expected_index=None) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        if 1 in obj.columns:
            s = obj[1]
        elif "1" in obj.columns:
            s = obj["1"]
        else:
            s = obj.iloc[:, -1]
    elif isinstance(obj, pd.Series):
        s = obj
    else:
        a = np.asarray(obj)
        if a.ndim == 2:
            a = a[:, -1]
        return a.astype(float)
    if expected_index is not None and len(s) == len(expected_index):
        if set(s.index.tolist()) == set(expected_index.tolist()):
            s = s.reindex(expected_index)
    return s.to_numpy(dtype=float)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "positive_prediction_rate": float(pred.mean()),
    }


def verify_contract(config: dict) -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    parent = read_json(PARENT / "results.json")
    if parent["selection"]["candidate"] != config["selected_model"]:
        raise ValueError("Parent v5 selected model changed")
    if not np.isclose(float(parent["selection"]["threshold"]), float(config["frozen_threshold"]), atol=1e-12):
        raise ValueError("Parent v5 threshold changed")
    if config["dynamic_stacking"] is not False:
        raise ValueError("v11 must disable dynamic stacking to avoid v5's unnecessary sub-fit overhead")
    return {
        "passed": True,
        "protected_files": len(protected),
        "parent_model": parent["selection"]["candidate"],
        "parent_threshold": float(parent["selection"]["threshold"]),
        "parent_dev_oof_f1": float(parent["selection"]["dev_oof_tuned_f1"]),
        "parent_audit_f1": float(parent["selected_audit"]["f1"]),
    }


def load_raw():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    sample = pd.read_csv(raw / "submission.csv")
    manifest = pd.read_csv(V1 / "split_manifest.csv", dtype={"group_hash": str})
    if len(train) != len(labels) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(train))):
        raise ValueError("Row identity mismatch")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")
    if {TARGET, "vacc_seas_f"} & set(train.columns):
        raise ValueError("Target leaked into feature table")
    y = labels[TARGET].to_numpy(dtype=int)
    return train, test, y, sample, manifest


def build_full_folds(y: np.ndarray, manifest: pd.DataFrame, seed: int) -> np.ndarray:
    dev_mask = manifest.partition.eq("development").to_numpy()
    audit_mask = manifest.partition.eq("audit").to_numpy()
    dev = np.flatnonzero(dev_mask)
    audit = np.flatnonzero(audit_mask)
    folds = np.full(len(manifest), -1, dtype=int)
    folds[dev] = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)

    audit_groups = manifest.loc[audit, "group_hash"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, va_local) in enumerate(splitter.split(np.zeros(len(audit)), y[audit], audit_groups)):
        folds[audit[va_local]] = fold
    if np.any(folds < 0) or sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Incomplete full-data fold assignment")
    fold_by_group = pd.DataFrame({"g": manifest.group_hash.astype(str), "fold": folds}).groupby("g")["fold"].nunique()
    if int(fold_by_group.max()) != 1:
        raise ValueError("Exact duplicate feature group crosses full-data folds")
    return folds


def fold_report(y: np.ndarray, folds: np.ndarray) -> dict:
    return {
        str(f): {
            "rows": int(np.sum(folds == f)),
            "positives": int(np.sum(y[folds == f])),
            "positive_rate": float(np.mean(y[folds == f])),
        }
        for f in range(5)
    }


def old_stack_info() -> dict:
    from autogluon.tabular import TabularPredictor

    p = TabularPredictor.load(PARENT / "predictor")
    model = p._trainer.load_model("CatBoost_BAG_L2")
    info = model.get_info().get("stacker_info", {})
    names = list(info.get("base_model_names", []))
    return {"base_model_count": len(names), "base_model_names": names}


def check_phase(config: dict) -> None:
    version = importlib.metadata.version("autogluon.tabular")
    if version != config["autogluon_version"]:
        raise RuntimeError(f"Expected AutoGluon {config['autogluon_version']}, got {version}")
    contract = verify_contract(config)
    train, test, y, _, manifest = load_raw()
    folds = build_full_folds(y, manifest, int(config["seed"]))
    features = build_features(train, config["feature_mode"])
    test_features = build_features(test, config["feature_mode"])
    if features.shape[1] != 64 or list(features.columns) != list(test_features.columns):
        raise ValueError("Expected stable 64-column v2 survey representation")
    stack = old_stack_info()
    payload = {
        "check_passed": True,
        "autogluon_version": version,
        "contract": contract,
        "all_training_rows": len(train),
        "test_rows": len(test),
        "features": int(features.shape[1]),
        "full_fold_report": fold_report(y, folds),
        "all_duplicate_groups_fold_safe": True,
        "parent_stack": stack,
        "dynamic_stacking": False,
        "time_limit_seconds": int(config["time_limit_seconds"]),
        "remote_submission_authorized": False,
    }
    write_json(ROOT / "reports" / "v11_fullstack_preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def fit_or_load(config: dict, resume: bool):
    from autogluon.tabular import TabularPredictor

    if resume:
        if not (PREDICTOR_PATH / "predictor.pkl").is_file():
            raise FileNotFoundError("Cannot resume/export: predictor.pkl is absent")
        return TabularPredictor.load(PREDICTOR_PATH), None
    if OUT.exists():
        raise FileExistsError("v11 output already exists; inspect it and use --resume if appropriate")

    train, _, y, _, manifest = load_raw()
    folds = build_full_folds(y, manifest, int(config["seed"]))
    features = build_features(train, config["feature_mode"])
    train_data = features.copy()
    train_data[TARGET] = y
    train_data["__fold__"] = folds

    OUT.mkdir(parents=True)
    write_json(OUT / "config.json", config)
    pd.DataFrame({
        "row_id": np.arange(len(train)),
        "group_hash": manifest.group_hash.astype(str),
        "parent_partition": manifest.partition,
        "parent_dev_fold": manifest.dev_fold,
        "full_fold": folds,
        "target": y,
    }).to_csv(OUT / "full_fold_manifest.csv", index=False)

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
        dynamic_stacking=False,
        calibrate_decision_threshold=False,
    )
    seconds = time.monotonic() - started
    write_json(OUT / "fit_receipt.json", {
        "completed_utc": now(), "seconds": seconds, "training_rows": len(train),
        "autogluon_version": importlib.metadata.version("autogluon.tabular"),
        "predictor_best": predictor.model_best, "dynamic_stacking": False,
        "time_limit_seconds": int(config["time_limit_seconds"]), "submitted": False,
    })
    return predictor, seconds


def run_phase(config: dict, resume: bool) -> None:
    version = importlib.metadata.version("autogluon.tabular")
    if version != config["autogluon_version"]:
        raise RuntimeError(f"Expected AutoGluon {config['autogluon_version']}, got {version}")
    contract = verify_contract(config)
    predictor, fit_seconds = fit_or_load(config, resume)
    train, test, y, sample, manifest = load_raw()
    folds = build_full_folds(y, manifest, int(config["seed"]))
    model_name = config["selected_model"]
    if model_name not in predictor.model_names():
        write_json(OUT / "failure_missing_selected_model.json", {
            "selected_model": model_name,
            "available_models": predictor.model_names(),
            "completed_utc": now(),
        })
        raise RuntimeError(f"Frozen v5 model {model_name} was not trained within the v11 budget")

    old_stack = old_stack_info()
    new_model = predictor._trainer.load_model(model_name)
    new_stacker = new_model.get_info().get("stacker_info", {})
    new_bases = list(new_stacker.get("base_model_names", []))
    base_match = new_bases == old_stack["base_model_names"]

    raw_oof = predictor.predict_proba_oof(model=model_name, as_multiclass=False)
    oof = normalize_proba(raw_oof, expected_index=np.arange(len(train)))
    if len(oof) != len(train) or not np.isfinite(oof).all():
        raise ValueError("Full-data OOF export failed")

    test_features = build_features(test, config["feature_mode"])
    test_p = normalize_proba(predictor.predict_proba(test_features, model=model_name, as_multiclass=False))
    if len(test_p) != len(test) or not np.isfinite(test_p).all():
        raise ValueError("Test probability export failed")

    threshold = float(config["frozen_threshold"])
    overall_oof = metrics(y, oof, threshold)
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    former_audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    slice_oof = {
        "former_development": metrics(y[dev], oof[dev], threshold),
        "former_audit": metrics(y[former_audit], oof[former_audit], threshold),
    }
    fold_f1 = [float(f1_score(y[folds == f], oof[folds == f] >= threshold, zero_division=0)) for f in range(5)]

    pd.DataFrame({"row_id": np.arange(len(train)), "target": y, "full_fold": folds, "probability": oof}).to_csv(OUT / "selected_oof.csv", index=False)
    pd.DataFrame({"Id": sample.Id, "probability": test_p}).to_csv(OUT / "test_probabilities.csv", index=False)
    predictor.leaderboard(extra_info=True, silent=True).to_csv(OUT / "leaderboard.csv", index=False)

    submission = sample.copy()
    submission[TARGET] = (test_p >= threshold).astype(np.int64)
    sub_path = ROOT / "submissions" / "v11_v5_fullstack_catboost_l2_t0315.csv"
    if sub_path.exists():
        raise FileExistsError(sub_path)
    submission.to_csv(sub_path, index=False)
    validation = check_submission(sample, pd.read_csv(sub_path))

    comparisons = {}
    old_test_path = PARENT / "test_probabilities.csv"
    if old_test_path.is_file():
        old = pd.read_csv(old_test_path, usecols=[model_name])[model_name].to_numpy(dtype=float)
        comparisons["vs_v5_dev_only"] = {
            "probability_correlation": float(np.corrcoef(test_p, old)[0, 1]),
            "mean_absolute_probability_difference": float(np.mean(np.abs(test_p - old))),
            "hard_disagreement_at_0315": int(np.sum((test_p >= threshold) != (old >= threshold))),
        }
    v9_path = ROOT / "artifacts" / "v9_context_full" / "test_probabilities.csv"
    if v9_path.is_file():
        v9 = pd.read_csv(v9_path)["probability"].to_numpy(dtype=float)
        comparisons["vs_v9_context_full"] = {
            "probability_correlation": float(np.corrcoef(test_p, v9)[0, 1]),
            "mean_absolute_probability_difference": float(np.mean(np.abs(test_p - v9))),
        }

    result = {
        "version": config["version"],
        "completed_utc": now(),
        "engine": "AutoGluon",
        "autogluon_version": version,
        "training_rows": len(train),
        "all_available_labels_used": True,
        "feature_mode": config["feature_mode"],
        "features": 64,
        "selected_model": model_name,
        "frozen_threshold": threshold,
        "dynamic_stacking": False,
        "full_fold_report": fold_report(y, folds),
        "parent_stack": old_stack,
        "rebuilt_stack": {"base_model_count": len(new_bases), "base_model_names": new_bases, "exact_base_name_match_parent": base_match},
        "oof_at_frozen_threshold": overall_oof,
        "oof_slice_descriptive_only": slice_oof,
        "fold_f1_at_frozen_threshold": fold_f1,
        "fold_f1_std": float(np.std(fold_f1, ddof=1)),
        "fit_seconds": fit_seconds,
        "submission": {"path": str(sub_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(sub_path), **validation},
        "comparisons": comparisons,
        "contract": contract,
        "submitted": False,
        "limitations": [
            "The model identity and threshold are frozen from v5; v11 is intended as full-label finalization, not renewed model selection.",
            "Full-data OOF uses new five-fold bagging where former audit rows participate in training for other folds; former audit OOF is descriptive and is not an untouched holdout.",
            "AutoGluon best_quality is time-budgeted. Exact parent base-model name matching is reported and should be checked before submission.",
            "No Kaggle submission occurs in this run."
        ],
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": config["version"], "id": "CatBoost_BAG_L2_full42154", "parent": "v5_automl",
            "timestamp_utc": result["completed_utc"],
            "hypothesis": "Rebuilding the entire v5 L1 bagging plus L2 CatBoost stack on all 42,154 labels improves final hidden/private generalization.",
            "validation": "Frozen v5 model identity and t=0.315; group-safe 5-fold full-data bagging; OOF descriptive only; no remote submission.",
            "result": {"oof_f1": overall_oof["f1"], "fold_f1_std": result["fold_f1_std"], "submission": result["submission"], "comparisons": comparisons},
            "artifact": "artifacts/v11_v5_fullstack", "submitted": False
        }, allow_nan=False) + "\n")
    print(json.dumps({
        "completed": True,
        "selected_model": model_name,
        "training_rows": len(train),
        "base_model_count": len(new_bases),
        "exact_parent_stack_match": base_match,
        "full_oof_f1_at_frozen_0315": overall_oof["f1"],
        "submission": result["submission"],
        "submitted": False,
    }, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v11_v5_fullstack.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check_phase(config)
    else:
        run_phase(config, args.resume)


if __name__ == "__main__":
    main()
