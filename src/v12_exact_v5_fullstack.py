"""v12: exact v5 44-base L1 portfolio + CatBoost_BAG_L2 on all labels.

The v5 predictor retains the exact zeroshot hyperparameter portfolio that was
used in the original run.  This script filters that portfolio to precisely the
44 L1 base model names consumed by v5 CatBoost_BAG_L2, preserves their original
ag_args (including name suffixes and priorities), trains only those L1 models,
and trains only a single default CatBoost at L2.  No weighted ensembles or other
L2 models are trained.  Threshold 0.315 is frozen from v5.  No remote submit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from baseline import check_submission
from v2_features import build_features
from v11_v5_fullstack import build_full_folds, load_raw


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v12_exact_v5_fullstack"
PREDICTOR_PATH = OUT / "predictor"
PARENT = ROOT / "artifacts" / "v5_automl"

PREFIX = {
    "GBM": "LightGBM",
    "CAT": "CatBoost",
    "XGB": "XGBoost",
    "FASTAI": "NeuralNetFastAI",
    "NN_TORCH": "NeuralNetTorch",
    "RF": "RandomForest",
    "XT": "ExtraTrees",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if expected_index is not None and len(s) == len(expected_index) and set(s.index.tolist()) == set(expected_index.tolist()):
        s = s.reindex(expected_index)
    return s.to_numpy(dtype=float)


def metrics(y, p, t: float) -> dict:
    pred = np.asarray(p) >= t
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "positive_prediction_rate": float(pred.mean()),
    }


def config_model_name(model_type: str, cfg: dict) -> str:
    suffix = str(cfg.get("ag_args", {}).get("name_suffix", ""))
    return f"{PREFIX[model_type]}{suffix}_BAG_L1"


def exact_hyperparameters():
    from autogluon.tabular import TabularPredictor

    parent = TabularPredictor.load(PARENT / "predictor")
    stack = parent._trainer.load_model("CatBoost_BAG_L2")
    parent_names = list(stack.get_info()["stacker_info"]["base_model_names"])
    fit_hp = copy.deepcopy(parent.fit_hyperparameters_)
    level1 = {}
    generated = []
    parent_rank = {name: i for i, name in enumerate(parent_names)}
    for model_type, prefix in PREFIX.items():
        raw = fit_hp.get(model_type, [])
        configs = raw if isinstance(raw, list) else [raw]
        kept = []
        for cfg in configs:
            name = config_model_name(model_type, cfg)
            if name in parent_names:
                cloned = copy.deepcopy(cfg)
                # Priority changes scheduling only, not model hyperparameters.  We
                # explicitly encode the parent's L1 order so the rebuilt stack's
                # feature ordering is deterministic and parent-identical.
                cloned.setdefault("ag_args", {})["priority"] = 1000 - parent_rank[name]
                kept.append(cloned)
                generated.append(name)
        if kept:
            level1[model_type] = kept
    missing = [n for n in parent_names if n not in generated]
    extra = [n for n in generated if n not in parent_names]
    if missing or extra or len(generated) != 44:
        raise RuntimeError(f"Could not reconstruct exact parent L1 portfolio: missing={missing}, extra={extra}, count={len(generated)}")
    # Original ag_args priorities recover the same cross-family training order.
    generated_by_priority = []
    for mt, cfgs in level1.items():
        for cfg in cfgs:
            generated_by_priority.append((int(cfg.get("ag_args", {}).get("priority", 0)), config_model_name(mt, cfg)))
    generated_by_priority.sort(key=lambda z: z[0], reverse=True)
    ordered = [n for _, n in generated_by_priority]
    if ordered != parent_names:
        raise RuntimeError("Failed to impose exact parent L1 scheduling order")
    return {1: level1, 2: {"CAT": [{}]}}, parent_names


def check(config: dict) -> None:
    version = importlib.metadata.version("autogluon.tabular")
    if version != "1.6.3":
        raise RuntimeError(f"Expected AutoGluon 1.6.3, got {version}")
    hp, names = exact_hyperparameters()
    train, test, y, _, manifest = load_raw()
    folds = build_full_folds(y, manifest, int(config["seed"]))
    f = build_features(train, config["feature_mode"])
    if f.shape[1] != 64:
        raise ValueError(f"Expected 64 survey features, got {f.shape[1]}")
    report = {
        "check_passed": True,
        "autogluon_version": version,
        "rows": len(train),
        "test_rows": len(test),
        "features": f.shape[1],
        "parent_l1_count": len(names),
        "parent_l1_names": names,
        "recovered_l1_count": sum(len(v) for v in hp[1].values()),
        "l2_models_requested": ["CatBoost_BAG_L2"],
        "weighted_ensemble": False,
        "num_bag_folds": int(config["num_bag_folds"]),
        "fold_counts": {str(k): int(np.sum(folds == k)) for k in range(5)},
        "time_limit_seconds": int(config["time_limit_seconds"]),
        "remote_submission_authorized": False,
    }
    write_json(ROOT / "reports" / "v12_exact_preflight.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def run(config: dict) -> None:
    from autogluon.tabular import TabularPredictor

    if OUT.exists():
        if (OUT / "results.json").is_file():
            print("v12 already complete; leaving immutable artifacts untouched", flush=True)
            return
        raise FileExistsError("Partial v12 output exists; inspect before retrying")

    hp, parent_names = exact_hyperparameters()
    train, test, y, sample, manifest = load_raw()
    folds = build_full_folds(y, manifest, int(config["seed"]))
    features = build_features(train, config["feature_mode"])
    test_features = build_features(test, config["feature_mode"])
    train_data = features.copy()
    train_data[TARGET] = y
    train_data["__fold__"] = folds
    OUT.mkdir(parents=True)
    write_json(OUT / "config.json", config)
    write_json(OUT / "exact_hyperparameters.json", {str(k): v for k, v in hp.items()})

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
        hyperparameters=hp,
        time_limit=int(config["time_limit_seconds"]),
        num_cpus=int(config["num_cpus"]),
        num_gpus=int(config["num_gpus"]),
        num_bag_folds=int(config["num_bag_folds"]),
        num_stack_levels=1,
        num_bag_sets=1,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        fit_full_last_level_weighted_ensemble=False,
        full_weighted_ensemble_additionally=False,
        calibrate_decision_threshold=False,
    )
    fit_seconds = time.monotonic() - started
    write_json(OUT / "fit_receipt.json", {
        "completed_utc": now(), "seconds": fit_seconds, "training_rows": len(train),
        "models": predictor.model_names(), "submitted": False,
    })

    model_name = config["selected_model"]
    if model_name not in predictor.model_names():
        raise RuntimeError("Exact CatBoost_BAG_L2 was not trained")
    m = predictor._trainer.load_model(model_name)
    rebuilt_names = list(m.get_info()["stacker_info"]["base_model_names"])
    exact_set = set(rebuilt_names) == set(parent_names) and len(rebuilt_names) == len(parent_names)
    exact_order = rebuilt_names == parent_names
    if not exact_set:
        write_json(OUT / "structural_mismatch.json", {
            "parent": parent_names, "rebuilt": rebuilt_names,
            "missing": [x for x in parent_names if x not in rebuilt_names],
            "extra": [x for x in rebuilt_names if x not in parent_names],
        })
        raise RuntimeError("v12 trained but exact 44-base parent reproduction failed; no candidate emitted")

    oof = normalize_proba(predictor.predict_proba_oof(model=model_name, as_multiclass=False), np.arange(len(train)))
    test_p = normalize_proba(predictor.predict_proba(test_features, model=model_name, as_multiclass=False))
    t = float(config["frozen_threshold"])
    fold_f1 = [float(f1_score(y[folds == f], oof[folds == f] >= t, zero_division=0)) for f in range(5)]

    pd.DataFrame({"row_id": np.arange(len(train)), "target": y, "full_fold": folds, "probability": oof}).to_csv(OUT / "selected_oof.csv", index=False)
    pd.DataFrame({"Id": sample.Id, "probability": test_p}).to_csv(OUT / "test_probabilities.csv", index=False)
    predictor.leaderboard(extra_info=True, silent=True).to_csv(OUT / "leaderboard.csv", index=False)
    sub = sample.copy()
    sub[TARGET] = (test_p >= t).astype(np.int64)
    sub_path = ROOT / "submissions" / "v12_v5_exact44_fullstack_t0315.csv"
    if sub_path.exists():
        raise FileExistsError(sub_path)
    sub.to_csv(sub_path, index=False)
    validation = check_submission(sample, pd.read_csv(sub_path))

    comparisons = {}
    for label, path, col in [
        ("v5_dev_only", PARENT / "test_probabilities.csv", "CatBoost_BAG_L2"),
        ("v11_full37", ROOT / "artifacts" / "v11_v5_fullstack" / "test_probabilities.csv", "probability"),
    ]:
        if path.is_file():
            q = pd.read_csv(path)[col].to_numpy(float)
            comparisons[label] = {
                "corr": float(np.corrcoef(test_p, q)[0, 1]),
                "mae": float(np.mean(np.abs(test_p - q))),
                "hard_disagreement": int(np.sum((test_p >= t) != (q >= t))),
            }
    result = {
        "version": config["version"], "completed_utc": now(), "training_rows": len(train),
        "all_available_labels_used": True, "features": 64, "selected_model": model_name,
        "frozen_threshold": t, "parent_l1_count": 44, "rebuilt_l1_count": len(rebuilt_names),
        "exact_parent_l1_match": exact_set, "exact_parent_l1_order": exact_order,
        "parent_l1_names": parent_names, "rebuilt_l1_names": rebuilt_names,
        "oof": metrics(y, oof, t), "fold_f1": fold_f1, "fold_f1_std": float(np.std(fold_f1, ddof=1)),
        "fit_seconds": fit_seconds,
        "submission": {"path": str(sub_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(sub_path), **validation},
        "comparisons": comparisons, "submitted": False,
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({"version": config["version"], "id": "exact44_CatBoost_BAG_L2_full42154",
                            "parent": "v5_automl", "result": result, "artifact": str(OUT.relative_to(ROOT)),
                            "submitted": False}, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"completed": True, "exact_parent_l1_match": exact_set, "exact_parent_l1_order": exact_order, "oof_f1": result["oof"]["f1"],
                      "submission": result["submission"]}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v12_exact_v5_fullstack.json")
    args = ap.parse_args()
    cfg = read_json(args.config)
    if args.phase == "check": check(cfg)
    else: run(cfg)


if __name__ == "__main__":
    main()
