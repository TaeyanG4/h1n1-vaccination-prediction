"""Controlled HPO and full-label L2 refit around v5 AutoGluon CatBoost_BAG_L2.

The parent is an L2 CatBoost stacker over 44 L1 OOF predictions plus 64 survey
features. AutoGluon's own transform_features API is used to recover the exact
108-column L2 design matrix. HPO uses development labels/folds only. After the
winning config and threshold are frozen, the reused audit is scored
diagnostically and then included only in the final all-label L2 meta-model.
The L1 base models remain the original v5 development-trained models.

No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from autogluon.tabular import TabularPredictor

from v2_features import build_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v7_catboost_l2_hpo"
META = OUT / "meta"
MODELS = OUT / "models"
PARENT = ROOT / "artifacts" / "baseline_v1"
V5 = ROOT / "artifacts" / "v5_automl"
PREDICTOR_PATH = V5 / "predictor"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_v1() -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    expected = read_json(PARENT / "metrics.json")
    if sha(PARENT / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen split changed")
    return {"passed": True, "protected_files": len(protected)}


def select_threshold(y: np.ndarray, p: np.ndarray, config: dict) -> tuple[float, float]:
    grid = np.linspace(float(config["threshold_min"]), float(config["threshold_max"]), int(config["threshold_steps"]))
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def load_raw():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    y = labels[TARGET].to_numpy(dtype=int)
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    folds = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected frozen folds 0..4")
    return train, test, y, dev, audit, folds


def sanitize_for_catboost(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame.copy()
    cats = []
    for col in x.columns:
        if str(x[col].dtype) == "category" or x[col].dtype == "object":
            cats.append(col)
            x[col] = x[col].astype(object).where(x[col].notna(), "__MISSING__").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="raise").astype(float)
    return x, cats


def build_meta(config: dict, predictor: TabularPredictor, force: bool = False):
    META.mkdir(parents=True, exist_ok=True)
    paths = {name: META / f"{name}.pkl" for name in ["dev", "audit", "test"]}
    if not force and all(p.is_file() for p in paths.values()):
        dev_meta = pd.read_pickle(paths["dev"])
        audit_meta = pd.read_pickle(paths["audit"])
        test_meta = pd.read_pickle(paths["test"])
        return dev_meta, audit_meta, test_meta

    train, test, _, dev, audit, _ = load_raw()
    dev_meta = predictor.transform_features(model=config["parent_model"])
    audit_features = build_features(train.iloc[audit].copy(), config["feature_mode"])
    test_features = build_features(test.copy(), config["feature_mode"])
    audit_meta = predictor.transform_features(data=audit_features, model=config["parent_model"])
    test_meta = predictor.transform_features(data=test_features, model=config["parent_model"])
    if dev_meta.shape[1] != 108 or audit_meta.shape[1] != 108 or test_meta.shape[1] != 108:
        raise ValueError(f"Expected 108 L2 features, got {dev_meta.shape}, {audit_meta.shape}, {test_meta.shape}")
    if list(dev_meta.columns) != list(audit_meta.columns) or list(dev_meta.columns) != list(test_meta.columns):
        raise ValueError("L2 meta feature schema mismatch")
    dev_meta.to_pickle(paths["dev"])
    audit_meta.to_pickle(paths["audit"])
    test_meta.to_pickle(paths["test"])
    return dev_meta, audit_meta, test_meta


def cat_params(config: dict, spec: dict, seed: int, iterations: int | None = None) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": int(iterations or config["iterations"]),
        "depth": int(spec["depth"]),
        "learning_rate": float(spec["learning_rate"]),
        "l2_leaf_reg": float(spec["l2_leaf_reg"]),
        "random_strength": float(spec["random_strength"]),
        "bagging_temperature": float(spec["bagging_temperature"]),
        "scale_pos_weight": float(spec["scale_pos_weight"]),
        "random_seed": int(seed),
        "thread_count": int(config["thread_count"]),
        "allow_writing_files": False,
        "verbose": False,
    }


def fold_artifacts(spec_id: str, fold: int):
    d = OUT / "folds" / spec_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fold_{fold}_val.npy", MODELS / spec_id / f"fold_{fold}.cbm", d / f"fold_{fold}_meta.json"


def fit_fold(config: dict, spec: dict, fold: int, x: pd.DataFrame, cats: list[str], y: np.ndarray, folds: np.ndarray, resume: bool):
    val_path, model_path, meta_path = fold_artifacts(spec["id"], fold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    va = folds == fold
    tr = ~va
    if resume and val_path.is_file() and model_path.is_file() and meta_path.is_file():
        return np.load(val_path), read_json(meta_path)
    model = CatBoostClassifier(**cat_params(config, spec, int(config["random_seed"]) + fold))
    start = time.perf_counter()
    model.fit(
        x.loc[tr], y[tr], cat_features=cats,
        eval_set=(x.loc[va], y[va]),
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        use_best_model=True,
    )
    p = model.predict_proba(x.loc[va])[:, 1]
    elapsed = time.perf_counter() - start
    best_iter = int(model.get_best_iteration())
    meta = {
        "spec": spec["id"], "fold": fold, "train_rows": int(tr.sum()), "valid_rows": int(va.sum()),
        "best_iteration": best_iter, "seconds": float(elapsed), "completed_utc": now()
    }
    np.save(val_path, p)
    model.save_model(str(model_path))
    write_json(meta_path, meta)
    print(json.dumps(meta), flush=True)
    return p, meta


def screen(config: dict, x: pd.DataFrame, cats: list[str], y: np.ndarray, folds: np.ndarray, resume: bool):
    rows = []
    screen_folds = [int(f) for f in config["screen_folds"]]
    mask = np.isin(folds, screen_folds)
    for spec in config["candidates"]:
        pred = np.full(len(y), np.nan)
        metas = []
        for fold in screen_folds:
            p, meta = fit_fold(config, spec, fold, x, cats, y, folds, resume)
            pred[folds == fold] = p
            metas.append(meta)
        t, score = select_threshold(y[mask], pred[mask], config)
        rows.append({
            "id": spec["id"], "screen_f1": score, "threshold": t,
            "roc_auc": float(roc_auc_score(y[mask], pred[mask])),
            "mean_best_iteration": float(np.mean([m["best_iteration"] for m in metas])),
        })
    frame = pd.DataFrame(rows).sort_values(["screen_f1", "roc_auc"], ascending=False)
    frame.to_csv(OUT / "screening.csv", index=False)
    return frame


def full_eval(config: dict, promoted: list[dict], x: pd.DataFrame, cats: list[str], y: np.ndarray, folds: np.ndarray, resume: bool):
    rows = []
    all_pred = {}
    all_meta = {}
    for spec in promoted:
        pred = np.full(len(y), np.nan)
        metas = []
        for fold in range(5):
            p, meta = fit_fold(config, spec, fold, x, cats, y, folds, resume)
            pred[folds == fold] = p
            metas.append(meta)
        t, score = select_threshold(y, pred, config)
        fs = [float(f1_score(y[folds == f], pred[folds == f] >= t, zero_division=0)) for f in range(5)]
        rows.append({
            "id": spec["id"], "dev_oof_f1": score, "threshold": t,
            "roc_auc": float(roc_auc_score(y, pred)), "log_loss": float(log_loss(y, pred)),
            "fold_f1_std": float(np.std(fs, ddof=1)), "fold_f1": json.dumps(fs),
            "median_best_iteration": int(np.median([m["best_iteration"] + 1 for m in metas])),
        })
        all_pred[spec["id"]] = pred
        all_meta[spec["id"]] = metas
    frame = pd.DataFrame(rows).sort_values(["dev_oof_f1", "fold_f1_std"], ascending=[False, True])
    frame.to_csv(OUT / "full_hpo.csv", index=False)
    return frame, all_pred, all_meta


def validate_submission(path: Path, ids: np.ndarray):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Id", TARGET] or len(frame) != len(ids):
        raise ValueError(f"Bad submission schema: {path}")
    if not np.array_equal(frame["Id"].to_numpy(), ids):
        raise ValueError(f"ID mismatch: {path}")
    if set(frame[TARGET].unique()) - {0, 1} or frame.isna().any().any():
        raise ValueError(f"Invalid labels/missing values: {path}")


def check_phase(config: dict):
    predictor = TabularPredictor.load(PREDICTOR_PATH)
    info = predictor.model_info(config["parent_model"])
    train, test, y, dev, audit, folds = load_raw()
    meta = predictor.transform_features(model=config["parent_model"])
    print(json.dumps({
        "check_passed": True, "integrity": verify_v1(), "parent_model": config["parent_model"],
        "parent_model_type": info["model_type"], "parent_base_models": info["stacker_info"]["num_base_models"],
        "meta_features": meta.shape[1], "dev_rows": len(dev), "audit_rows": len(audit), "test_rows": len(test),
        "fold_counts": {str(i): int((folds == i).sum()) for i in range(5)}, "hpo_candidates": len(config["candidates"]),
        "remote_submission_authorized": bool(config["remote_submission_authorized"])
    }, indent=2))


def run_phase(config: dict, resume: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config)
    predictor = TabularPredictor.load(PREDICTOR_PATH)
    train, test, y_all, dev, audit, folds = load_raw()
    dev_meta, audit_meta, test_meta = build_meta(config, predictor)
    x_dev, cats = sanitize_for_catboost(dev_meta)
    x_audit, cats_a = sanitize_for_catboost(audit_meta)
    x_test, cats_t = sanitize_for_catboost(test_meta)
    if cats != cats_a or cats != cats_t:
        raise ValueError("Categorical meta columns mismatch")
    y_dev = y_all[dev]
    y_audit = y_all[audit]

    screening = screen(config, x_dev, cats, y_dev, folds, resume)
    spec_map = {s["id"]: s for s in config["candidates"]}
    promoted_ids = screening.head(int(config["promote_top_k"]))["id"].tolist()
    promoted = [spec_map[x] for x in promoted_ids]
    full, pred_map, _ = full_eval(config, promoted, x_dev, cats, y_dev, folds, resume)
    winner_id = str(full.iloc[0]["id"])
    winner = spec_map[winner_id]
    threshold = float(full.iloc[0]["threshold"])
    dev_f1 = float(full.iloc[0]["dev_oof_f1"])
    median_iter = int(full.iloc[0]["median_best_iteration"])

    audit_fold_pred, test_fold_pred = [], []
    for fold in range(5):
        _, model_path, _ = fold_artifacts(winner_id, fold)
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        audit_fold_pred.append(model.predict_proba(x_audit)[:, 1])
        test_fold_pred.append(model.predict_proba(x_test)[:, 1])
    audit_p = np.mean(np.vstack(audit_fold_pred), axis=0)
    test_cv_p = np.mean(np.vstack(test_fold_pred), axis=0)
    audit_f1 = float(f1_score(y_audit, audit_p >= threshold, zero_division=0))

    # Freeze complete: audit labels are only introduced now for final all-label L2 fitting.
    x_full = pd.concat([x_dev, x_audit], axis=0, ignore_index=True)
    y_full = np.concatenate([y_dev, y_audit])
    final_model = CatBoostClassifier(**cat_params(config, winner, int(config["random_seed"]) + 999, iterations=median_iter))
    final_model.fit(x_full, y_full, cat_features=cats, verbose=False)
    final_model.save_model(str(OUT / "final_l2_full_label.cbm"))
    test_full_p = final_model.predict_proba(x_test)[:, 1]

    ids = pd.read_csv(ROOT / "data" / "raw" / "submission.csv")["Id"].to_numpy()
    sub_dir = ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    cv_path = sub_dir / "v7_catboost_l2_hpo_cv.csv"
    full_path = sub_dir / "v7_catboost_l2_hpo_full_label.csv"
    pd.DataFrame({"Id": ids, TARGET: (test_cv_p >= threshold).astype(int)}).to_csv(cv_path, index=False)
    pd.DataFrame({"Id": ids, TARGET: (test_full_p >= threshold).astype(int)}).to_csv(full_path, index=False)
    validate_submission(cv_path, ids)
    validate_submission(full_path, ids)
    pd.DataFrame({"Id": ids, "probability": test_cv_p}).to_csv(OUT / "test_cv_probability.csv", index=False)
    pd.DataFrame({"Id": ids, "probability": test_full_p}).to_csv(OUT / "test_full_label_probability.csv", index=False)
    pd.DataFrame({"row_id": dev, "target": y_dev, "dev_fold": folds, "probability": pred_map[winner_id]}).to_csv(OUT / "winner_oof.csv", index=False)
    pd.DataFrame({"row_id": audit, "target": y_audit, "probability": audit_p}).to_csv(OUT / "winner_audit.csv", index=False)

    parent = read_json(V5 / "results.json")
    result = {
        "version": config["version"], "completed_utc": now(), "winner": winner,
        "screen_promoted": promoted_ids, "threshold": threshold, "dev_oof_f1": dev_f1,
        "dev_gain_vs_v5_parent": dev_f1 - float(parent["selection"]["dev_oof_tuned_f1"]),
        "audit_f1": audit_f1, "v5_parent_audit_f1": float(parent["selected_audit"]["f1"]),
        "median_best_iteration": median_iter,
        "final_training_rows": len(y_full),
        "final_scope": "All 42,154 labels used only after HPO/threshold frozen. L1 stack features still come from the original v5 development-trained base models.",
        "submissions_created": {
            "cv_ensemble": str(cv_path.relative_to(ROOT)).replace("\\", "/"),
            "full_label_l2": str(full_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "submission_hashes": {"cv_ensemble": sha(cv_path), "full_label_l2": sha(full_path)},
        "submitted": False,
        "limitations": [
            "HPO and threshold selection reuse development OOF and are therefore selection-biased.",
            "The audit set is reused and diagnostic; its labels are used only after freeze for the final all-label L2 model.",
            "This is not a complete full-data AutoGluon stack refit: the 44 L1 base learners remain the v5 development-trained models.",
            "No Kaggle submission occurs in this run."
        ]
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": config["version"], "id": winner_id, "parent": "v5_automl/CatBoost_BAG_L2",
            "timestamp_utc": now(), "hypothesis": "Controlled L2 CatBoost HPO plus all-label meta refit improves the strongest v5 scout.",
            "validation": "2-fold screen -> top-3 exact 5-fold OOF on frozen development folds; threshold frozen before reused audit.",
            "result": {"dev_oof_f1": dev_f1, "threshold": threshold, "audit_f1": audit_f1, "median_best_iteration": median_iter},
            "decision": "candidate_ready_for_dry_run", "artifact": "artifacts/v7_catboost_l2_hpo", "submitted": False
        }, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v7_catboost_hpo.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check_phase(config)
    else:
        run_phase(config, args.resume)


if __name__ == "__main__":
    main()
