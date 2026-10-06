"""v8 public-intelligence interaction feature experiment.

Primary candidate is predeclared (`survey_all`). Opinion/context-only candidates are
two-fold diagnostics only, reducing candidate-selection bias. The CatBoost recipe is
identical to v2 `cat_survey`; only the row-local representation changes.

No Kaggle submission is performed.
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
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from v2_features import fit_schema, transform
from v8_features import build_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v8_interactions"
PARENT_V1 = ROOT / "artifacts" / "baseline_v1"
PARENT_V2 = ROOT / "artifacts" / "v2" / "candidates" / "cat_survey"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_parent() -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    expected = read_json(PARENT_V1 / "metrics.json")
    if sha(PARENT_V1 / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen split changed")
    return {"passed": True, "protected_files": len(protected)}


def load_data():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    sample = pd.read_csv(raw / "submission.csv")
    manifest = pd.read_csv(PARENT_V1 / "split_manifest.csv", dtype={"group_hash": str})
    if len(train) != len(labels) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(train))):
        raise ValueError("Row identity mismatch")
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    folds = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    return train, test, labels[TARGET].to_numpy(dtype=int), sample, manifest, dev, audit, folds


def threshold_grid(config: dict) -> np.ndarray:
    return np.linspace(float(config["threshold_min"]), float(config["threshold_max"]), int(config["threshold_steps"]))


def select_threshold(y: np.ndarray, p: np.ndarray, config: dict) -> tuple[float, float]:
    grid = threshold_grid(config)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "positive_prediction_rate": float(pred.mean()),
    }


def model_for(config: dict, schema: dict, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        **config["catboost"], random_seed=seed, thread_count=int(config["threads"]),
        cat_features=schema["categorical"], allow_writing_files=False, verbose=False,
    )


def fold_paths(candidate: str, fold: int):
    d = OUT / "candidates" / candidate
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fold{fold}.joblib", d / f"fold{fold}_oof.csv", d / f"fold{fold}_predictions.npz", d / f"fold{fold}_metrics.json"


def fit_fold(config: dict, spec: dict, fold: int, train: pd.DataFrame, test: pd.DataFrame,
             y: np.ndarray, dev: np.ndarray, audit: np.ndarray, folds: np.ndarray, resume: bool):
    model_path, oof_path, pred_path, receipt_path = fold_paths(spec["id"], fold)
    valid_ids = dev[folds == fold]
    train_ids = dev[folds != fold]
    if resume and all(p.is_file() for p in [model_path, oof_path, pred_path, receipt_path]):
        saved = pd.read_csv(oof_path)
        if not np.array_equal(saved.row_id.to_numpy(), valid_ids):
            raise ValueError(f"Resume row mismatch {spec['id']} fold {fold}")
        print(f"REUSE {spec['id']} fold={fold}", flush=True)
        return
    if any(p.exists() for p in [model_path, oof_path, pred_path, receipt_path]):
        raise FileExistsError(f"Partial fold artifacts for {spec['id']} fold {fold}; inspect before rerun")

    started = time.perf_counter()
    all_features = build_features(train, spec["block"])
    test_features = build_features(test, spec["block"])
    schema = fit_schema(all_features.iloc[train_ids], "catboost")
    model = model_for(config, schema, int(config["seed"]) + fold)
    model.fit(transform(all_features.iloc[train_ids], schema), y[train_ids])
    val_p = model.predict_proba(transform(all_features.iloc[valid_ids], schema))[:, 1]
    audit_p = model.predict_proba(transform(all_features.iloc[audit], schema))[:, 1]
    test_p = model.predict_proba(transform(test_features, schema))[:, 1]
    if not all(np.isfinite(a).all() for a in [val_p, audit_p, test_p]):
        raise ValueError("Nonfinite probabilities")
    joblib.dump({"model": model, "schema": schema, "spec": spec}, model_path, compress=3)
    pd.DataFrame({"row_id": valid_ids, "target": y[valid_ids], "probability": val_p}).to_csv(oof_path, index=False)
    np.savez_compressed(pred_path, audit=audit_p, test=test_p)
    receipt = {
        "candidate": spec["id"], "block": spec["block"], "fold": fold,
        "train_rows": int(len(train_ids)), "valid_rows": int(len(valid_ids)),
        "features": int(all_features.shape[1]), "seconds": float(time.perf_counter() - started),
        "completed_utc": now(),
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def assemble(candidate: str, y_dev: np.ndarray, dev: np.ndarray, folds: np.ndarray, config: dict):
    d = OUT / "candidates" / candidate
    parts = [pd.read_csv(d / f"fold{f}_oof.csv") for f in range(5)]
    oof = pd.concat(parts).set_index("row_id").loc[dev].reset_index()
    if not np.array_equal(oof.target.to_numpy(), y_dev):
        raise ValueError("OOF alignment failure")
    audit_parts, test_parts = [], []
    for f in range(5):
        with np.load(d / f"fold{f}_predictions.npz") as z:
            audit_parts.append(z["audit"])
            test_parts.append(z["test"])
    audit_p = np.mean(np.vstack(audit_parts), axis=0)
    test_p = np.mean(np.vstack(test_parts), axis=0)
    p = oof.probability.to_numpy()
    t, score = select_threshold(y_dev, p, config)
    fold_f1 = [float(f1_score(y_dev[folds == f], p[folds == f] >= t, zero_division=0)) for f in range(5)]
    oof.to_csv(d / "oof.csv", index=False)
    np.savez_compressed(d / "ensemble_predictions.npz", audit=audit_p, test=test_p)
    return p, audit_p, test_p, {
        "threshold": t, "dev_oof_f1": score, "metrics": metrics(y_dev, p, t),
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


def load_v2_reference(dev: np.ndarray, audit: np.ndarray):
    oof = pd.read_csv(PARENT_V2 / "oof.csv").set_index("row_id").loc[dev]
    with np.load(PARENT_V2 / "ensemble_predictions.npz") as z:
        audit_p = z["audit"]
        test_p = z["test"]
    if len(audit_p) != len(audit):
        raise ValueError("v2 audit reference size mismatch")
    return oof.probability.to_numpy(), audit_p, test_p


def check(config: dict) -> None:
    train, test, y, _, _, dev, audit, folds = load_data()
    counts = {}
    for spec in config["candidates"]:
        fx = build_features(train.iloc[:100], spec["block"])
        tx = build_features(test.iloc[:100], spec["block"])
        if list(fx.columns) != list(tx.columns):
            raise ValueError(f"Schema mismatch for {spec['id']}")
        counts[spec["id"]] = int(fx.shape[1])
    ref, _, _ = load_v2_reference(dev, audit)
    print(json.dumps({
        "check_passed": True, "integrity": verify_parent(), "dev_rows": len(dev), "audit_rows": len(audit),
        "fold_counts": {str(f): int((folds == f).sum()) for f in range(5)}, "feature_counts": counts,
        "v2_reference_rows": len(ref), "primary": config["primary_candidate"],
        "remote_submission_authorized": bool(config["remote_submission_authorized"]),
    }, indent=2), flush=True)


def smoke(config: dict) -> None:
    train, test, y, _, _, dev, _, folds = load_data()
    spec = next(s for s in config["candidates"] if s["id"] == config["primary_candidate"])
    tr_ids = dev[folds != 0][:2500]
    va_ids = dev[folds == 0][:750]
    fx = build_features(train, spec["block"])
    schema = fit_schema(fx.iloc[tr_ids], "catboost")
    params = dict(config)
    params["catboost"] = dict(config["catboost"])
    params["catboost"]["iterations"] = 40
    model = model_for(params, schema, int(config["seed"]))
    started = time.perf_counter()
    model.fit(transform(fx.iloc[tr_ids], schema), y[tr_ids])
    p = model.predict_proba(transform(fx.iloc[va_ids], schema))[:, 1]
    report = {"smoke_passed": True, "train_rows": len(tr_ids), "valid_rows": len(va_ids),
              "features": fx.shape[1], "auc": float(roc_auc_score(y[va_ids], p)),
              "seconds": float(time.perf_counter() - started)}
    write_json(ROOT / "reports" / "v8_interactions_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run(config: dict, resume: bool) -> None:
    if OUT.exists() and not resume:
        raise FileExistsError("v8 artifacts already exist; use --resume or a new version")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config)
    verify_parent()
    train, test, y, sample, manifest, dev, audit, folds = load_data()
    y_dev, y_audit = y[dev], y[audit]
    ref_oof, ref_audit, ref_test = load_v2_reference(dev, audit)

    # Two-fold ablation diagnostics for all three blocks.
    screen_mask = np.isin(folds, np.asarray(config["screen_folds"]))
    screen_rows = []
    for spec in config["candidates"]:
        for fold in config["screen_folds"]:
            fit_fold(config, spec, int(fold), train, test, y, dev, audit, folds, resume)
        d = OUT / "candidates" / spec["id"]
        part = pd.concat([pd.read_csv(d / f"fold{int(f)}_oof.csv") for f in config["screen_folds"]]).set_index("row_id").loc[dev[screen_mask]]
        t, score = select_threshold(part.target.to_numpy(), part.probability.to_numpy(), config)
        screen_rows.append({"candidate": spec["id"], "block": spec["block"], "f1": score, "threshold": t,
                            "auc": float(roc_auc_score(part.target, part.probability))})
    screen = pd.DataFrame(screen_rows).sort_values("f1", ascending=False)
    screen.to_csv(OUT / "screening.csv", index=False)

    # Predeclared primary always receives full five-fold evaluation; no winner-picking here.
    primary = next(s for s in config["candidates"] if s["id"] == config["primary_candidate"])
    for fold in range(5):
        fit_fold(config, primary, fold, train, test, y, dev, audit, folds, resume)
    p, audit_p, test_p, primary_result = assemble(primary["id"], y_dev, dev, folds, config)

    ref_t, ref_f1 = select_threshold(y_dev, ref_oof, config)
    ref_fold_f1 = [float(f1_score(y_dev[folds == f], ref_oof[folds == f] >= ref_t, zero_division=0)) for f in range(5)]
    primary_fold = np.asarray(primary_result["fold_f1"])
    ref_fold = np.asarray(ref_fold_f1)
    nested_primary = nested_threshold(y_dev, p, folds, config)
    nested_ref = nested_threshold(y_dev, ref_oof, folds, config)

    # Audit remains diagnostic and is inspected only after primary recipe is fixed.
    audit_primary = metrics(y_audit, audit_p, primary_result["threshold"])
    audit_ref = metrics(y_audit, ref_audit, ref_t)
    pd.DataFrame({"row_id": dev, "target": y_dev, "dev_fold": folds, "probability": p,
                  "v2_probability": ref_oof}).to_csv(OUT / "primary_oof.csv", index=False)
    pd.DataFrame({"row_id": audit, "target": y_audit, "probability": audit_p,
                  "v2_probability": ref_audit}).to_csv(OUT / "primary_audit.csv", index=False)
    pd.DataFrame({"Id": sample["Id"].to_numpy(), "probability": test_p,
                  "v2_probability": ref_test}).to_csv(OUT / "test_probabilities.csv", index=False)

    result = {
        "version": config["version"], "completed_utc": now(), "primary_candidate": primary["id"],
        "primary": primary_result,
        "v2_reference": {"threshold": ref_t, "dev_oof_f1": ref_f1,
                         "fold_f1": ref_fold_f1, "fold_f1_std": float(np.std(ref_fold, ddof=1))},
        "comparison": {
            "dev_delta_vs_v2": float(primary_result["dev_oof_f1"] - ref_f1),
            "paired_fold_deltas": (primary_fold - ref_fold).tolist(),
            "positive_folds": int(np.sum(primary_fold > ref_fold)),
            "oof_correlation_v2": float(np.corrcoef(p, ref_oof)[0, 1]),
            "nested_threshold_primary_f1": nested_primary["pooled_f1"],
            "nested_threshold_v2_f1": nested_ref["pooled_f1"],
            "nested_delta_vs_v2": float(nested_primary["pooled_f1"] - nested_ref["pooled_f1"]),
        },
        "nested_primary": nested_primary, "nested_v2": nested_ref,
        "audit_primary": audit_primary, "audit_v2": audit_ref,
        "audit_delta": float(audit_primary["f1"] - audit_ref["f1"]),
        "screening": screen_rows, "integrity": verify_parent(), "submitted": False,
        "limitations": [
            "The primary survey_all feature block was predeclared before full five-fold scoring; opinion/context are screening diagnostics only.",
            "The primary threshold is tuned on the same development OOF and remains selection-biased; nested-threshold score is stronger evidence.",
            "The audit set has been reused historically and is diagnostic only.",
            "No leaderboard tuning and no Kaggle submission occur in this run."
        ]
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": config["version"], "id": primary["id"], "parent": "v2/cat_survey",
            "timestamp_utc": now(), "hypothesis": "Public-solution-inspired row-local interactions improve the proven survey representation without changing CatBoost hyperparameters.",
            "validation": "Predeclared survey_all primary on exact frozen 5-fold grouped dev split; nested threshold comparison; reused audit diagnostic only.",
            "result": {"dev_oof_f1": primary_result["dev_oof_f1"], "threshold": primary_result["threshold"],
                       "nested_delta_vs_v2": result["comparison"]["nested_delta_vs_v2"], "audit_f1": audit_primary["f1"]},
            "artifact": "artifacts/v8_interactions", "submitted": False
        }, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "smoke", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v8_interactions.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check(config)
    elif args.phase == "smoke":
        smoke(config)
    else:
        run(config, args.resume)


if __name__ == "__main__":
    main()
