"""Confirm the v8 context-only feature hypothesis on untouched folds 2/3/4.

The context block was selected after inspecting v8 folds 0/1, so those folds are
treated as the selection surface only. Its threshold is locked from folds 0/1 and
evaluated unchanged on folds 2/3/4. v2 receives the same treatment: its threshold
is chosen on folds 0/1 and fixed for folds 2/3/4. This makes the confirmation more
credible than simply retuning both models on all five folds.

Only if the confirmation candidate beats v2 with at least 2/3 positive folds do we
inspect the already-reused audit and create a local submission candidate. No remote
Kaggle submission occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score

from v2_features import fit_schema, transform
from v8_features import build_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v9_context_confirm"
V1 = ROOT / "artifacts" / "baseline_v1"
V2 = ROOT / "artifacts" / "v2" / "candidates" / "cat_survey"
V8 = ROOT / "artifacts" / "v8_interactions" / "candidates" / "survey_context"


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
    expected = read_json(V1 / "metrics.json")
    if sha(V1 / "split_manifest.csv") != expected["split_sha256"]:
        raise ValueError("Frozen split changed")
    return {"passed": True, "protected_files": len(protected)}


def load_data():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    sample = pd.read_csv(raw / "submission.csv")
    manifest = pd.read_csv(V1 / "split_manifest.csv", dtype={"group_hash": str})
    dev = manifest.loc[manifest.partition == "development", "row_id"].to_numpy(dtype=int)
    audit = manifest.loc[manifest.partition == "audit", "row_id"].to_numpy(dtype=int)
    folds = manifest.loc[dev, "dev_fold"].to_numpy(dtype=int)
    return train, test, labels[TARGET].to_numpy(dtype=int), sample, manifest, dev, audit, folds


def grid(config: dict) -> np.ndarray:
    return np.linspace(float(config["threshold_min"]), float(config["threshold_max"]), int(config["threshold_steps"]))


def select_threshold(y: np.ndarray, p: np.ndarray, config: dict) -> tuple[float, float]:
    thresholds = grid(config)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in thresholds])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(thresholds[idx]), float(scores[idx])


def metric_bundle(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "positive_prediction_rate": float(pred.mean()),
    }


def read_fold_oof(base: Path, folds_to_read: list[int]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(base / f"fold{f}_oof.csv") for f in folds_to_read], ignore_index=True)


def v8_selection_receipt(config: dict, dev: np.ndarray, folds: np.ndarray):
    selection_folds = [int(x) for x in config["selection_folds"]]
    mask = np.isin(folds, selection_folds)
    expected_ids = dev[mask]
    part = read_fold_oof(V8, selection_folds).set_index("row_id").loc[expected_ids].reset_index()
    t, score = select_threshold(part.target.to_numpy(), part.probability.to_numpy(), config)
    if not np.isclose(t, float(config["expected_context_selection_threshold"]), atol=1e-12):
        raise ValueError(f"v8 context selection threshold changed: {t}")
    if not np.isclose(score, float(config["expected_context_selection_f1"]), atol=1e-12):
        raise ValueError(f"v8 context selection F1 changed: {score}")
    return part, t, score


def v2_selection_receipt(config: dict, dev: np.ndarray, folds: np.ndarray):
    selection_folds = [int(x) for x in config["selection_folds"]]
    mask = np.isin(folds, selection_folds)
    expected_ids = dev[mask]
    part = read_fold_oof(V2, selection_folds).set_index("row_id").loc[expected_ids].reset_index()
    t, score = select_threshold(part.target.to_numpy(), part.probability.to_numpy(), config)
    return part, t, score


def fold_paths(fold: int):
    d = OUT / "folds"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"fold{fold}.joblib", d / f"fold{fold}_oof.csv", d / f"fold{fold}_predictions.npz", d / f"fold{fold}_metrics.json"


def fit_confirmation_fold(config: dict, fold: int, train: pd.DataFrame, test: pd.DataFrame,
                          y: np.ndarray, dev: np.ndarray, audit: np.ndarray, folds: np.ndarray,
                          resume: bool) -> None:
    model_path, oof_path, pred_path, receipt_path = fold_paths(fold)
    valid_ids = dev[folds == fold]
    train_ids = dev[folds != fold]
    if resume and all(p.is_file() for p in [model_path, oof_path, pred_path, receipt_path]):
        saved = pd.read_csv(oof_path)
        if not np.array_equal(saved.row_id.to_numpy(), valid_ids):
            raise ValueError(f"Resume row mismatch fold {fold}")
        print(f"REUSE context fold={fold}", flush=True)
        return
    if any(p.exists() for p in [model_path, oof_path, pred_path, receipt_path]):
        raise FileExistsError(f"Partial v9 fold {fold} exists; inspect before retry")

    started = time.perf_counter()
    fx = build_features(train, "context")
    tx = build_features(test, "context")
    schema = fit_schema(fx.iloc[train_ids], "catboost")
    model = CatBoostClassifier(
        **config["catboost"], random_seed=int(config["seed"]) + fold,
        thread_count=int(config["threads"]), cat_features=schema["categorical"],
        allow_writing_files=False, verbose=False,
    )
    model.fit(transform(fx.iloc[train_ids], schema), y[train_ids])
    valid_p = model.predict_proba(transform(fx.iloc[valid_ids], schema))[:, 1]
    audit_p = model.predict_proba(transform(fx.iloc[audit], schema))[:, 1]
    test_p = model.predict_proba(transform(tx, schema))[:, 1]
    if not all(np.isfinite(a).all() for a in [valid_p, audit_p, test_p]):
        raise ValueError("Nonfinite probabilities")
    joblib.dump({"model": model, "schema": schema, "fold": fold, "features": "context"}, model_path, compress=3)
    pd.DataFrame({"row_id": valid_ids, "target": y[valid_ids], "probability": valid_p}).to_csv(oof_path, index=False)
    np.savez_compressed(pred_path, audit=audit_p, test=test_p)
    receipt = {"fold": fold, "train_rows": len(train_ids), "valid_rows": len(valid_ids),
               "features": fx.shape[1], "seconds": float(time.perf_counter() - started), "completed_utc": now()}
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def confirmation_predictions(config: dict, dev: np.ndarray, folds: np.ndarray):
    confirm = [int(x) for x in config["confirmation_folds"]]
    mask = np.isin(folds, confirm)
    expected = dev[mask]
    part = read_fold_oof(OUT / "folds", confirm).set_index("row_id").loc[expected].reset_index()
    return mask, part


def bootstrap_fixed_threshold_delta(y: np.ndarray, base_p: np.ndarray, cand_p: np.ndarray,
                                    base_t: float, cand_t: float, groups: np.ndarray,
                                    repeats: int, seed: int) -> dict:
    base_pred = base_p >= base_t
    cand_pred = cand_p >= cand_t
    _, inverse = np.unique(groups, return_inverse=True)
    n = int(inverse.max()) + 1

    def counts(pred):
        return np.column_stack([
            np.bincount(inverse, weights=((y == 1) & pred).astype(float), minlength=n),
            np.bincount(inverse, weights=((y == 0) & pred).astype(float), minlength=n),
            np.bincount(inverse, weights=((y == 1) & ~pred).astype(float), minlength=n),
        ])

    b, c = counts(base_pred), counts(cand_pred)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=float)
    for i in range(repeats):
        ix = rng.integers(0, n, size=n)
        bc, cc = b[ix].sum(axis=0), c[ix].sum(axis=0)
        bf = 2 * bc[0] / max(1.0, 2 * bc[0] + bc[1] + bc[2])
        cf = 2 * cc[0] / max(1.0, 2 * cc[0] + cc[1] + cc[2])
        deltas[i] = cf - bf
    return {
        "groups": n, "repeats": repeats,
        "delta_95pct_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "positive_fraction": float(np.mean(deltas > 0)),
        "limitation": "Conditional on the frozen candidate, thresholds and confirmation split."
    }


def full_context_oof(config: dict, y_dev: np.ndarray, dev: np.ndarray, folds: np.ndarray):
    parts = []
    for fold in range(5):
        if fold in [int(x) for x in config["selection_folds"]]:
            src = V8 / f"fold{fold}_oof.csv"
        else:
            src = OUT / "folds" / f"fold{fold}_oof.csv"
        parts.append(pd.read_csv(src))
    frame = pd.concat(parts).set_index("row_id").loc[dev].reset_index()
    p = frame.probability.to_numpy()
    t, f1 = select_threshold(y_dev, p, config)
    fold_f1 = [float(f1_score(y_dev[folds == f], p[folds == f] >= t, zero_division=0)) for f in range(5)]
    return frame, p, {"posthoc_threshold": t, "posthoc_dev_oof_f1": f1,
                      "fold_f1": fold_f1, "fold_f1_std": float(np.std(fold_f1, ddof=1))}


def average_audit_test(config: dict):
    audit_parts, test_parts = [], []
    for fold in range(5):
        if fold in [int(x) for x in config["selection_folds"]]:
            src = V8 / f"fold{fold}_predictions.npz"
        else:
            src = OUT / "folds" / f"fold{fold}_predictions.npz"
        with np.load(src) as z:
            audit_parts.append(z["audit"])
            test_parts.append(z["test"])
    return np.mean(np.vstack(audit_parts), axis=0), np.mean(np.vstack(test_parts), axis=0)


def check(config: dict) -> None:
    train, test, y, _, _, dev, audit, folds = load_data()
    v8_part, v8_t, v8_f1 = v8_selection_receipt(config, dev, folds)
    v2_part, v2_t, v2_f1 = v2_selection_receipt(config, dev, folds)
    fx = build_features(train.iloc[:100], "context")
    tx = build_features(test.iloc[:100], "context")
    if list(fx.columns) != list(tx.columns):
        raise ValueError("Context feature schema mismatch")
    print(json.dumps({
        "check_passed": True, "integrity": verify_parent(), "features": fx.shape[1],
        "selection_folds": config["selection_folds"], "confirmation_folds": config["confirmation_folds"],
        "context_selection": {"rows": len(v8_part), "threshold": v8_t, "f1": v8_f1},
        "v2_selection": {"rows": len(v2_part), "threshold": v2_t, "f1": v2_f1},
        "audit_rows": len(audit), "remote_submission_authorized": bool(config["remote_submission_authorized"]),
    }, indent=2), flush=True)


def run(config: dict, resume: bool) -> None:
    if OUT.exists() and not resume:
        raise FileExistsError("v9 output exists; use --resume or new version")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", config)
    verify_parent()
    train, test, y, sample, manifest, dev, audit, folds = load_data()
    y_dev = y[dev]

    _, cand_t, cand_selection_f1 = v8_selection_receipt(config, dev, folds)
    _, base_t, base_selection_f1 = v2_selection_receipt(config, dev, folds)

    for fold in [int(x) for x in config["confirmation_folds"]]:
        fit_confirmation_fold(config, fold, train, test, y, dev, audit, folds, resume)

    confirm_mask, cand_confirm = confirmation_predictions(config, dev, folds)
    confirm_ids = dev[confirm_mask]
    base_all = pd.read_csv(V2 / "oof.csv").set_index("row_id").loc[confirm_ids].reset_index()
    if not np.array_equal(cand_confirm.row_id.to_numpy(), base_all.row_id.to_numpy()):
        raise ValueError("Confirmation alignment mismatch")
    y_confirm = y[confirm_ids]
    cand_p = cand_confirm.probability.to_numpy()
    base_p = base_all.probability.to_numpy()
    candidate_metrics = metric_bundle(y_confirm, cand_p, cand_t)
    base_metrics = metric_bundle(y_confirm, base_p, base_t)

    confirm_folds = folds[confirm_mask]
    cand_fold_f1, base_fold_f1 = [], []
    for f in [int(x) for x in config["confirmation_folds"]]:
        m = confirm_folds == f
        cand_fold_f1.append(float(f1_score(y_confirm[m], cand_p[m] >= cand_t, zero_division=0)))
        base_fold_f1.append(float(f1_score(y_confirm[m], base_p[m] >= base_t, zero_division=0)))
    fold_delta = np.asarray(cand_fold_f1) - np.asarray(base_fold_f1)
    confirm_delta = candidate_metrics["f1"] - base_metrics["f1"]
    confirmation_pass = bool(confirm_delta > 0 and np.sum(fold_delta > 0) >= 2)
    bootstrap = bootstrap_fixed_threshold_delta(
        y_confirm, base_p, cand_p, base_t, cand_t,
        manifest.loc[confirm_ids, "group_hash"].to_numpy(), int(config["bootstrap_repeats"]), 20261005,
    )

    full_frame, full_p, full_posthoc = full_context_oof(config, y_dev, dev, folds)
    full_frame.to_csv(OUT / "full_oof.csv", index=False)
    full_posthoc["correlation_v2"] = float(np.corrcoef(full_p, pd.read_csv(V2 / "oof.csv").set_index("row_id").loc[dev, "probability"].to_numpy())[0, 1])

    result = {
        "version": config["version"], "completed_utc": now(),
        "selection_surface": {
            "folds": config["selection_folds"],
            "context_threshold_locked": cand_t, "context_selection_f1": cand_selection_f1,
            "v2_threshold_locked": base_t, "v2_selection_f1": base_selection_f1,
        },
        "confirmation_surface": {
            "folds": config["confirmation_folds"], "rows": int(confirm_mask.sum()),
            "context": candidate_metrics, "v2": base_metrics,
            "f1_delta_vs_v2": float(confirm_delta),
            "context_fold_f1": cand_fold_f1, "v2_fold_f1": base_fold_f1,
            "paired_fold_deltas": fold_delta.tolist(), "positive_folds": int(np.sum(fold_delta > 0)),
            "bootstrap": bootstrap, "passed": confirmation_pass,
        },
        "full_oof_posthoc": full_posthoc,
        "integrity": verify_parent(), "submitted": False,
    }

    if confirmation_pass:
        audit_p, test_p = average_audit_test(config)
        y_audit = y[audit]
        with np.load(V2 / "ensemble_predictions.npz") as z:
            v2_audit = z["audit"]
        result["audit_diagnostic"] = {
            "context": metric_bundle(y_audit, audit_p, cand_t),
            "v2": metric_bundle(y_audit, v2_audit, base_t),
        }
        result["audit_diagnostic"]["f1_delta_vs_v2"] = result["audit_diagnostic"]["context"]["f1"] - result["audit_diagnostic"]["v2"]["f1"]
        pd.DataFrame({"row_id": audit, "target": y_audit, "probability": audit_p}).to_csv(OUT / "audit_probabilities.csv", index=False)
        pd.DataFrame({"Id": sample["Id"].to_numpy(), "probability": test_p}).to_csv(OUT / "test_probabilities.csv", index=False)
        submission = pd.DataFrame({"Id": sample["Id"].to_numpy(), TARGET: (test_p >= cand_t).astype(np.int64)})
        sub_path = ROOT / "submissions" / "v9_context_locked031.csv"
        submission.to_csv(sub_path, index=False)
        result["local_submission_candidate"] = {
            "path": str(sub_path.relative_to(ROOT)).replace("\\", "/"),
            "threshold": cand_t, "positive_count": int(submission[TARGET].sum()),
            "positive_rate": float(submission[TARGET].mean()), "sha256": sha(sub_path),
            "submitted": False,
        }
    else:
        result["audit_diagnostic"] = {"skipped": True, "reason": "Confirmation gate failed; avoid additional audit peeking."}
        result["local_submission_candidate"] = None

    result["limitations"] = [
        "The context recipe was selected after v8 folds 0/1; only folds 2/3/4 are treated as confirmation evidence.",
        "Thresholds for both context and v2 are locked using folds 0/1 before confirmation scoring.",
        "The group bootstrap is conditional on this frozen candidate and confirmation split.",
        "Full five-fold OOF is post-selection descriptive evidence, not an independent confirmation surface.",
        "Audit is already reused historically and is inspected only if the confirmation gate passes.",
        "No Kaggle submission occurs in this run."
    ]
    write_json(OUT / "results.json", result)
    pd.DataFrame({"row_id": confirm_ids, "target": y_confirm, "dev_fold": confirm_folds,
                  "context_probability": cand_p, "v2_probability": base_p}).to_csv(OUT / "confirmation_predictions.csv", index=False)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": config["version"], "id": "survey_context_confirm", "parent": "v2/cat_survey",
            "timestamp_utc": now(), "hypothesis": "The v8 context-only feature block generalizes beyond the two folds used to discover it.",
            "validation": "Candidate and thresholds frozen on folds 0/1; confirmation on untouched folds 2/3/4 using fixed thresholds; grouped bootstrap; reused audit only after pass.",
            "result": {"confirmation_f1": candidate_metrics["f1"], "v2_confirmation_f1": base_metrics["f1"],
                       "delta": confirm_delta, "positive_folds": int(np.sum(fold_delta > 0)), "passed": confirmation_pass},
            "artifact": "artifacts/v9_context_confirm", "submitted": False
        }, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "run"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v9_context_confirm.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    config = read_json(args.config)
    if args.phase == "check":
        check(config)
    else:
        run(config, args.resume)


if __name__ == "__main__":
    main()
