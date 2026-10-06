"""Finalize the frozen v9 context recipe on all 42,154 labeled training rows.

No model/feature/threshold selection occurs here. The v9 recipe is already frozen:
context features, CatBoost(500, depth=6, lr=.05, l2=5), threshold=.31.
The only change versus the submitted v9 CV ensemble is fitting one final model on all
available labels after validation decisions were frozen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from baseline import check_submission
from v2_features import fit_schema, transform
from v8_features import build_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v9_context_confirm.json"
V1 = ROOT / "artifacts" / "baseline_v1"
V9 = ROOT / "artifacts" / "v9_context_confirm"
OUT = ROOT / "artifacts" / "v9_context_full"
TARGET = "vacc_h1n1_f"
THRESHOLD = 0.31


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def verify_contract(config: dict) -> dict:
    protected = read_json(ROOT / "versions" / "v1" / "manifest.json")["sha256"]
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f"Protected v1 files changed: {changed}")
    result = read_json(V9 / "results.json")
    if not bool(result["confirmation_surface"]["passed"]):
        raise ValueError("v9 confirmation did not pass")
    frozen = float(result["selection_surface"]["context_threshold_locked"])
    if not np.isclose(frozen, THRESHOLD, atol=1e-12):
        raise ValueError(f"Frozen v9 threshold changed: {frozen}")
    expected = {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "loss_function": "Logloss"}
    if config["catboost"] != expected:
        raise ValueError(f"v9 CatBoost recipe changed: {config['catboost']}")
    return {"passed": True, "protected_files": len(protected), "threshold": frozen, "catboost": expected}


def main() -> None:
    config = read_json(CONFIG)
    contract = verify_contract(config)
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    sample = pd.read_csv(raw / "submission.csv")
    y = labels[TARGET].to_numpy(dtype=int)
    if len(train) != len(y) or len(test) != len(sample):
        raise ValueError("Input row mismatch")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")

    features = build_features(train, "context")
    test_features = build_features(test, "context")
    if features.shape[1] != 82:
        raise ValueError(f"Unexpected v9 context width: {features.shape[1]}")
    schema = fit_schema(features, "catboost")
    train_x = transform(features, schema)
    test_x = transform(test_features, schema)

    started = time.perf_counter()
    model = CatBoostClassifier(
        **config["catboost"], random_seed=int(config["seed"]),
        thread_count=int(config["threads"]), cat_features=schema["categorical"],
        allow_writing_files=False, verbose=False,
    )
    model.fit(train_x, y)
    test_p = model.predict_proba(test_x)[:, 1]
    if not np.isfinite(test_p).all():
        raise ValueError("Nonfinite test probabilities")

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "v9_context_full42154.joblib"
    if model_path.exists():
        raise FileExistsError(model_path)
    joblib.dump({"model": model, "schema": schema, "recipe": config["catboost"], "threshold": THRESHOLD}, model_path, compress=3)

    prob_path = OUT / "test_probabilities.csv"
    pd.DataFrame({"Id": sample["Id"], "probability": test_p}).to_csv(prob_path, index=False)
    submission = sample.copy()
    submission[TARGET] = (test_p >= THRESHOLD).astype(np.int64)
    submission_path = ROOT / "submissions" / "v9_context_full42154_t031.csv"
    if submission_path.exists():
        raise FileExistsError(submission_path)
    submission.to_csv(submission_path, index=False)
    validation = check_submission(sample, pd.read_csv(submission_path))

    prior_path = ROOT / "submissions" / "v9_context_locked031.csv"
    comparison = None
    if prior_path.is_file():
        prior = pd.read_csv(prior_path)
        old = prior[TARGET].to_numpy(dtype=int)
        new = submission[TARGET].to_numpy(dtype=int)
        comparison = {
            "prior_cv_submission": str(prior_path.relative_to(ROOT)).replace("\\", "/"),
            "prior_positive_count": int(old.sum()),
            "full_positive_count": int(new.sum()),
            "disagree_rows": int(np.sum(old != new)),
            "disagree_rate": float(np.mean(old != new)),
        }

    receipt = {
        "version": "v9_context_full",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_training_rows": int(len(train)),
        "all_available_labels_used": True,
        "test_rows": int(len(test)),
        "features": int(features.shape[1]),
        "recipe_frozen_from_v9": True,
        "threshold": THRESHOLD,
        "training_seconds": float(time.perf_counter() - started),
        "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
        "model_sha256": sha(model_path),
        "submission": {
            "path": str(submission_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha(submission_path),
            **validation,
        },
        "comparison_to_submitted_v9_cv": comparison,
        "contract": contract,
        "submitted": False,
        "note": "This is a full-label finalization only. No retuning, no seed ensemble, and no test-label/pseudo-label use."
    }
    write_json(OUT / "results.json", receipt)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({
            "version": "v9_context_full", "id": "context_full42154_t031", "parent": "v9_context_confirm",
            "timestamp_utc": receipt["completed_utc"],
            "hypothesis": "Using all 42,154 labeled rows for the already-frozen v9 context recipe improves final inference versus the development-only CV ensemble.",
            "validation": "No new validation selection; exact frozen v9 recipe/threshold; full-label refit only.",
            "result": {"training_rows": len(train), "positive_predictions": validation["positive_predictions"],
                       "positive_rate": validation["positive_rate"], "comparison": comparison},
            "artifact": "artifacts/v9_context_full", "submitted": False
        }, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
