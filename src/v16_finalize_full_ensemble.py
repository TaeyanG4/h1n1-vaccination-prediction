"""Build the full-data analogue of v16's best simple OOF ensemble.

Components are fixed from the v16 scan:
- v2 full cat_survey probability
- v12 exact44 full CatBoost_BAG_L2 probability
- v14 full survey_freq_missing CatBoost probability

Weights are equal thirds and threshold 0.33 is frozen from dev OOF v16.
No remote submission occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import check_submission

ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v16_full_ensemble"
THRESHOLD = 0.33


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(ROOT / "data" / "raw" / "submission.csv")
    sources = {
        "v2_cat_full": ROOT / "artifacts" / "v2" / "final" / "test_probabilities.csv",
        "v12_exact44_full": ROOT / "artifacts" / "v12_exact_v5_fullstack" / "test_probabilities.csv",
        "v14_cat_full": ROOT / "artifacts" / "v14_catboost_feature_hpo" / "selected_test_probabilities.csv",
    }
    arrays = {}
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "Id" not in frame or "probability" not in frame:
            raise ValueError(f"Bad probability file schema: {path}")
        if not frame["Id"].equals(sample["Id"]):
            raise ValueError(f"ID/order mismatch: {path}")
        p = frame["probability"].to_numpy(float)
        if len(p) != len(sample) or not np.isfinite(p).all():
            raise ValueError(f"Invalid probabilities: {path}")
        arrays[name] = p

    prob = np.mean(np.vstack(list(arrays.values())), axis=0)
    prob_path = OUT / "test_probabilities.csv"
    pd.DataFrame({"Id": sample.Id, "probability": prob}).to_csv(prob_path, index=False)

    sub = sample.copy()
    sub[TARGET] = (prob >= THRESHOLD).astype(np.int64)
    sub_path = ROOT / "submissions" / "v16_full_equal_v2_v12_v14_t033.csv"
    if sub_path.exists():
        raise FileExistsError(sub_path)
    sub.to_csv(sub_path, index=False)
    checked = check_submission(sample, pd.read_csv(sub_path))

    result = {
        "version": "v16_full_ensemble",
        "components": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in sources.items()},
        "weights": {k: 1.0 / 3.0 for k in sources},
        "frozen_threshold": THRESHOLD,
        "selection_source": "v16 dev OOF simple equal-probability scan; no optimized blending",
        "submission": {
            "path": str(sub_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha(sub_path),
            **checked,
        },
        "component_correlation": pd.DataFrame(arrays).corr().to_dict(),
        "submitted": False,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
