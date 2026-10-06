"""Focused grouped-5 confirmation for EBM and RealMLP-TD only.

Reuses v21 preprocessing/evaluation code, writes to a separate artifact folder,
computes OOF diversity against existing anchors, and stops. No full refit and no
Kaggle submission are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from v21_remaining_suite import (
    ROOT,
    load_data,
    read_json,
    write_json,
    evaluate_family,
    existing_anchors,
)


OUT = ROOT / "artifacts" / "v21_focus_confirm"


def now():
    return datetime.now(timezone.utc).isoformat()


def run(cfg):
    result_path = OUT / "results.json"
    if result_path.is_file():
        print("v21 focus confirm already complete; immutable results preserved", flush=True)
        return

    train, test, y, sample, dev, audit, folds = load_data()
    wanted = ["ebm", "realmlp_td"]
    specs = {x["id"]: x for x in cfg["families"] if x["id"] in wanted}
    missing = [x for x in wanted if x not in specs]
    if missing:
        raise RuntimeError(f"missing family specs: {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", {
        "source_config": "configs/v21_remaining_suite.json",
        "families": wanted,
        "folds": cfg["full_folds"],
        "full_refit": False,
        "remote_submission_authorized": False,
    })

    confirmed = []
    for fam in wanted:
        print(f"CONFIRM {fam}", flush=True)
        r = evaluate_family(
            specs[fam], cfg, train, test, y, dev, audit, folds,
            cfg["full_folds"], fam, OUT / "confirm"
        )
        confirmed.append(r)

    confirmed.sort(
        key=lambda r: (
            r["nested"]["nested_f1"],
            -r["nested"]["fold_std"],
            r["tuned_f1"],
            r["auc"],
        ),
        reverse=True,
    )

    anchors = existing_anchors(dev)
    diversity = []
    for r in confirmed:
        p = np.load(OUT / "confirm" / f"{r['family']}.npz")["oof"]
        diversity.append({
            "family": r["family"],
            "corr_v12": float(np.corrcoef(p, anchors["v12"])[0, 1]),
            "corr_xgb": float(np.corrcoef(p, anchors["xgb"])[0, 1]),
            "corr_lgb": float(np.corrcoef(p, anchors["lgb"])[0, 1]),
            "corr_tabm": float(np.corrcoef(p, anchors["tabm"])[0, 1]),
        })

    result = {
        "version": "v21_focus_confirm",
        "completed_utc": now(),
        "families": wanted,
        "confirmed": confirmed,
        "diversity": diversity,
        "full_refit": False,
        "submitted": False,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def status(cfg):
    done = []
    for fam in ["ebm", "realmlp_td"]:
        p = OUT / "confirm" / f"{fam}.json"
        if p.is_file():
            r = read_json(p)
            done.append({
                "family": fam,
                "nested_f1": r["nested"]["nested_f1"],
                "fold_std": r["nested"]["fold_std"],
                "auc": r["auc"],
            })
    print(json.dumps({
        "completed_families": len(done),
        "target_families": 2,
        "done": done,
        "results": (OUT / "results.json").is_file(),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "status"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v21_remaining_suite.json")
    a = ap.parse_args()
    cfg = read_json(a.config)
    if a.phase == "status":
        status(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
