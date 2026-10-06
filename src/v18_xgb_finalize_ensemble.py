"""v18: fair fixed-round comparison of v17 finalists, full refit, then simple v12+XGB ensemble.

No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from baseline import check_submission
import v17_xgb_optuna as v17


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v18_xgb_finalize_ensemble"
CONFIRM_PATH = ROOT / "artifacts" / "v17_xgb_optuna" / "confirmation_leaderboard.json"
V12_OOF_PATH = ROOT / "artifacts" / "v12_exact_v5_fullstack" / "selected_oof.csv"
V12_TEST_PATH = ROOT / "artifacts" / "v12_exact_v5_fullstack" / "test_probabilities.csv"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_grid(cfg):
    return np.linspace(float(cfg["threshold_min"]), float(cfg["threshold_max"]), int(cfg["threshold_steps"]))


def choose_threshold(y, p, cfg):
    grid = threshold_grid(cfg)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    ix = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    j = ix[len(ix) // 2]
    return float(grid[j]), float(scores[j])


def nested_metrics(y, p, folds, cfg):
    hard = np.zeros(len(y), dtype=bool)
    thresholds, fold_f1 = [], []
    for fold in range(5):
        tune = folds != fold
        valid = folds == fold
        t, _ = choose_threshold(y[tune], p[tune], cfg)
        thresholds.append(t)
        hard[valid] = p[valid] >= t
        fold_f1.append(float(f1_score(y[valid], hard[valid], zero_division=0)))
    return {
        "nested_f1": float(f1_score(y, hard, zero_division=0)),
        "thresholds": thresholds,
        "median_threshold": float(statistics.median(thresholds)),
        "fold_f1": fold_f1,
        "fold_std": float(np.std(fold_f1, ddof=1)),
    }


def fixed_eval(candidate, rounds, cfg, cache, y, dev, audit, fold_ids):
    trial = int(candidate["trial"])
    mode = candidate["feature_mode"]
    label = f"trial{trial}_r{rounds}"
    jpath = OUT / "fixed_round" / f"{label}.json"
    ppath = OUT / "fixed_round" / f"{label}.npz"
    if jpath.is_file() and ppath.is_file():
        return read_json(jpath)

    params = candidate["params"]
    features = cache[mode][0]
    ydev = y[dev]
    oof = np.full(len(dev), np.nan, dtype=float)
    audit_parts = []
    seconds = []
    for fold in range(5):
        tr_ids = dev[fold_ids != fold]
        va_pos = np.flatnonzero(fold_ids == fold)
        va_ids = dev[va_pos]
        started = time.perf_counter()
        model, transformer, pv, _ = v17.fit_fold(
            params, read_json(ROOT / "configs" / "v17_xgb_optuna.json"),
            features, y, tr_ids, va_ids,
            int(cfg["seed"]) + 9000 + fold,
            int(rounds), False, mode,
        )
        oof[va_pos] = pv
        audit_parts.append(model.predict_proba(v17.transform_representation(features.iloc[audit], transformer))[:, 1])
        seconds.append(time.perf_counter() - started)

    nested = nested_metrics(ydev, oof, fold_ids, cfg)
    tuned_t, tuned_f1 = choose_threshold(ydev, oof, cfg)
    audit_p = np.mean(audit_parts, axis=0)
    result = {
        "trial": trial,
        "feature_mode": mode,
        "rounds": int(rounds),
        "multiplier": float(rounds / candidate["median_best_iteration"]),
        "params": params,
        "nested": nested,
        "tuned_threshold": tuned_t,
        "tuned_f1": tuned_f1,
        "auc": float(roc_auc_score(ydev, oof)),
        "logloss": float(log_loss(ydev, oof, labels=[0, 1])),
        "audit_f1_at_nested_median": float(f1_score(y[audit], audit_p >= nested["median_threshold"], zero_division=0)),
        "fold_seconds": seconds,
    }
    jpath.parent.mkdir(parents=True, exist_ok=True)
    write_json(jpath, result)
    np.savez_compressed(ppath, oof=oof, audit=audit_p)
    return result


def fixed_sort_key(r):
    return (
        r["nested"]["nested_f1"],
        -r["nested"]["fold_std"],
        r["tuned_f1"],
        r["auc"],
        -r["logloss"],
    )


def rank01(p):
    return pd.Series(p).rank(method="average", pct=True).to_numpy(float)


def combine(a, b, w, kind):
    if kind == "prob":
        return w * a + (1.0 - w) * b
    return w * rank01(a) + (1.0 - w) * rank01(b)


def full_refit(best, cfg, cache, train, test, y, sample):
    final_dir = OUT / "final_xgb"
    final_dir.mkdir(parents=True, exist_ok=True)
    mode = best["feature_mode"]
    params = best["params"]
    rounds = int(best["rounds"])
    threshold = float(best["nested"]["median_threshold"])
    features, test_features = cache[mode]
    transformer, xall = v17.fit_transform_representation(features, mode)
    xtest = v17.transform_representation(test_features, transformer)
    v17_cfg = read_json(ROOT / "configs" / "v17_xgb_optuna.json")

    preds, models = [], []
    for seed in cfg["full_seeds"]:
        mpath = final_dir / f"trial{best['trial']}_r{rounds}_seed{seed}.joblib"
        ppath = final_dir / f"test_seed{seed}.npy"
        if mpath.is_file() and ppath.is_file():
            p = np.load(ppath)
            sec = None
        else:
            model = v17.model_from_params(params, v17_cfg, int(seed), rounds, False, mode)
            started = time.perf_counter()
            model.fit(xall, y, verbose=False)
            sec = time.perf_counter() - started
            p = model.predict_proba(xtest)[:, 1]
            joblib.dump({"model": model, "transformer": transformer, "feature_mode": mode, "params": params, "rounds": rounds}, mpath, compress=3)
            np.save(ppath, p)
        preds.append(p)
        models.append({"seed": int(seed), "path": str(mpath.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(mpath), "seconds": sec})

    arr = np.vstack(preds)
    single = arr[0]
    seed5 = arr.mean(axis=0)
    pd.DataFrame({"Id": sample.Id, "probability": single}).to_csv(final_dir / "test_single.csv", index=False)
    pd.DataFrame({"Id": sample.Id, "probability": seed5}).to_csv(final_dir / "test_seed5.csv", index=False)

    candidates = {}
    for name, p in [("single", single), ("seed5", seed5)]:
        sub = sample.copy()
        sub[TARGET] = (p >= threshold).astype(np.int64)
        path = ROOT / "submissions" / f"v18_xgb_topfixed_full_{name}.csv"
        if not path.exists():
            sub.to_csv(path, index=False)
        checked = check_submission(sample, pd.read_csv(path))
        candidates[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), **checked}
    return {"single": single, "seed5": seed5, "models": models, "threshold": threshold, "candidates": candidates,
            "single_vs_seed5_corr": float(np.corrcoef(single, seed5)[0, 1]),
            "single_vs_seed5_hard_disagreement": int(np.sum((single >= threshold) != (seed5 >= threshold)))}


def error_slices(train, y, dev, v12p, xgbp, v12_t, xgb_t):
    ydev = y[dev]
    p12 = v12p >= v12_t
    px = xgbp >= xgb_t
    frame = train.iloc[dev].copy()
    frame.insert(0, "row_id", dev)
    frame["target"] = ydev
    frame["v12_prob"] = v12p
    frame["xgb_prob"] = xgbp
    frame["v12_pred"] = p12.astype(int)
    frame["xgb_pred"] = px.astype(int)
    frame["v12_correct"] = (p12 == ydev).astype(int)
    frame["xgb_correct"] = (px == ydev).astype(int)
    frame["xgb_rescue"] = ((p12 != ydev) & (px == ydev)).astype(int)
    frame["v12_rescue"] = ((px != ydev) & (p12 == ydev)).astype(int)
    frame.to_csv(OUT / "error_disagreements.csv", index=False)

    cols = ["agegrp", "doctor_recc_h1n1", "opinion_h1n1_risk", "opinion_h1n1_vacc_effective",
            "health_worker", "chronic_med_condition", "employment_status", "hhs_region", "state"]
    rows = []
    for col in cols:
        key = frame[col].where(frame[col].notna(), "__MISSING__").astype(str)
        tmp = frame.assign(__group__=key).groupby("__group__", dropna=False)
        for value, g in tmp:
            if len(g) < 50:
                continue
            rows.append({
                "feature": col, "value": str(value), "count": int(len(g)),
                "target_rate": float(g.target.mean()),
                "v12_error_rate": float(1 - g.v12_correct.mean()),
                "xgb_error_rate": float(1 - g.xgb_correct.mean()),
                "xgb_rescues": int(g.xgb_rescue.sum()),
                "v12_rescues": int(g.v12_rescue.sum()),
                "net_xgb_rescue": int(g.xgb_rescue.sum() - g.v12_rescue.sum()),
            })
    pd.DataFrame(rows).sort_values(["net_xgb_rescue", "count"], ascending=[False, False]).to_csv(OUT / "error_slices.csv", index=False)


def check(cfg):
    confirms = read_json(CONFIRM_PATH)
    required = [CONFIRM_PATH, V12_OOF_PATH, V12_TEST_PATH, ROOT / "data/raw/submission.csv"]
    if any(not Path(p).is_file() for p in required):
        raise FileNotFoundError([str(p) for p in required if not Path(p).is_file()])
    train, test, y, sample, manifest, dev, audit, folds = v17.load_data()
    payload = {
        "check_passed": True,
        "confirmed_available": len(confirms),
        "top_k": int(cfg["top_k_confirmed"]),
        "trial_ids": [int(x["trial"]) for x in confirms[:int(cfg["top_k_confirmed"])]],
        "round_multipliers": cfg["round_multipliers"],
        "ensemble_v12_weights": cfg["ensemble_v12_weights"],
        "train_rows": len(train), "dev_rows": len(dev), "audit_rows": len(audit), "test_rows": len(test),
        "full_refit_required": True,
        "remote_submission_authorized": False,
    }
    write_json(ROOT / "reports" / "v18_preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def status(cfg):
    fixed = list((OUT / "fixed_round").glob("*.json")) if (OUT / "fixed_round").exists() else []
    payload = {
        "fixed_round_completed": len(fixed),
        "fixed_round_target": int(cfg["top_k_confirmed"]) * len(cfg["round_multipliers"]),
        "selection_exists": (OUT / "selection.json").is_file(),
        "full_refit_models": len(list((OUT / "final_xgb").glob("*.joblib"))) if (OUT / "final_xgb").exists() else 0,
        "results_exists": (OUT / "results.json").is_file(),
    }
    print(json.dumps(payload, indent=2))


def run(cfg):
    if (OUT / "results.json").is_file():
        print("v18 already complete; immutable results preserved")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "config.json", cfg)
    confirms = read_json(CONFIRM_PATH)[:int(cfg["top_k_confirmed"])]
    train, test, y, sample, manifest, dev, audit, fold_ids = v17.load_data()
    v17_cfg = read_json(ROOT / "configs" / "v17_xgb_optuna.json")
    cache = v17.feature_cache(train, test, v17_cfg["feature_modes"])

    fixed = []
    for rank, cand in enumerate(confirms, 1):
        base = int(cand["median_best_iteration"])
        for mult in cfg["round_multipliers"]:
            rounds = max(20, int(round(base * float(mult))))
            print(f"FIXED trial={cand['trial']} rank={rank}/{len(confirms)} mult={mult} rounds={rounds}", flush=True)
            fixed.append(fixed_eval(cand, rounds, cfg, cache, y, dev, audit, fold_ids))
    fixed.sort(key=fixed_sort_key, reverse=True)
    write_json(OUT / "fixed_round_leaderboard.json", fixed)
    best = fixed[0]
    best_pred_path = OUT / "fixed_round" / f"trial{best['trial']}_r{best['rounds']}.npz"
    best_preds = np.load(best_pred_path)
    xgb_oof = best_preds["oof"]

    selection = {
        "trial": best["trial"], "feature_mode": best["feature_mode"], "params": best["params"],
        "rounds": best["rounds"], "threshold": best["nested"]["median_threshold"],
        "nested_f1": best["nested"]["nested_f1"], "fold_std": best["nested"]["fold_std"],
        "tuned_f1": best["tuned_f1"], "auc": best["auc"], "logloss": best["logloss"],
        "audit_f1": best["audit_f1_at_nested_median"],
    }
    write_json(OUT / "selection.json", selection)

    # Freeze ensemble recipe using development OOF only.
    v12_all = pd.read_csv(V12_OOF_PATH).set_index("row_id")
    v12_dev = v12_all.loc[dev, "probability"].to_numpy(float)
    ydev = y[dev]
    ensemble_rows = []
    for w in cfg["ensemble_v12_weights"]:
        for kind in cfg["ensemble_kinds"]:
            p = combine(v12_dev, xgb_oof, float(w), kind)
            nested = nested_metrics(ydev, p, fold_ids, cfg)
            t, tf1 = choose_threshold(ydev, p, cfg)
            ensemble_rows.append({
                "kind": kind, "v12_weight": float(w), "xgb_weight": float(1 - w),
                "nested": nested, "tuned_threshold": t, "tuned_f1": tf1,
                "auc": float(roc_auc_score(ydev, p)),
            })
    ensemble_rows.sort(key=lambda r: (r["nested"]["nested_f1"], -r["nested"]["fold_std"], r["tuned_f1"], r["auc"]), reverse=True)
    write_json(OUT / "ensemble_scan.json", ensemble_rows)
    ensemble_best = ensemble_rows[0]
    write_json(OUT / "ensemble_selection.json", ensemble_best)

    # Audit/error-slice diagnostics happen only after XGB + ensemble selection is frozen.
    error_slices(train, y, dev, v12_dev, xgb_oof, 0.315, float(selection["threshold"]))

    # Mandatory all-label full refit.
    full = full_refit(best, cfg, cache, train, test, y, sample)

    v12_test = pd.read_csv(V12_TEST_PATH)["probability"].to_numpy(float)
    ens_candidates = {}
    for xgb_name, xgb_test in [("single", full["single"]), ("seed5", full["seed5"])]:
        p = combine(v12_test, xgb_test, float(ensemble_best["v12_weight"]), ensemble_best["kind"])
        threshold = float(ensemble_best["nested"]["median_threshold"])
        pd.DataFrame({"Id": sample.Id, "probability": p}).to_csv(OUT / f"ensemble_test_{xgb_name}.csv", index=False)
        sub = sample.copy()
        sub[TARGET] = (p >= threshold).astype(np.int64)
        path = ROOT / "submissions" / f"v18_v12_xgb_{ensemble_best['kind']}_{int(round(ensemble_best['v12_weight']*100))}_{xgb_name}.csv"
        if not path.exists():
            sub.to_csv(path, index=False)
        checked = check_submission(sample, pd.read_csv(path))
        ens_candidates[xgb_name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path), **checked}

    result = {
        "version": cfg["version"], "completed_utc": now(),
        "fixed_round_candidates": len(fixed), "xgb_selection": selection,
        "ensemble_selection": ensemble_best,
        "full_refit": {"training_rows": len(train), "all_available_labels_used": True,
                       "models": full["models"], "standalone_candidates": full["candidates"],
                       "single_vs_seed5_corr": full["single_vs_seed5_corr"],
                       "single_vs_seed5_hard_disagreement": full["single_vs_seed5_hard_disagreement"]},
        "ensemble_candidates": ens_candidates,
        "submitted": False,
    }
    write_json(OUT / "results.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "run", "status"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v18_xgb_finalize_ensemble.json")
    args = ap.parse_args()
    cfg = read_json(args.config)
    if args.phase == "check":
        check(cfg)
    elif args.phase == "status":
        status(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
