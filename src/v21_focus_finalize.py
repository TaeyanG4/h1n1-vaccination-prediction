"""Finalize EBM + RealMLP-TD after grouped-5 confirmation.

Creates all evidence-backed OOF-improving probability blends, plus standalone
information candidates, using true all-label refits. Both single-seed and 3-seed
averages are generated. Performs local submission dry-runs only; never uploads.
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

from baseline import check_submission
from v21_remaining_suite import (
    ROOT, load_data, read_json, write_json, build_block, fit_cleaner,
    apply_cleaner, make_estimator, rank01, nested_metrics, choose_threshold,
    existing_anchors,
)


OUT = ROOT / "artifacts" / "v21_focus_finalize"
FOCUS = ROOT / "artifacts" / "v21_focus_confirm"
TARGET = "vacc_h1n1_f"
SEEDS = [42, 2026, 31415]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_fit_one(family, block, cfg, train, test, y, seed):
    tr0 = build_block(train.reset_index(drop=True), block)
    te0 = build_block(test.reset_index(drop=True), block)
    cleaner = fit_cleaner(tr0)
    tr = apply_cleaner(tr0, cleaner)
    te = apply_cleaner(te0, cleaner)
    cats = cleaner["cats"]
    model = make_estimator(family, int(seed), cfg, True)
    started = time.perf_counter()
    if family == "ebm":
        types = ["nominal" if c in set(cats) else "continuous" for c in cleaner["columns"]]
        model.set_params(feature_names=cleaner["columns"], feature_types=types)
        model.fit(tr, y)
    elif family == "realmlp_td":
        model.fit(tr, y, cat_col_names=cats)
    else:
        raise ValueError(family)
    p = model.predict_proba(te)[:, 1]
    return p, model, cleaner, time.perf_counter() - started


def make_candidate(sample, name, p, threshold):
    path = ROOT / "submissions" / name
    sub = sample.copy()
    sub[TARGET] = (p >= threshold).astype(np.int64)
    sub.to_csv(path, index=False)
    ck = check_submission(sample, pd.read_csv(path))
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "threshold": float(threshold),
        "sha256": sha(path),
        **ck,
    }


def scan_oof(cfg, ydev, folds, anchors, ebm, mlp):
    base = .6 * anchors["v12"] + .3 * anchors["xgb"] + .1 * anchors["lgb"]
    base_nested = nested_metrics(ydev, base, folds, [0,1,2,3,4], cfg)
    base_t, base_tf = choose_threshold(ydev, base, cfg)
    rows = []
    def add(name, wb, we, wm):
        q = wb * base + we * ebm + wm * mlp
        n = nested_metrics(ydev, q, folds, [0,1,2,3,4], cfg)
        t, tf = choose_threshold(ydev, q, cfg)
        rows.append({
            "name": name, "base_weight": wb, "ebm_weight": we,
            "realmlp_weight": wm, "nested": n, "tuned_threshold": t,
            "tuned_f1": tf,
        })
    for w in [.05,.10,.15,.20,.25]:
        add(f"base_ebm_{int(w*100)}", 1-w, w, 0.0)
        add(f"base_realmlp_{int(w*100)}", 1-w, 0.0, w)
    for we in [.05,.10,.15]:
        for wm in [.05,.10,.15]:
            if we + wm <= .25:
                add(f"base_ebm_{int(we*100)}_realmlp_{int(wm*100)}", 1-we-wm, we, wm)
    rows.sort(key=lambda r:(r["nested"]["nested_f1"], -r["nested"]["fold_std"], r["tuned_f1"]), reverse=True)
    winners = [r for r in rows if r["nested"]["nested_f1"] > base_nested["nested_f1"] + 1e-12]
    return {
        "baseline": {"nested": base_nested, "tuned_threshold": base_t, "tuned_f1": base_tf},
        "rows": rows,
        "winners": winners,
    }


def dry_run(sample, candidates):
    rows=[]
    for key,c in candidates.items():
        p=ROOT/c["path"]
        df=pd.read_csv(p)
        ok=(list(df.columns)==list(sample.columns) and len(df)==len(sample)
            and df.Id.equals(sample.Id) and df.isna().sum().sum()==0
            and set(df[TARGET].unique()) <= {0,1})
        rows.append({
            "candidate_id":key,"path":c["path"],"passed":bool(ok),
            "rows":len(df),"ids_exact":bool(df.Id.equals(sample.Id)),
            "missing":int(df.isna().sum().sum()),
            "positives":int(df[TARGET].sum()),
            "positive_rate":float(df[TARGET].mean()),
            "sha256":sha(p),"uploaded":False,
        })
    return {"dry_run_passed":all(x["passed"] for x in rows),"candidates":rows,
            "fresh_post_dry_run_approval_required":True}


def main():
    cfg = read_json(ROOT / "configs" / "v21_remaining_suite.json")
    if (OUT / "results.json").is_file():
        print("v21 focus finalize already complete; immutable results preserved")
        return
    train,test,y,sample,dev,audit,folds=load_data(); yd=y[dev]
    conf=read_json(FOCUS/"results.json")
    cmap={r["family"]:r for r in conf["confirmed"]}
    ebm_oof=np.load(FOCUS/"confirm"/"ebm.npz")["oof"]
    mlp_oof=np.load(FOCUS/"confirm"/"realmlp_td.npz")["oof"]
    anchors=existing_anchors(dev)
    scan=scan_oof(cfg,yd,folds,anchors,ebm_oof,mlp_oof)
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/"ensemble_scan.json",scan)
    print("OOF winners:", flush=True)
    for r in scan["winners"]: print(r["name"],r["nested"]["nested_f1"],r["nested"]["median_threshold"],flush=True)

    full={}
    for family,block in [("ebm","survey_native"),("realmlp_td","survey_native")]:
        preds=[]; models=[]
        d=OUT/"full"/family; d.mkdir(parents=True,exist_ok=True)
        for seed in SEEDS:
            pp=d/f"test_seed{seed}.npy"; mp=d/f"model_seed{seed}.joblib"
            if pp.is_file():
                p=np.load(pp); sec=None
            else:
                print(f"FULL {family} seed={seed}",flush=True)
                p,m,cleaner,sec=full_fit_one(family,block,cfg,train,test,y,seed)
                np.save(pp,p)
                try: joblib.dump({"model":m,"cleaner":cleaner,"family":family,"seed":seed},mp,compress=3)
                except Exception: pass
            preds.append(p); models.append({"seed":seed,"seconds":sec,"prediction_path":str(pp.relative_to(ROOT)).replace("\\","/")})
        arr=np.vstack(preds); single=arr[0]; seed3=arr.mean(axis=0)
        np.save(d/"test_single.npy",single); np.save(d/"test_seed3.npy",seed3)
        full[family]={"single":single,"seed3":seed3,"models":models}

    # Existing full test anchor: v12 + v18 XGB single + v19 LGBM seed5.
    v12t=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/test_probabilities.csv").probability.to_numpy(float)
    xgbt=pd.read_csv(ROOT/"artifacts/v18_xgb_finalize_ensemble/final_xgb/test_single.csv").probability.to_numpy(float)
    lgbt=pd.read_csv(ROOT/"artifacts/v19_lightgbm/final/test_seed5.csv").probability.to_numpy(float)
    baset=.6*v12t+.3*xgbt+.1*lgbt

    candidates={}
    # Standalone information candidates.
    for lane in ["single","seed3"]:
        candidates[f"ebm_{lane}"]=make_candidate(sample,f"v21_ebm_full_{lane}.csv",full["ebm"][lane],float(cmap["ebm"]["nested"]["median_threshold"]))
        candidates[f"realmlp_{lane}"]=make_candidate(sample,f"v21_realmlp_td_full_{lane}.csv",full["realmlp_td"][lane],float(cmap["realmlp_td"]["nested"]["median_threshold"]))

    # Every OOF recipe that strictly beat the probability-base nested F1.
    for r in scan["winners"]:
        for lane in ["single","seed3"]:
            p=r["base_weight"]*baset+r["ebm_weight"]*full["ebm"][lane]+r["realmlp_weight"]*full["realmlp_td"][lane]
            fn=f"v21_{r['name']}_{lane}.csv"
            candidates[f"{r['name']}_{lane}"]=make_candidate(sample,fn,p,float(r["nested"]["median_threshold"]))

    report=dry_run(sample,candidates)
    write_json(ROOT/"reports"/"v21_focus_submission_dry_run.json",report)
    serial_full={k:{"models":v["models"]} for k,v in full.items()}
    result={"version":"v21_focus_finalize","completed_utc":now(),"confirmed":conf["confirmed"],"diversity":conf["diversity"],"scan":scan,"full_refit":serial_full,"candidates":candidates,"dry_run":report,"submitted":False}
    write_json(OUT/"results.json",result)
    print(json.dumps({"completed":True,"num_oof_winners":len(scan["winners"]),"num_candidates":len(candidates),"dry_run":report},indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
