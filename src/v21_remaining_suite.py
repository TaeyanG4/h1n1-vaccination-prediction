"""One-command remaining high-value tabular research suite.

Families: RealMLP, RealMLP-TD-S, ExtraTrees, EBM, TabR, xRFM, TabPFN.
Every family is screened on frozen grouped folds first. Survivors receive exact
grouped-5 OOF evaluation. Only families with a correct all-label refit path are
eligible for final submission candidates. No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from baseline import check_submission
from v2_features import build_features, categorical_columns
from v17_xgb_optuna import historical_features
from v19_lightgbm import add_missing


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v21_remaining_suite"
PARENT = ROOT / "artifacts" / "baseline_v1"
TARGET = "vacc_h1n1_f"


def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_data():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    y = pd.read_csv(raw / "train_labels.csv")[TARGET].to_numpy(np.int64)
    sample = pd.read_csv(raw / "submission.csv")
    man = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    dev = man.loc[man.partition.eq("development"), "row_id"].to_numpy(int)
    audit = man.loc[man.partition.eq("audit"), "row_id"].to_numpy(int)
    folds = man.loc[dev, "dev_fold"].to_numpy(int)
    return train, test, y, sample, dev, audit, folds


def build_block(raw, block):
    if block == "raw_native": return build_features(raw, "raw")
    if block == "survey_native": return build_features(raw, "survey")
    if block == "historical_missing": return add_missing(historical_features(raw), raw)
    raise ValueError(block)


def fit_cleaner(frame):
    cats0 = set(categorical_columns(frame))
    cats = [c for c in frame.columns if c in cats0 or frame[c].dtype == object or str(frame[c].dtype).startswith("category")]
    nums = [c for c in frame.columns if c not in set(cats)]
    med = {}
    for c in nums:
        s = pd.to_numeric(frame[c], errors="raise").astype(float)
        med[c] = float(s.median()) if s.notna().any() else 0.0
    return {"columns": list(frame.columns), "cats": cats, "nums": nums, "medians": med}


def apply_cleaner(frame, cleaner):
    if list(frame.columns) != cleaner["columns"]: raise ValueError("schema mismatch")
    z = frame.copy()
    for c in cleaner["cats"]: z[c] = z[c].where(z[c].notna(), "__MISSING__").astype(str)
    for c in cleaner["nums"]: z[c] = pd.to_numeric(z[c], errors="raise").astype(float).fillna(cleaner["medians"][c])
    return z


def fit_ordinal(frame, cleaner):
    maps = {}
    for c in cleaner["cats"]:
        levels = sorted(frame[c].astype(str).unique().tolist())
        maps[c] = {v:i for i,v in enumerate(levels)}
    return maps


def ordinal_matrix(frame, cleaner, maps):
    cols=[]
    for c in cleaner["columns"]:
        if c in maps: cols.append(frame[c].astype(str).map(maps[c]).fillna(-1).to_numpy(float))
        else: cols.append(frame[c].to_numpy(float))
    return np.column_stack(cols).astype(np.float32)


def threshold_grid(cfg): return np.linspace(cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_steps"])
def choose_threshold(y,p,cfg):
    g=threshold_grid(cfg); s=np.array([f1_score(y,p>=t,zero_division=0) for t in g]); ix=np.flatnonzero(np.isclose(s,s.max(),atol=1e-12,rtol=0)); j=ix[len(ix)//2]
    return float(g[j]),float(s[j])


def nested_metrics(y,p,folds,used_folds,cfg):
    used=np.isin(folds,used_folds)&np.isfinite(p); hard=np.zeros(len(y),dtype=bool); ts=[]; fs=[]
    for f in used_folds:
        valid=folds==f; tune=used&(folds!=f); t,_=choose_threshold(y[tune],p[tune],cfg); ts.append(t); hard[valid]=p[valid]>=t; fs.append(float(f1_score(y[valid],hard[valid],zero_division=0)))
    return {"nested_f1":float(f1_score(y[used],hard[used],zero_division=0)),"thresholds":ts,"median_threshold":float(statistics.median(ts)),"fold_f1":fs,"fold_std":float(np.std(fs,ddof=1)) if len(fs)>1 else 0.0}


def make_estimator(family, seed, cfg, full_refit=False):
    if family == "realmlp_td":
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
        return RealMLP_TD_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=1 if full_refit else 0, n_threads=cfg["threads"], verbosity=0)
    if family == "realmlp_tds":
        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_S_Classifier
        return RealMLP_TD_S_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=1 if full_refit else 0, n_threads=cfg["threads"], verbosity=0)
    if family == "tabr":
        from pytabkit.models.sklearn.sklearn_interfaces import TabR_S_D_Classifier
        return TabR_S_D_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=0, n_threads=cfg["threads"], verbosity=0, context_size=96, patience=12, n_epochs=100000, eval_batch_size=4096)
    if family == "xrfm":
        from pytabkit.models.sklearn.sklearn_interfaces import XRFM_D_Classifier
        return XRFM_D_Classifier(device="cuda", random_state=seed, n_cv=1, n_refit=1 if full_refit else 0, n_threads=cfg["threads"], verbosity=0, iters=5, time_limit_s=240)
    if family == "extratrees":
        return ExtraTreesClassifier(n_estimators=1200, max_features=0.75, min_samples_leaf=2, n_jobs=cfg["threads"], random_state=seed)
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(interactions="3x", outer_bags=10, learning_rate=0.02, max_rounds=15000, early_stopping_rounds=120, n_jobs=cfg["threads"], random_state=seed)
    if family == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(n_estimators=4, device="cuda", random_state=seed, fit_mode="fit_preprocessors", show_progress_bar=False)
    raise ValueError(family)


def full_refit_supported(family):
    return family in {"realmlp_td","realmlp_tds","extratrees","ebm","xrfm","tabpfn"}


def fit_predict_one(family, block, cfg, train, test, y, tr_ids, va_ids, audit_ids, seed):
    tr0=build_block(train.iloc[tr_ids].reset_index(drop=True),block); va0=build_block(train.iloc[va_ids].reset_index(drop=True),block); au0=build_block(train.iloc[audit_ids].reset_index(drop=True),block); te0=build_block(test.reset_index(drop=True),block)
    cleaner=fit_cleaner(tr0); tr=apply_cleaner(tr0,cleaner); va=apply_cleaner(va0,cleaner); au=apply_cleaner(au0,cleaner); te=apply_cleaner(te0,cleaner); cats=cleaner["cats"]
    model=make_estimator(family,seed,cfg,False); started=time.perf_counter()
    if family == "extratrees":
        maps=fit_ordinal(tr,cleaner); xtr=ordinal_matrix(tr,cleaner,maps); xva=ordinal_matrix(va,cleaner,maps); xau=ordinal_matrix(au,cleaner,maps); xte=ordinal_matrix(te,cleaner,maps); model.fit(xtr,y[tr_ids]); pv=model.predict_proba(xva)[:,1]; pa=model.predict_proba(xau)[:,1]; pt=model.predict_proba(xte)[:,1]
    elif family == "ebm":
        types=["nominal" if c in set(cats) else "continuous" for c in cleaner["columns"]]; model.set_params(feature_names=cleaner["columns"],feature_types=types); model.fit(tr,y[tr_ids]); pv=model.predict_proba(va)[:,1]; pa=model.predict_proba(au)[:,1]; pt=model.predict_proba(te)[:,1]
    elif family == "tabpfn":
        cat_idx=[tr.columns.get_loc(c) for c in cats]; model.set_params(categorical_features_indices=cat_idx); model.fit(tr,y[tr_ids]); pv=model.predict_proba(va)[:,1]; pa=model.predict_proba(au)[:,1]; pt=model.predict_proba(te)[:,1]
    else:
        model.fit(tr,y[tr_ids],X_val=va,y_val=y[va_ids],cat_col_names=cats); pv=model.predict_proba(va)[:,1]; pa=model.predict_proba(au)[:,1]; pt=model.predict_proba(te)[:,1]
    return pv,pa,pt,time.perf_counter()-started


def evaluate_family(spec,cfg,train,test,y,dev,audit,folds,used_folds,label,out_dir):
    jp=out_dir/f"{label}.json"; npz=out_dir/f"{label}.npz"
    if jp.is_file() and npz.is_file(): return read_json(jp)
    oof=np.full(len(dev),np.nan); aps=[]; tps=[]; secs=[]
    for f in used_folds:
        tri=dev[folds!=f]; pos=np.flatnonzero(folds==f); vai=dev[pos]
        print(f"{label} fold={f}",flush=True)
        pv,pa,pt,sec=fit_predict_one(spec["id"],spec["block"],cfg,train,test,y,tri,vai,audit,int(cfg["seed"])+f*101+sum(map(ord,spec["id"])))
        oof[pos]=pv; aps.append(pa); tps.append(pt); secs.append(sec)
    yd=y[dev]; mask=np.isfinite(oof); nested=nested_metrics(yd,oof,folds,used_folds,cfg); t,tf=choose_threshold(yd[mask],oof[mask],cfg)
    r={"family":spec["id"],"block":spec["block"],"used_folds":list(used_folds),"nested":nested,"tuned_threshold":t,"tuned_f1":tf,"auc":float(roc_auc_score(yd[mask],oof[mask])),"logloss":float(log_loss(yd[mask],oof[mask],labels=[0,1])),"fold_seconds":secs}
    if len(used_folds)==5: r["audit_f1"]=float(f1_score(y[audit],np.mean(aps,axis=0)>=nested["median_threshold"],zero_division=0))
    out_dir.mkdir(parents=True,exist_ok=True); write_json(jp,r); np.savez_compressed(npz,oof=oof,audit=np.mean(aps,axis=0),test=np.mean(tps,axis=0)); return r


def rank01(p): return pd.Series(p).rank(method="average",pct=True).to_numpy(float)


def existing_anchors(dev):
    v12=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/selected_oof.csv").set_index("row_id").loc[dev,"probability"].to_numpy(float)
    xgb=np.load(ROOT/"artifacts/v18_xgb_finalize_ensemble/fixed_round/trial59_r1367.npz")["oof"]
    lgb=np.load(ROOT/"artifacts/v19_lightgbm/fixed/trial22_r192.npz")["oof"]
    tabm=np.load(ROOT/"artifacts/v20_tabm/confirm/survey_native_plain256.npz")["oof"]
    return {"v12":v12,"xgb":xgb,"lgb":lgb,"tabm":tabm}


def ensemble_scan(cfg,ydev,folds,anchors,confirmed):
    base=.6*rank01(anchors["v12"])+.3*rank01(anchors["xgb"])+.1*rank01(anchors["lgb"])
    rows=[]
    candidates={"tabm":anchors["tabm"]}
    for r in confirmed:
        candidates[r["family"]]=np.load(OUT/"confirm"/f"{r['family']}.npz")["oof"]
    for name,p0 in candidates.items():
        for w in cfg["ensemble_new_weights"]:
            p=(1-float(w))*base+float(w)*rank01(p0); n=nested_metrics(ydev,p,folds,cfg["full_folds"],cfg); t,tf=choose_threshold(ydev,p,cfg)
            rows.append({"added":name,"weight":float(w),"nested":n,"tuned_threshold":t,"tuned_f1":tf,"auc":float(roc_auc_score(ydev,p))})
    rows.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True)
    return rows


def full_refit_predict(spec,cfg,train,test,y,seed):
    tr0=build_block(train.reset_index(drop=True),spec["block"]); te0=build_block(test.reset_index(drop=True),spec["block"]); cleaner=fit_cleaner(tr0); tr=apply_cleaner(tr0,cleaner); te=apply_cleaner(te0,cleaner); cats=cleaner["cats"]
    model=make_estimator(spec["id"],seed,cfg,True); started=time.perf_counter()
    if spec["id"]=="extratrees":
        maps=fit_ordinal(tr,cleaner); xtr=ordinal_matrix(tr,cleaner,maps); xte=ordinal_matrix(te,cleaner,maps); model.fit(xtr,y); p=model.predict_proba(xte)[:,1]
    elif spec["id"]=="ebm":
        types=["nominal" if c in set(cats) else "continuous" for c in cleaner["columns"]]; model.set_params(feature_names=cleaner["columns"],feature_types=types); model.fit(tr,y); p=model.predict_proba(te)[:,1]
    elif spec["id"]=="tabpfn":
        cat_idx=[tr.columns.get_loc(c) for c in cats]; model.set_params(categorical_features_indices=cat_idx); model.fit(tr,y); p=model.predict_proba(te)[:,1]
    else:
        model.fit(tr,y,cat_col_names=cats); p=model.predict_proba(te)[:,1]
    return p,model,cleaner,time.perf_counter()-started


def make_candidate(sample,name,p,t):
    sub=sample.copy(); sub[TARGET]=(p>=t).astype(np.int64); path=ROOT/"submissions"/name; sub.to_csv(path,index=False); ck=check_submission(sample,pd.read_csv(path)); return {"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(path),**ck}


def run(cfg):
    if (OUT/"results.json").is_file(): print("v21 already complete; immutable results preserved",flush=True); return
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/"config.json",cfg)
    train,test,y,sample,dev,audit,folds=load_data(); yd=y[dev]
    screens=[]; failures=[]; specs={x["id"]:x for x in cfg["families"] if x.get("enabled",True)}
    for fam,spec in specs.items():
        try:
            r=evaluate_family(spec,cfg,train,test,y,dev,audit,folds,cfg["screen_folds"],fam,OUT/"screen"); screens.append(r)
        except Exception as e:
            failures.append({"family":fam,"stage":"screen","error":repr(e),"traceback":traceback.format_exc()[-5000:]}); print(f"SCREEN FAILED {fam}: {e}",flush=True)
    screens.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["auc"]),reverse=True); write_json(OUT/"screen_leaderboard.json",screens)
    confirmed=[]
    for s in screens:
        if s["nested"]["nested_f1"] < float(cfg["screen_gate_nested_f1"]): continue
        fam=s["family"]
        try:
            r=evaluate_family(specs[fam],cfg,train,test,y,dev,audit,folds,cfg["full_folds"],fam,OUT/"confirm"); confirmed.append(r)
        except Exception as e:
            failures.append({"family":fam,"stage":"confirm","error":repr(e),"traceback":traceback.format_exc()[-5000:]}); print(f"CONFIRM FAILED {fam}: {e}",flush=True)
    confirmed.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["auc"]),reverse=True); write_json(OUT/"confirmation_leaderboard.json",confirmed)

    anchors=existing_anchors(dev); diversity=[]
    for r in confirmed:
        p=np.load(OUT/"confirm"/f"{r['family']}.npz")["oof"]
        diversity.append({"family":r["family"],"corr_v12":float(np.corrcoef(p,anchors["v12"])[0,1]),"corr_xgb":float(np.corrcoef(p,anchors["xgb"])[0,1]),"corr_lgb":float(np.corrcoef(p,anchors["lgb"])[0,1]),"corr_tabm":float(np.corrcoef(p,anchors["tabm"])[0,1])})
    write_json(OUT/"diversity.json",diversity)
    ens=ensemble_scan(cfg,yd,folds,anchors,confirmed); write_json(OUT/"ensemble_scan.json",ens)

    # Full-refit the best new families that are both competitive/diverse and technically refittable.
    eligible=[]
    ens_rank={r["added"]:i for i,r in enumerate(ens)}
    for r in confirmed:
        if full_refit_supported(r["family"]): eligible.append(r)
    eligible.sort(key=lambda r:(ens_rank.get(r["family"],9999),-r["nested"]["nested_f1"]))
    chosen=eligible[:int(cfg["full_refit_max_new_families"])]
    full={}
    for r in chosen:
        fam=r["family"]
        try:
            p,m,cleaner,sec=full_refit_predict(specs[fam],cfg,train,test,y,int(cfg["seed"])+777)
            d=OUT/"full"; d.mkdir(parents=True,exist_ok=True); np.save(d/f"{fam}_test.npy",p)
            try: joblib.dump({"model":m,"cleaner":cleaner,"spec":specs[fam]},d/f"{fam}.joblib",compress=3); model_path=str((d/f"{fam}.joblib").relative_to(ROOT)).replace("\\","/")
            except Exception as pe: model_path=None; failures.append({"family":fam,"stage":"model_save","error":repr(pe)})
            cand=make_candidate(sample,f"v21_{fam}_full.csv",p,float(r["nested"]["median_threshold"]))
            full[fam]={"seconds":sec,"model_path":model_path,"candidate":cand}
        except Exception as e:
            failures.append({"family":fam,"stage":"full_refit","error":repr(e),"traceback":traceback.format_exc()[-5000:]}); print(f"FULL REFIT FAILED {fam}: {e}",flush=True)

    # Produce final anchor+new rank candidate only if the selected added family has a true full-refit test vector.
    final_candidates={}
    v12t=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/test_probabilities.csv").probability.to_numpy(float)
    xgbt=pd.read_csv(ROOT/"artifacts/v18_xgb_finalize_ensemble/final_xgb/test_single.csv").probability.to_numpy(float)
    lgbt=pd.read_csv(ROOT/"artifacts/v19_lightgbm/final/test_seed5.csv").probability.to_numpy(float)
    base_test=.6*rank01(v12t)+.3*rank01(xgbt)+.1*rank01(lgbt)
    for e in ens[:10]:
        fam=e["added"]
        if fam=="tabm": newt=pd.read_csv(ROOT/"artifacts/v20_tabm/final/test_seed3.csv").probability.to_numpy(float)
        elif fam in full: newt=np.load(OUT/"full"/f"{fam}_test.npy")
        else: continue
        p=(1-e["weight"])*base_test+e["weight"]*rank01(newt); nm=f"v21_base_plus_{fam}_rank_{int(e['weight']*100)}.csv"; final_candidates[nm]=make_candidate(sample,nm,p,float(e["nested"]["median_threshold"]))
        if len(final_candidates)>=3: break

    result={"version":cfg["version"],"completed_utc":now(),"screen":screens,"confirmed":confirmed,"diversity":diversity,"ensemble_top":ens[:20],"full_refit":full,"final_candidates":final_candidates,"failures":failures,"notes":cfg.get("notes",{}),"submitted":False}
    write_json(OUT/"results.json",result); print(json.dumps({"completed":True,"confirmed":confirmed,"ensemble_top":ens[:8],"full_refit":full,"final_candidates":final_candidates,"failures":failures},indent=2,ensure_ascii=False),flush=True)


def check(cfg):
    import importlib.util
    tr,te,y,sample,dev,audit,folds=load_data()
    avail={"pytabkit":bool(importlib.util.find_spec("pytabkit")),"interpret":bool(importlib.util.find_spec("interpret")),"tabpfn":bool(importlib.util.find_spec("tabpfn")),"xrfm":bool(importlib.util.find_spec("xrfm")),"deeptab_modernnca":bool(importlib.util.find_spec("deeptab"))}
    payload={"check_passed":all(avail[k] for k in ["pytabkit","interpret","tabpfn","xrfm"]),"availability":avail,"train":len(tr),"dev":len(dev),"audit":len(audit),"test":len(te),"families":[x["id"] for x in cfg["families"]],"remote_submission_authorized":False}
    write_json(ROOT/"reports"/"v21_remaining_suite_preflight.json",payload); print(json.dumps(payload,indent=2))


def smoke(cfg):
    tr,te,y,sample,dev,audit,folds=load_data(); rows=[]
    for fam in ["realmlp_td","extratrees","ebm","xrfm"]:
        spec=next(x for x in cfg["families"] if x["id"]==fam); tri=dev[folds!=0][:3500]; vai=dev[folds==0][:800]
        try:
            pv,_,_,sec=fit_predict_one(fam,spec["block"],cfg,tr,te.head(1000),y,tri,vai,audit[:500],123); rows.append({"family":fam,"ok":True,"auc":float(roc_auc_score(y[vai],pv)),"seconds":sec})
        except Exception as e: rows.append({"family":fam,"ok":False,"error":repr(e)})
    payload={"smoke_passed":all(x["ok"] for x in rows),"rows":rows,"note":"TabR and TabPFN are intentionally not forced in smoke; the main suite catches/logs family-specific failures and continues."}; write_json(ROOT/"reports"/"v21_remaining_suite_smoke.json",payload); print(json.dumps(payload,indent=2))


def status(cfg):
    print(json.dumps({"screen":len(list((OUT/"screen").glob("*.json"))) if (OUT/"screen").exists() else 0,"screen_target":len([x for x in cfg["families"] if x.get("enabled",True)]),"confirm":len(list((OUT/"confirm").glob("*.json"))) if (OUT/"confirm").exists() else 0,"full":len(list((OUT/"full").glob("*_test.npy"))) if (OUT/"full").exists() else 0,"results":(OUT/"results.json").is_file()},indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["check","smoke","run","status"]); ap.add_argument("--config",type=Path,default=ROOT/"configs"/"v21_remaining_suite.json"); a=ap.parse_args(); cfg=read_json(a.config)
    if a.phase=="check": check(cfg)
    elif a.phase=="smoke": smoke(cfg)
    elif a.phase=="status": status(cfg)
    else: run(cfg)


if __name__=="__main__": main()
