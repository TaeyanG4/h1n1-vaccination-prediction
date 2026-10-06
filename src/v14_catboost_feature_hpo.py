"""Compact CatBoost feature ablation + structural HPO on frozen grouped folds."""
from __future__ import annotations
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, log_loss

from baseline import check_submission
from v2_features import build_features, categorical_columns

ROOT=Path(__file__).resolve().parents[1]; TARGET="vacc_h1n1_f"; OUT=ROOT/"artifacts"/"v14_catboost_feature_hpo"; PARENT=ROOT/"artifacts"/"baseline_v1"
def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data():
    raw=ROOT/"data"/"raw"; tr=pd.read_csv(raw/"train.csv"); te=pd.read_csv(raw/"test.csv"); y=pd.read_csv(raw/"train_labels.csv")[TARGET].to_numpy(int); sample=pd.read_csv(raw/"submission.csv")
    man=pd.read_csv(PARENT/"split_manifest.csv",dtype={"group_hash":str}); dev=man.loc[man.partition.eq("development"),"row_id"].to_numpy(int); audit=man.loc[man.partition.eq("audit"),"row_id"].to_numpy(int); folds=man.loc[dev,"dev_fold"].to_numpy(int)
    return tr,te,y,sample,man,dev,audit,folds

def token(s): return s.where(s.notna(),"__MISSING__").astype(str)
def fit_freq(raw, ids, cols):
    return {c:(token(raw.iloc[ids][c]).value_counts(dropna=False)/len(ids)).to_dict() for c in cols}
def fit_freq_frame(frame, cols):
    return {c:(token(frame[c]).value_counts(dropna=False)/len(frame)).to_dict() for c in cols}
def add_freq(x, raw, maps, cols):
    z=x.copy()
    for c in cols: z[f"v14_freq_{c}"]=token(raw[c]).map(maps[c]).fillna(0).astype(float)
    return z
def add_missing(x, raw):
    z=x.copy(); h1=[c for c in raw if c.startswith("opinion_h1n1_")]; seas=[c for c in raw if c.startswith("opinion_seas_")]
    z["v14_employment_pair_missing"]=(raw.employment_occupation.isna()&raw.employment_industry.isna()).astype(float)
    z["v14_employment_status_missing"]=raw.employment_status.isna().astype(float)
    z["v14_doctor_any_missing"]=raw[["doctor_recc_h1n1","doctor_recc_seasonal"]].isna().any(axis=1).astype(float)
    z["v14_doctor_both_missing"]=raw[["doctor_recc_h1n1","doctor_recc_seasonal"]].isna().all(axis=1).astype(float)
    z["v14_h1n1_opinion_all_missing"]=raw[h1].isna().all(axis=1).astype(float); z["v14_seas_opinion_all_missing"]=raw[seas].isna().all(axis=1).astype(float)
    z["v14_demographic_missing_count"]=raw[["education_comp","marital","rent_own_r","employment_status"]].isna().sum(axis=1).astype(float)
    z["v14_health_missing_count"]=raw[["health_insurance","health_worker","chronic_med_condition","child_under_6_months","doctor_recc_h1n1","doctor_recc_seasonal"]].isna().sum(axis=1).astype(float)
    z["v14_rent_special_code"]=pd.to_numeric(raw.rent_own_r,errors="coerce").isin([77,99]).astype(float)
    z["v14_h1n1_unknown_count"]=raw[h1].isin(["Dont Know","Don't Know","Refused"]).sum(axis=1).astype(float); z["v14_seas_unknown_count"]=raw[seas].isin(["Dont Know","Don't Know","Refused"]).sum(axis=1).astype(float)
    z["v14_any_refused"]=raw[[*h1,*seas]].eq("Refused").any(axis=1).astype(float)
    return z

def prepare(block, raw, maps, cfg, medians=None):
    x=build_features(raw,"survey")
    if "freq" in block: x=add_freq(x,raw,maps,cfg["frequency_columns"])
    if "missing" in block: x=add_missing(x,raw)
    cats=categorical_columns(x); z=x.copy()
    for c in z.columns:
        if c in cats: z[c]=z[c].where(z[c].notna(),"__MISSING__").astype(str)
        else: z[c]=pd.to_numeric(z[c],errors="raise").astype(float)
    if block.endswith("_impute"):
        nums=[c for c in z if c not in cats]
        if medians is None: medians=z[nums].median(numeric_only=True).to_dict()
        z[nums]=z[nums].fillna(medians)
    return z,cats,medians

def thresholds(y,p,cfg):
    grid=np.linspace(cfg["threshold_min"],cfg["threshold_max"],cfg["threshold_steps"]); scores=np.array([f1_score(y,p>=t,zero_division=0) for t in grid]); mx=scores.max(); ix=np.flatnonzero(np.isclose(scores,mx,atol=1e-12,rtol=0)); j=ix[len(ix)//2]; return float(grid[j]),float(scores[j])
def nested(y,p,folds,cfg):
    hard=np.zeros(len(y),dtype=bool); ts=[]
    for f in range(5):
        tr=folds!=f; va=folds==f; t,_=thresholds(y[tr],p[tr],cfg); ts.append(t); hard[va]=p[va]>=t
    return {"f1":float(f1_score(y,hard,zero_division=0)),"thresholds":ts}

def fit_one(params,cats,seed,threads):
    clean={k:v for k,v in params.items() if k!="id"}
    return CatBoostClassifier(**clean,cat_features=cats,random_seed=seed,thread_count=threads,allow_writing_files=False,verbose=False)

def cv_candidate(name,block,params,cfg,tr,te,y,man,dev,audit,folds,trans_maps=None,screen_folds=None):
    use_folds=list(range(5)) if screen_folds is None else list(screen_folds); oof=np.full(len(dev),np.nan); audit_parts=[]; test_parts=[]; fold_rows=[]
    for f in use_folds:
        tri=dev[folds!=f]; vai=dev[folds==f]
        maps=trans_maps if trans_maps is not None else fit_freq(tr,tri,cfg["frequency_columns"])
        xtr,cats,med=prepare(block,tr.iloc[tri].reset_index(drop=True),maps,cfg)
        xva,_,_=prepare(block,tr.iloc[vai].reset_index(drop=True),maps,cfg,med)
        xa,_,_=prepare(block,tr.iloc[audit].reset_index(drop=True),maps,cfg,med)
        xt,_,_=prepare(block,te.reset_index(drop=True),maps,cfg,med)
        m=fit_one(params,cats,cfg["seed"]+f,cfg["threads"]); st=time.perf_counter(); m.fit(xtr,y[tri]); sec=time.perf_counter()-st
        pv=m.predict_proba(xva)[:,1]; oof[np.flatnonzero(folds==f)]=pv; audit_parts.append(m.predict_proba(xa)[:,1]); test_parts.append(m.predict_proba(xt)[:,1]); fold_rows.append({"fold":f,"seconds":sec})
    mask=np.isfinite(oof); yy=y[dev][mask]; pp=oof[mask]; t,score=thresholds(yy,pp,cfg)
    out={"name":name,"block":block,"params":params,"threshold":t,"oof_f1":score,"auc":float(roc_auc_score(yy,pp)),"logloss":float(log_loss(yy,pp,labels=[0,1])),"fold_rows":fold_rows,"oof":oof}
    if screen_folds is None:
        out["nested"]=nested(y[dev],oof,folds,cfg); out["fold_f1"]=[float(f1_score(y[dev][folds==f],oof[folds==f]>=t,zero_division=0)) for f in range(5)]; out["fold_std"]=float(np.std(out["fold_f1"],ddof=1)); out["audit"]=np.mean(audit_parts,axis=0); out["test"]=np.mean(test_parts,axis=0)
    return out

def run(cfg):
    if (OUT/"results.json").is_file(): print("v14 already complete",flush=True); return
    OUT.mkdir(parents=True,exist_ok=True); tr,te,y,sample,man,dev,audit,folds=load_data(); ydev=y[dev]
    trans_maps=fit_freq_frame(pd.concat([tr,te],ignore_index=True),cfg["frequency_columns"])
    block_results={}
    for block in cfg["feature_blocks"]:
        tm=trans_maps if block=="survey_freq_transductive" else None; r=cv_candidate(block,block,cfg["base_catboost"],cfg,tr,te,y,man,dev,audit,folds,tm)
        block_results[block]=r; pd.DataFrame({"row_id":dev,"target":ydev,"probability":r["oof"]}).to_csv(OUT/f"{block}_oof.csv",index=False)
        print(f"BLOCK {block} nested={r['nested']['f1']:.6f} tuned={r['oof_f1']:.6f}",flush=True)
    best_block=max(block_results,key=lambda k:(block_results[k]["nested"]["f1"],block_results[k]["oof_f1"])); tm=trans_maps if best_block=="survey_freq_transductive" else None
    screens=[]
    for p in cfg["hpo"]:
        r=cv_candidate(p["id"],best_block,p,cfg,tr,te,y,man,dev,audit,folds,tm,cfg["hpo_screen_folds"]); screens.append(r); print(f"HPO SCREEN {p['id']} f1={r['oof_f1']:.6f}",flush=True)
    screens.sort(key=lambda r:(r["oof_f1"],r["auc"]),reverse=True); top=screens[:cfg["hpo_top_k"]]
    full_hpo={}
    for s in top:
        r=cv_candidate(s["name"],best_block,s["params"],cfg,tr,te,y,man,dev,audit,folds,tm); full_hpo[s["name"]]=r; pd.DataFrame({"row_id":dev,"target":ydev,"probability":r["oof"]}).to_csv(OUT/f"hpo_{s['name']}_oof.csv",index=False); print(f"HPO FULL {s['name']} nested={r['nested']['f1']:.6f} tuned={r['oof_f1']:.6f}",flush=True)
    all_final={f"block::{k}":v for k,v in block_results.items()}; all_final.update({f"hpo::{k}":v for k,v in full_hpo.items()}); chosen_key=max(all_final,key=lambda k:(all_final[k]["nested"]["f1"],all_final[k]["oof_f1"])); chosen=all_final[chosen_key]
    # Fit chosen recipe on all labels. Strict frequency maps use all labels; transductive block intentionally uses train+test X.
    fm=trans_maps if chosen["block"]=="survey_freq_transductive" else fit_freq(tr,np.arange(len(tr)),cfg["frequency_columns"])
    xall,cats,med=prepare(chosen["block"],tr.reset_index(drop=True),fm,cfg); xt,_,_=prepare(chosen["block"],te.reset_index(drop=True),fm,cfg,med)
    model=fit_one(chosen["params"],cats,cfg["seed"],cfg["threads"]); model.fit(xall,y); testp=model.predict_proba(xt)[:,1]; mp=OUT/"selected_full_model.joblib"; joblib.dump({"model":model,"block":chosen["block"],"params":chosen["params"],"freq_maps":fm,"medians":med},mp,compress=3)
    t=float(chosen["threshold"]); sub=sample.copy(); sub[TARGET]=(testp>=t).astype(np.int64); sp=ROOT/"submissions"/"v14_catboost_feature_hpo_full.csv"; sub.to_csv(sp,index=False); ck=check_submission(sample,pd.read_csv(sp)); pd.DataFrame({"Id":sample.Id,"probability":testp}).to_csv(OUT/"selected_test_probabilities.csv",index=False)
    def compact(r): return {k:v for k,v in r.items() if k not in {"oof","audit","test"}}
    result={"version":cfg["version"],"completed_utc":now(),"best_feature_block":best_block,"chosen":chosen_key,"chosen_block":chosen["block"],"chosen_threshold":t,"chosen_nested_f1":chosen["nested"]["f1"],"chosen_oof_f1":chosen["oof_f1"],"chosen_audit_f1":float(f1_score(y[audit],chosen["audit"]>=t,zero_division=0)),"blocks":{k:compact(v) for k,v in block_results.items()},"hpo_screen":[compact(v) for v in screens],"hpo_full":{k:compact(v) for k,v in full_hpo.items()},"full_model_sha256":sha(mp),"submission":{"path":str(sp.relative_to(ROOT)).replace("\\","/"),"sha256":sha(sp),**ck},"submitted":False}
    write_json(OUT/"results.json",result); print(json.dumps({"completed":True,"chosen":chosen_key,"nested_f1":result["chosen_nested_f1"],"submission":result["submission"]},indent=2),flush=True)

def check(cfg):
    tr,te,y,sample,man,dev,audit,folds=load_data(); assert sorted(set(folds))==[0,1,2,3,4]; print(json.dumps({"check_passed":True,"train":len(tr),"test":len(te),"dev":len(dev),"audit":len(audit),"blocks":cfg["feature_blocks"],"hpo":len(cfg["hpo"]),"submitted":False},indent=2))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["check","run"]); ap.add_argument("--config",type=Path,default=ROOT/"configs"/"v14_catboost_feature_hpo.json"); a=ap.parse_args(); cfg=read_json(a.config); check(cfg) if a.phase=="check" else run(cfg)
if __name__=="__main__": main()
