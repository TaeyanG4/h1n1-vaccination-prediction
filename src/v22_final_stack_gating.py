"""Final v22: leakage-safe logistic stacking + slice-aware soft gating.

Uses only frozen OOF predictions from established model families plus a very small
set of raw slice variables. Meta-model hyperparameters are selected inside each
outer fold, so the final OOF estimate remains honest. No Kaggle submission is
performed. A candidate CSV is generated only if the final meta blend clears the
predeclared OOF gate versus the v21 parent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, log_loss

from baseline import check_submission
from v21_remaining_suite import ROOT, load_data, existing_anchors, nested_metrics, choose_threshold, write_json


OUT = ROOT / "artifacts" / "v22_final_stack_gating"
TARGET = "vacc_h1n1_f"
BASE_NAMES = ["v12", "xgb", "lgb", "tabm", "ebm", "realmlp"]
SLICE_COLS = [
    "doctor_recc_h1n1",
    "agegrp",
    "employment_status",
    "opinion_h1n1_vacc_effective",
    "opinion_h1n1_risk",
    "opinion_h1n1_sick_from_vacc",
]
MODES = ["base6", "base6_slice", "gated"]
CS = [0.01, 0.03, 0.1, 0.3, 1.0]
ALPHAS = [0.5, 0.75, 1.0]
MIN_GAIN = 0.0005
MIN_BETTER_FOLDS = 4


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clip(p): return np.clip(np.asarray(p, float), 1e-5, 1-1e-5)
def logit(p):
    p=clip(p); return np.log(p/(1-p))


def load_oof_test(dev):
    a=existing_anchors(dev)
    ebm=np.load(ROOT/"artifacts/v21_focus_confirm/confirm/ebm.npz")["oof"]
    mlp=np.load(ROOT/"artifacts/v21_focus_confirm/confirm/realmlp_td.npz")["oof"]
    oof={**a,"ebm":ebm,"realmlp":mlp}

    v12t=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/test_probabilities.csv").probability.to_numpy(float)
    xgbt=pd.read_csv(ROOT/"artifacts/v18_xgb_finalize_ensemble/final_xgb/test_single.csv").probability.to_numpy(float)
    lgbt=pd.read_csv(ROOT/"artifacts/v19_lightgbm/final/test_seed5.csv").probability.to_numpy(float)
    tabmt=np.load(ROOT/"artifacts/v20_tabm/final/test_seed42.npy")
    ebmt=np.load(ROOT/"artifacts/v21_focus_finalize/full/ebm/test_single.npy")
    mlpt=np.load(ROOT/"artifacts/v21_focus_finalize/full/realmlp_td/test_single.npy")
    test={"v12":v12t,"xgb":xgbt,"lgb":lgbt,"tabm":tabmt,"ebm":ebmt,"realmlp":mlpt}
    return oof,test


def raw_slices(train,test,dev):
    tr=train.iloc[dev][SLICE_COLS].copy().reset_index(drop=True)
    te=test[SLICE_COLS].copy().reset_index(drop=True)
    for c in SLICE_COLS:
        tr[c]=tr[c].where(tr[c].notna(),"__MISSING__").astype(str)
        te[c]=te[c].where(te[c].notna(),"__MISSING__").astype(str)
    return tr,te


def base_matrix(preds):
    return np.column_stack([logit(preds[n]) for n in BASE_NAMES])


@dataclass
class MetaTransform:
    mode: str
    encoder: OneHotEncoder | None
    scaler: StandardScaler


def make_design(base, raw, fit_idx, eval_idx, mode, transform=None):
    Xb=base
    if transform is None:
        enc=None
        if mode in {"base6_slice","gated"}:
            enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float64)
            Sfit=enc.fit_transform(raw.iloc[fit_idx])
            Seval=enc.transform(raw.iloc[eval_idx])
        else:
            Sfit=Seval=None
        Xfit=Xb[fit_idx]
        Xeval=Xb[eval_idx]
        if mode=="base6_slice":
            Xfit=np.column_stack([Xfit,Sfit]); Xeval=np.column_stack([Xeval,Seval])
        elif mode=="gated":
            # Low-dimensional model disagreements become soft slice-specific gates.
            diffs_fit=np.column_stack([Xb[fit_idx,4]-Xb[fit_idx,0], Xb[fit_idx,5]-Xb[fit_idx,0], Xb[fit_idx,1]-Xb[fit_idx,0], Xb[fit_idx,2]-Xb[fit_idx,0]])
            diffs_eval=np.column_stack([Xb[eval_idx,4]-Xb[eval_idx,0], Xb[eval_idx,5]-Xb[eval_idx,0], Xb[eval_idx,1]-Xb[eval_idx,0], Xb[eval_idx,2]-Xb[eval_idx,0]])
            inter_fit=np.einsum("ij,ik->ijk",Sfit,diffs_fit).reshape(len(fit_idx),-1)
            inter_eval=np.einsum("ij,ik->ijk",Seval,diffs_eval).reshape(len(eval_idx),-1)
            Xfit=np.column_stack([Xfit,Sfit,inter_fit]); Xeval=np.column_stack([Xeval,Seval,inter_eval])
        scaler=StandardScaler().fit(Xfit)
        return scaler.transform(Xfit),scaler.transform(Xeval),MetaTransform(mode,enc,scaler)
    enc=transform.encoder
    Xeval=Xb[eval_idx]
    if mode=="base6_slice":
        Seval=enc.transform(raw.iloc[eval_idx]); Xeval=np.column_stack([Xeval,Seval])
    elif mode=="gated":
        Seval=enc.transform(raw.iloc[eval_idx])
        diffs_eval=np.column_stack([Xb[eval_idx,4]-Xb[eval_idx,0], Xb[eval_idx,5]-Xb[eval_idx,0], Xb[eval_idx,1]-Xb[eval_idx,0], Xb[eval_idx,2]-Xb[eval_idx,0]])
        inter_eval=np.einsum("ij,ik->ijk",Seval,diffs_eval).reshape(len(eval_idx),-1)
        Xeval=np.column_stack([Xeval,Seval,inter_eval])
    return transform.scaler.transform(Xeval)


def fit_meta(base,raw,y,fit_idx,eval_idx,mode,C):
    Xtr,Xva,t=make_design(base,raw,fit_idx,eval_idx,mode)
    # sklearn 1.9 deprecates the explicit `penalty` argument.  L2 remains the
    # default, so omit it to avoid emitting a FutureWarning on stderr.  The old
    # PowerShell runner treated any native stderr output as a terminating error.
    m=LogisticRegression(C=C,solver="lbfgs",max_iter=5000,random_state=20261006)
    m.fit(Xtr,y[fit_idx])
    return m.predict_proba(Xva)[:,1],m,t


def inner_select(base,raw,y,folds,outer_fold,cfg):
    train_folds=[f for f in range(5) if f!=outer_fold]
    rows=[]
    for mode in MODES:
        for C in CS:
            p=np.full(len(y),np.nan)
            for vf in train_folds:
                va=np.flatnonzero(folds==vf)
                tr=np.flatnonzero((folds!=outer_fold)&(folds!=vf))
                pv,_,_=fit_meta(base,raw,y,tr,va,mode,C); p[va]=pv
            mask=np.isfinite(p)
            t,score=choose_threshold(y[mask],p[mask],cfg)
            rows.append({"mode":mode,"C":C,"inner_f1":score,"threshold":t,"auc":float(roc_auc_score(y[mask],p[mask]))})
    rows.sort(key=lambda r:(r["inner_f1"],r["auc"],-r["C"]),reverse=True)
    return rows[0],rows


def main():
    if (OUT/"results.json").is_file():
        print("v22 already complete; immutable results preserved"); return
    cfg=json.loads((ROOT/"configs/v21_remaining_suite.json").read_text(encoding="utf-8"))
    train,test,y,sample,dev,audit,folds=load_data(); yd=y[dev]
    oof,testp=load_oof_test(dev)
    base=base_matrix(oof); test_base=base_matrix(testp)
    raw,test_raw=raw_slices(train,test,dev)

    # v21 private-best recipe's OOF parent, selected without using its LB score here.
    anchor=.6*oof["v12"]+.3*oof["xgb"]+.1*oof["lgb"]
    parent=.75*anchor+.15*oof["ebm"]+.10*oof["realmlp"]
    parent_metrics=nested_metrics(yd,parent,folds,[0,1,2,3,4],cfg)

    stack=np.full(len(yd),np.nan); selections=[]
    for f in range(5):
        best,table=inner_select(base,raw,yd,folds,f,cfg)
        tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f)
        pv,_,_=fit_meta(base,raw,yd,tr,va,best["mode"],best["C"]); stack[va]=pv
        selections.append({"outer_fold":f,"selected":best,"top5":table[:5]})
        print(f"fold={f} selected={best['mode']} C={best['C']} inner_f1={best['inner_f1']:.6f}",flush=True)

    stack_metrics=nested_metrics(yd,stack,folds,[0,1,2,3,4],cfg)
    blends=[]
    for a in ALPHAS:
        q=(1-a)*parent+a*stack
        n=nested_metrics(yd,q,folds,[0,1,2,3,4],cfg); t,tf=choose_threshold(yd,q,cfg)
        better=sum(np.array(n["fold_f1"])>np.array(parent_metrics["fold_f1"])+1e-12)
        blends.append({"alpha_stack":a,"nested":n,"tuned_threshold":t,"tuned_f1":tf,"better_folds":int(better),"auc":float(roc_auc_score(yd,q))})
    blends.sort(key=lambda r:(r["nested"]["nested_f1"],r["better_folds"],-r["nested"]["fold_std"]),reverse=True)
    best_blend=blends[0]
    gain=best_blend["nested"]["nested_f1"]-parent_metrics["nested_f1"]
    passed=bool(gain>=MIN_GAIN and best_blend["better_folds"]>=MIN_BETTER_FOLDS)

    candidate=None
    model_artifact=None
    if passed:
        # Freeze mode/C by majority/median tendency across honest outer selections.
        modes=[x["selected"]["mode"] for x in selections]
        mode=max(set(modes),key=lambda z:(modes.count(z),-MODES.index(z)))
        Cs=[x["selected"]["C"] for x in selections if x["selected"]["mode"]==mode]
        C=float(np.median(Cs)) if Cs else 0.1
        all_idx=np.arange(len(yd)); dummy=np.array([],dtype=int)
        # Fit encoder/scaler on all development OOF rows, then train logistic model.
        if mode=="base6":
            Xall=base
            scaler=StandardScaler().fit(Xall); Xall=scaler.transform(Xall); enc=None
        else:
            enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False,dtype=np.float64)
            Sall=enc.fit_transform(raw)
            Xall=base
            if mode=="base6_slice": Xall=np.column_stack([Xall,Sall])
            else:
                diffs=np.column_stack([base[:,4]-base[:,0],base[:,5]-base[:,0],base[:,1]-base[:,0],base[:,2]-base[:,0]])
                inter=np.einsum("ij,ik->ijk",Sall,diffs).reshape(len(yd),-1)
                Xall=np.column_stack([Xall,Sall,inter])
            scaler=StandardScaler().fit(Xall); Xall=scaler.transform(Xall)
        model=LogisticRegression(C=C,solver="lbfgs",max_iter=5000,random_state=20261006).fit(Xall,yd)
        if mode=="base6": Xtest=scaler.transform(test_base)
        else:
            Ste=enc.transform(test_raw); Xtest=test_base
            if mode=="base6_slice": Xtest=np.column_stack([Xtest,Ste])
            else:
                diffs=np.column_stack([test_base[:,4]-test_base[:,0],test_base[:,5]-test_base[:,0],test_base[:,1]-test_base[:,0],test_base[:,2]-test_base[:,0]])
                inter=np.einsum("ij,ik->ijk",Ste,diffs).reshape(len(test),-1)
                Xtest=np.column_stack([Xtest,Ste,inter])
            Xtest=scaler.transform(Xtest)
        stack_test=model.predict_proba(Xtest)[:,1]
        anchor_test=.6*testp["v12"]+.3*testp["xgb"]+.1*testp["lgb"]
        parent_test=.75*anchor_test+.15*testp["ebm"]+.10*testp["realmlp"]
        a=best_blend["alpha_stack"]; pred=(1-a)*parent_test+a*stack_test
        sub=sample.copy(); sub[TARGET]=(pred>=best_blend["nested"]["median_threshold"]).astype(np.int64)
        path=ROOT/"submissions/v22_final_stack_gating.csv"; sub.to_csv(path,index=False)
        ck=check_submission(sample,pd.read_csv(path))
        candidate={"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(path),"threshold":best_blend["nested"]["median_threshold"],**ck}
        OUT.mkdir(parents=True,exist_ok=True); model_artifact=OUT/"meta_model.joblib"; joblib.dump({"model":model,"encoder":enc,"scaler":scaler,"mode":mode,"C":C,"slice_cols":SLICE_COLS,"base_names":BASE_NAMES},model_artifact,compress=3)

    result={"version":"v22_final_stack_gating","completed_utc":now(),"parent":parent_metrics,"stack":stack_metrics,"outer_selections":selections,"blend_scan":blends,"best_blend":best_blend,"gate":{"min_gain":MIN_GAIN,"min_better_folds":MIN_BETTER_FOLDS,"gain":gain,"passed":passed},"candidate":candidate,"submitted":False}
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/"results.json",result)
    print(json.dumps({"parent":parent_metrics,"stack":stack_metrics,"best_blend":best_blend,"gate":result["gate"],"candidate":candidate},indent=2),flush=True)


if __name__=="__main__": main()
