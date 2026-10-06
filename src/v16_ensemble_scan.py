"""Simple OOF ensemble scan only: equal probability/rank averages, no optimized blend."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts"/"v16_ensemble_scan"; TARGET="vacc_h1n1_f"
def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def threshold(y,p):
    grid=np.linspace(.15,.60,91); s=np.array([f1_score(y,p>=t,zero_division=0) for t in grid]); ix=np.flatnonzero(np.isclose(s,s.max(),atol=1e-12,rtol=0)); j=ix[len(ix)//2]; return float(grid[j]),float(s[j])
def nested(y,p,folds):
    hard=np.zeros(len(y),bool); ts=[]
    for f in range(5):
        m=folds!=f; t,_=threshold(y[m],p[m]); ts.append(t); hard[folds==f]=p[folds==f]>=t
    return float(f1_score(y,hard,zero_division=0)),ts
def rank01(p):
    return pd.Series(p).rank(method="average",pct=True).to_numpy(float)
def main():
    OUT.mkdir(parents=True,exist_ok=True); man=pd.read_csv(ROOT/"artifacts"/"baseline_v1"/"split_manifest.csv"); dev=man.loc[man.partition.eq("development"),"row_id"].to_numpy(int); folds=man.loc[dev,"dev_fold"].to_numpy(int); yall=pd.read_csv(ROOT/"data"/"raw"/"train_labels.csv")[TARGET].to_numpy(int); y=yall[dev]
    sources={}
    def add(name,path,col="probability"):
        if Path(path).is_file():
            d=pd.read_csv(path).set_index("row_id").loc[dev]; sources[name]=d[col].to_numpy(float)
    add("v2_cat",ROOT/"artifacts"/"v2"/"selected_oof.csv")
    add("v3_xgb",ROOT/"artifacts"/"v3"/"candidates"/"xgb_d4_survey"/"oof.csv")
    if (ROOT/"artifacts"/"v5_automl"/"oof_probabilities.csv").is_file(): add("v5_stack",ROOT/"artifacts"/"v5_automl"/"oof_probabilities.csv","CatBoost_BAG_L2")
    if (ROOT/"artifacts"/"v6_tabicl"/"oof.csv").is_file(): add("v6_tabicl",ROOT/"artifacts"/"v6_tabicl"/"oof.csv")
    r14=ROOT/"artifacts"/"v14_catboost_feature_hpo"/"results.json"
    if r14.is_file():
        rr=json.loads(r14.read_text(encoding="utf-8")); key=rr["chosen"]
        p=ROOT/"artifacts"/"v14_catboost_feature_hpo"/(f"hpo_{key.split('::',1)[1]}_oof.csv" if key.startswith("hpo::") else f"{key.split('::',1)[1]}_oof.csv"); add("v14_cat",p)
    rows=[]
    for n,p in sources.items():
        t,s=threshold(y,p); ne,nts=nested(y,p,folds); rows.append({"name":n,"kind":"single","components":n,"tuned_f1":s,"nested_f1":ne,"threshold":t,"auc":float(roc_auc_score(y,p))})
    names=list(sources)
    for k in [2,3]:
        for combo in combinations(names,k):
            ps=[sources[n] for n in combo]
            for kind,p in [("prob_equal",np.mean(ps,axis=0)),("rank_equal",np.mean([rank01(x) for x in ps],axis=0))]:
                t,s=threshold(y,p); ne,nts=nested(y,p,folds); rows.append({"name":"+".join(combo)+":"+kind,"kind":kind,"components":"+".join(combo),"tuned_f1":s,"nested_f1":ne,"threshold":t,"auc":float(roc_auc_score(y,p))})
    df=pd.DataFrame(rows).sort_values(["nested_f1","tuned_f1","auc"],ascending=False); df.to_csv(OUT/"comparison.csv",index=False); result={"version":"v16_ensemble_scan","completed_utc":now(),"sources":names,"top":df.head(20).to_dict("records"),"policy":"simple equal probability/rank averages only; no optimized blending","submitted":False}; write_json(OUT/"results.json",result); print(df.head(20).to_string(index=False),flush=True)
if __name__=="__main__": main()
