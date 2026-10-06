"""Train/test shift audit: per-column JS divergence + grouped adversarial CV."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from v2_features import build_features, categorical_columns

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"artifacts"/"v15_shift_audit"
def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p,o): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def js_col(a,b):
    aa=a.where(a.notna(),"__MISSING__").astype(str); bb=b.where(b.notna(),"__MISSING__").astype(str); cats=sorted(set(aa)|set(bb)); p=aa.value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy(float); q=bb.value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy(float); m=.5*(p+q)
    def kl(x,y):
        mask=x>0; return float(np.sum(x[mask]*np.log(x[mask]/y[mask])))
    return .5*kl(p,m)+.5*kl(q,m)
def main():
    if (OUT/"results.json").is_file(): print("v15 already complete"); return
    OUT.mkdir(parents=True,exist_ok=True)
    raw=ROOT/"data"/"raw"; tr=pd.read_csv(raw/"train.csv"); te=pd.read_csv(raw/"test.csv"); allraw=pd.concat([tr,te],ignore_index=True); source=np.r_[np.zeros(len(tr),dtype=int),np.ones(len(te),dtype=int)]
    diverg=[{"feature":c,"js_divergence":js_col(tr[c],te[c]),"train_missing":float(tr[c].isna().mean()),"test_missing":float(te[c].isna().mean()),"missing_delta":float(te[c].isna().mean()-tr[c].isna().mean())} for c in tr.columns]; pd.DataFrame(diverg).sort_values("js_divergence",ascending=False).to_csv(OUT/"column_shift.csv",index=False)
    x=build_features(allraw,"survey"); cats=categorical_columns(x)
    for c in x.columns:
        if c in cats: x[c]=x[c].where(x[c].notna(),"__MISSING__").astype(str)
        else: x[c]=pd.to_numeric(x[c],errors="raise").astype(float)
    groups=pd.util.hash_pandas_object(allraw.fillna("__MISSING__").astype(str),index=False).astype(str).to_numpy(); sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261006); oof=np.full(len(x),np.nan); imp=[]
    for f,(ti,vi) in enumerate(sg.split(x,source,groups)):
        m=CatBoostClassifier(iterations=350,depth=6,learning_rate=.05,l2_leaf_reg=6,loss_function="Logloss",cat_features=cats,random_seed=20261006+f,thread_count=6,allow_writing_files=False,verbose=False); m.fit(x.iloc[ti],source[ti]); oof[vi]=m.predict_proba(x.iloc[vi])[:,1]; imp.append(m.feature_importances_)
    auc=float(roc_auc_score(source,oof)); fi=pd.DataFrame({"feature":x.columns,"importance":np.mean(imp,axis=0)}).sort_values("importance",ascending=False); fi.to_csv(OUT/"adversarial_feature_importance.csv",index=False); pd.DataFrame({"source":source,"prob_test":oof}).to_csv(OUT/"adversarial_oof.csv",index=False)
    result={"version":"v15_shift_audit","completed_utc":now(),"adversarial_auc":auc,"interpretation":"0.5 is indistinguishable; higher values indicate train/test distribution shift","top_js":sorted(diverg,key=lambda r:r["js_divergence"],reverse=True)[:15],"top_adversarial_features":fi.head(15).to_dict("records"),"submitted":False}; write_json(OUT/"results.json",result); print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
