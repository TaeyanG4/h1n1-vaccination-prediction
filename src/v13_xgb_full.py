"""Full-label finalization of the frozen v3 xgb_d4_survey recipe."""
from __future__ import annotations

import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from baseline import check_submission
from v2_features import build_features
from v3_pipeline import fit_xgb_schema, transform_xgb

ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v13_xgb_full"

def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def make_model(cfg, seed):
    p = cfg["params"]
    return XGBClassifier(
        objective="binary:logistic", enable_categorical=True, n_estimators=int(cfg["n_estimators"]),
        learning_rate=float(p["learning_rate"]), max_depth=int(p["max_depth"]),
        min_child_weight=float(p["min_child_weight"]), subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]), reg_lambda=float(p["reg_lambda"]),
        reg_alpha=float(p["reg_alpha"]), gamma=float(p["gamma"]), tree_method=p["tree_method"],
        max_cat_to_onehot=int(p["max_cat_to_onehot"]), eval_metric=p["eval_metric"],
        random_state=int(seed), n_jobs=int(cfg["threads"]), verbosity=0,
    )

def run(cfg):
    if (OUT / "results.json").is_file():
        print("v13 already complete", flush=True); return
    raw = ROOT / "data" / "raw"
    tr, te, lab, sample = [pd.read_csv(raw / n) for n in ["train.csv","test.csv","train_labels.csv","submission.csv"]]
    y = lab[TARGET].to_numpy(int)
    x, xt = build_features(tr, cfg["feature_mode"]), build_features(te, cfg["feature_mode"])
    schema = fit_xgb_schema(x)
    x2, xt2 = transform_xgb(x, schema), transform_xgb(xt, schema)
    OUT.mkdir(parents=True, exist_ok=True)
    preds, models, seconds = [], [], []
    for seed in cfg["seeds"]:
        started = time.perf_counter(); m = make_model(cfg, seed); m.fit(x2, y, verbose=False)
        p = m.predict_proba(xt2)[:,1]; seconds.append(time.perf_counter()-started)
        path = OUT / f"xgb_d4_survey_full_seed{seed}.joblib"; joblib.dump({"model":m,"schema":schema}, path, compress=3)
        preds.append(p); models.append({"seed":seed,"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(path),"seconds":seconds[-1]})
    arr = np.vstack(preds); single, mean5 = arr[0], arr.mean(axis=0); t=float(cfg["frozen_threshold"])
    candidates = {}
    for name,p in [("single",single),("seed5",mean5)]:
        pd.DataFrame({"Id":sample.Id,"probability":p}).to_csv(OUT/f"test_{name}.csv",index=False)
        sub=sample.copy(); sub[TARGET]=(p>=t).astype(np.int64)
        sp=ROOT/"submissions"/f"v13_xgb_d4_survey_full_{name}_t0315.csv"
        if sp.exists(): raise FileExistsError(sp)
        sub.to_csv(sp,index=False); ck=check_submission(sample,pd.read_csv(sp))
        candidates[name]={"path":str(sp.relative_to(ROOT)).replace("\\","/"),"sha256":sha(sp),**ck}
    result={"version":cfg["version"],"completed_utc":now(),"training_rows":len(tr),"features":x.shape[1],
            "n_estimators":cfg["n_estimators"],"frozen_threshold":t,"models":models,"candidates":candidates,
            "seed_prediction_mean_std":float(np.mean(np.std(arr,axis=0))),
            "single_vs_seed5_corr":float(np.corrcoef(single,mean5)[0,1]),
            "single_vs_seed5_hard_disagreement":int(np.sum((single>=t)!=(mean5>=t))),"submitted":False}
    write_json(OUT/"results.json",result); print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,default=ROOT/"configs"/"v13_xgb_full.json"); a=ap.parse_args(); run(read_json(a.config))
if __name__=="__main__": main()
