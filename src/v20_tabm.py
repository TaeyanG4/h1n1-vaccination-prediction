"""v20: TabM grouped-OOF diversity lane with mandatory full refit.

This uses the locally installed tabm package directly. It compares compact
feature representations and a small predeclared recipe set, then blends the
winner with the existing v12/v18/v19 anchors. No Kaggle submission is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
from sklearn.metrics import f1_score, log_loss, roc_auc_score
import torch
import torch.nn.functional as F
import tabm

from baseline import check_submission
from v2_features import build_features, categorical_columns
from v17_xgb_optuna import historical_features
from v19_lightgbm import add_missing


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
PARENT = ROOT / "artifacts" / "baseline_v1"
OUT = ROOT / "artifacts" / "v20_tabm"


def now(): return datetime.now(timezone.utc).isoformat()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write_json(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_data():
    raw = ROOT / "data" / "raw"
    tr = pd.read_csv(raw / "train.csv")
    te = pd.read_csv(raw / "test.csv")
    y = pd.read_csv(raw / "train_labels.csv")[TARGET].to_numpy(np.int64)
    sample = pd.read_csv(raw / "submission.csv")
    man = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    dev = man.loc[man.partition.eq("development"), "row_id"].to_numpy(int)
    audit = man.loc[man.partition.eq("audit"), "row_id"].to_numpy(int)
    folds = man.loc[dev, "dev_fold"].to_numpy(int)
    return tr, te, y, sample, man, dev, audit, folds


def build_block(raw, block):
    if block == "raw_native":
        return build_features(raw, "raw")
    if block == "survey_native":
        return build_features(raw, "survey")
    if block == "historical_missing":
        return add_missing(historical_features(raw), raw)
    raise ValueError(block)


def fit_preprocessor(frame):
    cats = categorical_columns(frame)
    cats = [c for c in frame.columns if c in set(cats) or frame[c].dtype == object]
    nums = [c for c in frame.columns if c not in set(cats)]
    cat_maps = {}
    cat_cardinalities = []
    for c in cats:
        vals = frame[c].dropna().astype(str)
        levels = sorted(vals.unique().tolist())
        cat_maps[c] = {v: i + 1 for i, v in enumerate(levels)}
        cat_cardinalities.append(len(levels) + 1)  # 0 = missing/unseen
    medians, means, stds = {}, {}, {}
    for c in nums:
        s = pd.to_numeric(frame[c], errors="raise").astype(float)
        med = float(s.median()) if s.notna().any() else 0.0
        filled = s.fillna(med)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6: std = 1.0
        medians[c], means[c], stds[c] = med, mean, std
    return {"columns": list(frame.columns), "cats": cats, "nums": nums, "cat_maps": cat_maps,
            "cat_cardinalities": cat_cardinalities, "medians": medians, "means": means, "stds": stds}


def apply_preprocessor(frame, prep):
    if list(frame.columns) != prep["columns"]:
        raise ValueError("feature schema mismatch")
    if prep["nums"]:
        num = np.column_stack([
            ((pd.to_numeric(frame[c], errors="raise").astype(float).fillna(prep["medians"][c]).to_numpy() - prep["means"][c]) / prep["stds"][c])
            for c in prep["nums"]
        ]).astype(np.float32)
    else:
        num = np.empty((len(frame), 0), dtype=np.float32)
    if prep["cats"]:
        cols = []
        for c in prep["cats"]:
            mp = prep["cat_maps"][c]
            vals = frame[c]
            arr = np.zeros(len(frame), dtype=np.int64)
            mask = vals.notna().to_numpy()
            if mask.any():
                arr[mask] = vals[mask].astype(str).map(mp).fillna(0).astype(np.int64).to_numpy()
            cols.append(arr)
        cat = np.column_stack(cols).astype(np.int64)
    else:
        cat = np.empty((len(frame), 0), dtype=np.int64)
    return num, cat


def threshold_grid(cfg): return np.linspace(cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_steps"])
def choose_threshold(y, p, cfg):
    g = threshold_grid(cfg)
    s = np.array([f1_score(y, p >= t, zero_division=0) for t in g])
    ix = np.flatnonzero(np.isclose(s, s.max(), atol=1e-12, rtol=0)); j = ix[len(ix)//2]
    return float(g[j]), float(s[j])


def nested_metrics(y, p, folds, used_folds, cfg):
    used = np.isin(folds, used_folds) & np.isfinite(p)
    hard = np.zeros(len(y), dtype=bool); ts=[]; fs=[]
    for f in used_folds:
        tune = used & (folds != f); valid = folds == f
        t,_ = choose_threshold(y[tune], p[tune], cfg); ts.append(t)
        hard[valid] = p[valid] >= t
        fs.append(float(f1_score(y[valid], hard[valid], zero_division=0)))
    return {"nested_f1":float(f1_score(y[used],hard[used],zero_division=0)),"thresholds":ts,
            "median_threshold":float(statistics.median(ts)),"fold_f1":fs,
            "fold_std":float(np.std(fs,ddof=1)) if len(fs)>1 else 0.0}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_model(recipe, n_num, cat_cards, x_num_train, device):
    num_embeddings = None
    if recipe["num_emb_type"] == "pwl" and n_num > 0:
        bins = rtdl_num_embeddings.compute_bins(x_num_train.cpu(), n_bins=int(recipe["num_emb_n_bins"]))
        num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins=bins, d_embedding=int(recipe["d_embedding"]), activation=False, version="B"
        )
    model = tabm.TabM.make(
        n_num_features=int(n_num), cat_cardinalities=list(cat_cards), d_out=2,
        num_embeddings=num_embeddings, n_blocks=int(recipe["n_blocks"]),
        d_block=int(recipe["d_block"]), dropout=float(recipe["dropout"]),
        k=int(recipe["tabm_k"]), arch_type=recipe["arch_type"],
    )
    return model.to(device)


@torch.inference_mode()
def predict(model, x_num, x_cat, batch_size):
    model.eval(); out=[]
    n = len(x_num) if x_num is not None else len(x_cat)
    for start in range(0,n,batch_size):
        sl=slice(start,min(n,start+batch_size))
        logits=model(x_num[sl] if x_num is not None else None, x_cat[sl] if x_cat is not None else None)
        out.append(logits.softmax(-1)[...,1].mean(dim=1).float().cpu().numpy())
    return np.concatenate(out)


def train_model(recipe, xtr_num, xtr_cat, ytr, xva_num, xva_cat, yva, cfg, seed, fixed_epochs=None):
    seed_all(seed)
    device=torch.device(cfg["device"] if cfg["device"]=="cpu" or torch.cuda.is_available() else "cpu")
    tn=torch.from_numpy(xtr_num).to(device); tc=torch.from_numpy(xtr_cat).to(device) if xtr_cat.shape[1] else None
    vn=torch.from_numpy(xva_num).to(device); vc=torch.from_numpy(xva_cat).to(device) if xva_cat.shape[1] else None
    ty=torch.from_numpy(ytr.astype(np.int64)).to(device); vy=yva.astype(np.int64)
    model=make_model(recipe,tn.shape[1],recipe["cat_cardinalities"],tn,device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(recipe["lr"]),weight_decay=float(recipe["weight_decay"]))
    max_epochs=int(fixed_epochs if fixed_epochs is not None else cfg["max_epochs"]); patience=int(cfg["patience"])
    best_loss=float("inf"); best_epoch=0; best_state=None; stale=0; bs=int(recipe["batch_size"]); k=int(recipe["tabm_k"])
    started=time.perf_counter()
    for epoch in range(1,max_epochs+1):
        model.train(); perm=torch.randperm(len(ty),device=device)
        for idx in perm.split(bs):
            opt.zero_grad(set_to_none=True)
            logits=model(tn[idx],tc[idx] if tc is not None else None)
            loss=F.cross_entropy(logits.flatten(0,1),ty[idx].repeat_interleave(k))
            loss.backward(); opt.step()
        if fixed_epochs is not None: continue
        pv=predict(model,vn,vc,int(cfg["eval_batch_size"])); vl=float(log_loss(vy,pv,labels=[0,1]))
        if vl < best_loss - 1e-6:
            best_loss=vl; best_epoch=epoch; stale=0
            best_state={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if fixed_epochs is None:
        if best_state is None: raise RuntimeError("no best TabM state")
        model.load_state_dict(best_state); best_epoch=max(1,best_epoch)
    else:
        best_epoch=int(fixed_epochs)
    return model,best_epoch,time.perf_counter()-started,device


def fold_fit(block, recipe0, cfg, train, test, y, tr_ids, va_ids, audit_ids, seed, fixed_epochs=None):
    trf=build_block(train.iloc[tr_ids].reset_index(drop=True),block)
    vaf=build_block(train.iloc[va_ids].reset_index(drop=True),block)
    auf=build_block(train.iloc[audit_ids].reset_index(drop=True),block)
    tef=build_block(test.reset_index(drop=True),block)
    prep=fit_preprocessor(trf); trn,trc=apply_preprocessor(trf,prep); van,vac=apply_preprocessor(vaf,prep); aun,auc=apply_preprocessor(auf,prep); ten,tec=apply_preprocessor(tef,prep)
    recipe=deepcopy(recipe0); recipe["cat_cardinalities"]=prep["cat_cardinalities"]
    model,epoch,sec,device=train_model(recipe,trn,trc,y[tr_ids],van,vac,y[va_ids],cfg,seed,fixed_epochs)
    def tt(a,b): return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device) if b.shape[1] else None
    van_t,vac_t=tt(van,vac); aun_t,auc_t=tt(aun,auc); ten_t,tec_t=tt(ten,tec)
    pv=predict(model,van_t,vac_t,int(cfg["eval_batch_size"])); pa=predict(model,aun_t,auc_t,int(cfg["eval_batch_size"])); pt=predict(model,ten_t,tec_t,int(cfg["eval_batch_size"]))
    del model,van_t,vac_t,aun_t,auc_t,ten_t,tec_t
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pv,pa,pt,epoch,sec,prep


def cv_recipe(label,block,recipe,cfg,train,test,y,dev,audit,folds,used_folds,fixed_epochs=None,out_dir=None):
    if out_dir is not None:
        jp=out_dir/f"{label}.json"; npz=out_dir/f"{label}.npz"
        if jp.is_file() and npz.is_file(): return read_json(jp)
    oof=np.full(len(dev),np.nan); aps=[]; tps=[]; eps=[]; secs=[]
    for f in used_folds:
        tri=dev[folds!=f]; pos=np.flatnonzero(folds==f); vai=dev[pos]
        print(f"  fold={f} block={block} recipe={recipe['id']}",flush=True)
        pv,pa,pt,ep,sec,_=fold_fit(block,recipe,cfg,train,test,y,tri,vai,audit,int(cfg["seed"])+f*97+(sum(map(ord,label))%5000),fixed_epochs)
        oof[pos]=pv; aps.append(pa); tps.append(pt); eps.append(ep); secs.append(sec)
    mask=np.isfinite(oof); yd=y[dev]; t,tf=choose_threshold(yd[mask],oof[mask],cfg); nested=nested_metrics(yd,oof,folds,used_folds,cfg)
    r={"label":label,"feature_block":block,"recipe":recipe,"used_folds":list(used_folds),"nested":nested,"tuned_threshold":t,"tuned_f1":tf,"auc":float(roc_auc_score(yd[mask],oof[mask])),"logloss":float(log_loss(yd[mask],oof[mask],labels=[0,1])),"best_epochs":eps,"median_best_epoch":int(statistics.median(eps)),"fold_seconds":secs}
    if len(used_folds)==5: r["audit_f1"]=float(f1_score(y[audit],np.mean(aps,axis=0)>=nested["median_threshold"],zero_division=0))
    if out_dir is not None:
        out_dir.mkdir(parents=True,exist_ok=True); write_json(jp,r); np.savez_compressed(npz,oof=oof,audit=np.mean(aps,axis=0),test=np.mean(tps,axis=0))
    return r


def rank01(p): return pd.Series(p).rank(method="average",pct=True).to_numpy(float)
def mix(arrs,weights,kind):
    xs=[rank01(x) for x in arrs] if kind=="rank" else arrs
    return sum(float(w)*x for w,x in zip(weights,xs))


def ensemble_scan(cfg,ydev,folds,tabp):
    man=pd.read_csv(PARENT/"split_manifest.csv"); dev=man.loc[man.partition.eq("development"),"row_id"].to_numpy(int)
    v12=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/selected_oof.csv").set_index("row_id").loc[dev,"probability"].to_numpy(float)
    xgb=np.load(ROOT/"artifacts/v18_xgb_finalize_ensemble/fixed_round/trial59_r1367.npz")["oof"]
    lgb=np.load(ROOT/"artifacts/v19_lightgbm/fixed/trial22_r192.npz")["oof"]
    rows=[]
    def add(name,arrs,w,kind):
        p=mix(arrs,w,kind); n=nested_metrics(ydev,p,folds,list(range(5)),cfg); t,tf=choose_threshold(ydev,p,cfg)
        rows.append({"name":name,"kind":kind,"weights":w,"nested":n,"tuned_threshold":t,"tuned_f1":tf,"auc":float(roc_auc_score(ydev,p))})
    for kind in ["prob","rank"]:
        for wt in [0.10,0.20,0.30]: add(f"v12_tabm_{kind}_{int(wt*100)}",[v12,tabp],[1-wt,wt],kind)
        for wt in [0.10,0.15,0.20]: add(f"v12_xgb_tabm_{kind}_{int(wt*100)}",[v12,xgb,tabp],[0.6,0.4-wt,wt],kind)
        for wt in cfg["ensemble_tabm_weights"]:
            scale=1-float(wt); w=[0.6*scale,0.3*scale,0.1*scale,float(wt)]
            add(f"v19base_plus_tabm_{kind}_{int(wt*100)}",[v12,xgb,lgb,tabp],w,kind)
    rows.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True)
    return rows,{"v12":v12,"xgb":xgb,"lgb":lgb}


def full_refit(best,cfg,train,test,y,sample):
    block=best["feature_block"]; recipe0=best["recipe"]; epochs=int(best["median_best_epoch"]); final=OUT/"final"; final.mkdir(parents=True,exist_ok=True)
    trf=build_block(train.reset_index(drop=True),block); tef=build_block(test.reset_index(drop=True),block); prep=fit_preprocessor(trf); trn,trc=apply_preprocessor(trf,prep); ten,tec=apply_preprocessor(tef,prep)
    joblib.dump(prep,final/"preprocessor.joblib",compress=3); preds=[]; models=[]
    for seed in cfg["full_seeds"]:
        pp=final/f"test_seed{seed}.npy"; mp=final/f"tabm_seed{seed}.pt"
        if pp.is_file() and mp.is_file(): p=np.load(pp); sec=None
        else:
            recipe=deepcopy(recipe0); recipe["cat_cardinalities"]=prep["cat_cardinalities"]
            # fixed-epoch full refit; a tiny held-out tensor is needed only by the shared trainer API and is never used for stopping.
            model,_,sec,device=train_model(recipe,trn,trc,y,ten[:1],tec[:1],np.array([0],dtype=np.int64),cfg,int(seed),epochs)
            tn=torch.from_numpy(ten).to(device); tc=torch.from_numpy(tec).to(device) if tec.shape[1] else None; p=predict(model,tn,tc,int(cfg["eval_batch_size"])); np.save(pp,p)
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"recipe":recipe0,"block":block,"epochs":epochs,"cat_cardinalities":prep["cat_cardinalities"]},mp)
            del model,tn,tc
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        preds.append(p); models.append({"seed":int(seed),"path":str(mp.relative_to(ROOT)).replace("\\","/"),"sha256":sha(mp),"seconds":sec})
    arr=np.vstack(preds); return arr[0],arr.mean(axis=0),models


def make_candidate(sample,name,p,t):
    sub=sample.copy(); sub[TARGET]=(p>=t).astype(np.int64); path=ROOT/"submissions"/name; sub.to_csv(path,index=False); ck=check_submission(sample,pd.read_csv(path)); return {"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(path),**ck}


def run(cfg):
    if (OUT/"results.json").is_file(): print("v20 already complete; immutable results preserved",flush=True); return
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/"config.json",cfg)
    train,test,y,sample,man,dev,audit,folds=load_data(); yd=y[dev]
    screens=[]
    for b in cfg["feature_blocks"]:
        for recipe in cfg["recipes"]:
            label=f"{b}_{recipe['id']}"; print(f"SCREEN {label}",flush=True)
            r=cv_recipe(label,b,recipe,cfg,train,test,y,dev,audit,folds,cfg["screen_folds"],None,OUT/"screen"); screens.append(r)
    screens.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True); write_json(OUT/"screen_leaderboard.json",screens)
    full=[]
    seen=set()
    for s in screens:
        key=(s["feature_block"],s["recipe"]["id"])
        if key in seen: continue
        seen.add(key)
        print(f"CONFIRM {s['label']}",flush=True); r=cv_recipe(s["label"],s["feature_block"],s["recipe"],cfg,train,test,y,dev,audit,folds,list(range(5)),None,OUT/"confirm"); full.append(r)
        if len(full)>=int(cfg["screen_top_k"]): break
    full.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True); write_json(OUT/"confirmation_leaderboard.json",full); best=full[0]
    bp=np.load(OUT/"confirm"/f"{best['label']}.npz"); tabp=bp["oof"]
    ens,anchors=ensemble_scan(cfg,yd,folds,tabp); write_json(OUT/"ensemble_scan.json",ens); ensbest=ens[0]; write_json(OUT/"ensemble_selection.json",ensbest)
    diversity={"tabm_v12_corr":float(np.corrcoef(tabp,anchors["v12"])[0,1]),"tabm_xgb_corr":float(np.corrcoef(tabp,anchors["xgb"])[0,1]),"tabm_lgb_corr":float(np.corrcoef(tabp,anchors["lgb"])[0,1])}
    tab_t=float(best["nested"]["median_threshold"]); diversity.update({"tabm_v12_hard_disagreement":int(np.sum((tabp>=tab_t)!=(anchors["v12"]>=.315))),"tabm_xgb_hard_disagreement":int(np.sum((tabp>=tab_t)!=(anchors["xgb"]>=.36)))})
    single,seed3,models=full_refit(best,cfg,train,test,y,sample)
    stand={"single":make_candidate(sample,"v20_tabm_full_single.csv",single,tab_t),"seed3":make_candidate(sample,"v20_tabm_full_seed3.csv",seed3,tab_t)}
    v12t=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/test_probabilities.csv").probability.to_numpy(float); xgbt=pd.read_csv(ROOT/"artifacts/v18_xgb_finalize_ensemble/final_xgb/test_single.csv").probability.to_numpy(float); lgbt=pd.read_csv(ROOT/"artifacts/v19_lightgbm/final/test_seed5.csv").probability.to_numpy(float)
    ec={}
    for nm,tp in [("single",single),("seed3",seed3)]:
        name=ensbest["name"]; w=ensbest["weights"]; kind=ensbest["kind"]
        if name.startswith("v12_tabm"): arrs=[v12t,tp]
        elif name.startswith("v12_xgb_tabm"): arrs=[v12t,xgbt,tp]
        else: arrs=[v12t,xgbt,lgbt,tp]
        p=mix(arrs,w,kind); th=float(ensbest["nested"]["median_threshold"]); ec[nm]=make_candidate(sample,f"v20_{name}_{nm}.csv",p,th)
    result={"version":cfg["version"],"completed_utc":now(),"screen_top":screens[:8],"confirmation":full,"selected":best,"diversity":diversity,"ensemble_selected":ensbest,"full_refit":{"training_rows":len(train),"all_available_labels_used":True,"models":models,"standalone_candidates":stand,"ensemble_candidates":ec},"submitted":False}
    write_json(OUT/"results.json",result); print(json.dumps({"completed":True,"selected":best,"diversity":diversity,"ensemble":ensbest,"full":result["full_refit"]},indent=2,ensure_ascii=False),flush=True)


def check(cfg):
    tr,te,y,sample,man,dev,audit,folds=load_data(); shapes={}
    for b in cfg["feature_blocks"]: shapes[b]=int(build_block(tr.head(200),b).shape[1])
    payload={"check_passed":True,"tabm":getattr(tabm,"__version__","unknown"),"torch":torch.__version__,"cuda":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"train":len(tr),"dev":len(dev),"audit":len(audit),"test":len(te),"feature_counts":shapes,"recipes":[x["id"] for x in cfg["recipes"]],"full_refit_required":True,"remote_submission_authorized":False}
    write_json(ROOT/"reports"/"v20_tabm_preflight.json",payload); print(json.dumps(payload,indent=2))


def smoke(cfg):
    tr,te,y,sample,man,dev,audit,folds=load_data(); recipe=deepcopy(cfg["recipes"][1]); recipe["id"]="smoke"; local=deepcopy(cfg); local["max_epochs"]=2; local["patience"]=2
    tri=dev[folds!=0][:6000]; vai=dev[folds==0][:1500]; pv,pa,pt,ep,sec,_=fold_fit("raw_native",recipe,local,tr,te,y,tri,vai,audit,123,2)
    payload={"smoke_passed":True,"auc":float(roc_auc_score(y[vai],pv)),"seconds":sec,"epoch":ep,"cuda":torch.cuda.is_available()}; write_json(ROOT/"reports"/"v20_tabm_smoke.json",payload); print(json.dumps(payload,indent=2))


def status(cfg):
    print(json.dumps({"screen":len(list((OUT/"screen").glob("*.json"))) if (OUT/"screen").exists() else 0,"screen_target":len(cfg["feature_blocks"])*len(cfg["recipes"]),"confirm":len(list((OUT/"confirm").glob("*.json"))) if (OUT/"confirm").exists() else 0,"confirm_target":cfg["screen_top_k"],"full_models":len(list((OUT/"final").glob("*.pt"))) if (OUT/"final").exists() else 0,"results":(OUT/"results.json").is_file()},indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["check","smoke","run","status"]); ap.add_argument("--config",type=Path,default=ROOT/"configs"/"v20_tabm.json"); a=ap.parse_args(); cfg=read_json(a.config)
    if a.phase=="check": check(cfg)
    elif a.phase=="smoke": smoke(cfg)
    elif a.phase=="status": status(cfg)
    else: run(cfg)


if __name__=="__main__": main()
