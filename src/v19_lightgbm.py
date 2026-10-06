"""Focused LightGBM preprocessing/HPO/diversity campaign on frozen grouped folds.

The old v10 LightGBM TE/WOE lane is intentionally not repeated. This version
tests native categorical representations with progressively stronger row-local
preprocessing, then lightly tunes only the best preprocessing blocks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import zlib

import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from baseline import check_submission
from v2_features import build_features, fit_schema, transform
from v17_xgb_optuna import historical_features


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
OUT = ROOT / "artifacts" / "v19_lightgbm"
STUDY_DB = OUT / "study.db"
PARENT = ROOT / "artifacts" / "baseline_v1"


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
    y = pd.read_csv(raw / "train_labels.csv")[TARGET].to_numpy(int)
    sample = pd.read_csv(raw / "submission.csv")
    man = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    dev = man.loc[man.partition.eq("development"), "row_id"].to_numpy(int)
    audit = man.loc[man.partition.eq("audit"), "row_id"].to_numpy(int)
    folds = man.loc[dev, "dev_fold"].to_numpy(int)
    return tr, te, y, sample, man, dev, audit, folds


def token(s): return s.where(s.notna(), "__MISSING__").astype(str)


def fit_freq(raw, cols):
    return {c: (token(raw[c]).value_counts(dropna=False) / len(raw)).to_dict() for c in cols}


def add_freq(x, raw, maps, cols):
    z = x.copy()
    for c in cols:
        z[f"v19_freq_{c}"] = token(raw[c]).map(maps[c]).fillna(0.0).astype(float)
    return z


def add_missing(x, raw):
    z = x.copy()
    h1 = [c for c in raw if c.startswith("opinion_h1n1_")]
    seas = [c for c in raw if c.startswith("opinion_seas_")]
    z["v19_employment_pair_missing"] = (raw.employment_occupation.isna() & raw.employment_industry.isna()).astype(float)
    z["v19_employment_status_missing"] = raw.employment_status.isna().astype(float)
    z["v19_doctor_any_missing"] = raw[["doctor_recc_h1n1", "doctor_recc_seasonal"]].isna().any(axis=1).astype(float)
    z["v19_doctor_both_missing"] = raw[["doctor_recc_h1n1", "doctor_recc_seasonal"]].isna().all(axis=1).astype(float)
    z["v19_h1n1_opinion_all_missing"] = raw[h1].isna().all(axis=1).astype(float)
    z["v19_seas_opinion_all_missing"] = raw[seas].isna().all(axis=1).astype(float)
    z["v19_demographic_missing_count"] = raw[["education_comp", "marital", "rent_own_r", "employment_status"]].isna().sum(axis=1).astype(float)
    z["v19_health_missing_count"] = raw[["health_insurance", "health_worker", "chronic_med_condition", "child_under_6_months", "doctor_recc_h1n1", "doctor_recc_seasonal"]].isna().sum(axis=1).astype(float)
    z["v19_h1n1_unknown_count"] = raw[h1].isin(["Dont Know", "Don't Know", "Refused"]).sum(axis=1).astype(float)
    z["v19_seas_unknown_count"] = raw[seas].isin(["Dont Know", "Don't Know", "Refused"]).sum(axis=1).astype(float)
    z["v19_any_refused"] = raw[[*h1, *seas]].eq("Refused").any(axis=1).astype(float)
    return z


def base_block(raw, block):
    if block == "raw_native": return build_features(raw, "raw")
    if block.startswith("survey_"): return build_features(raw, "survey")
    if block == "historical_missing": return historical_features(raw)
    raise ValueError(block)


def prepare_pair(train_raw, other_raw, block, cfg):
    """Fit all target-free fold-local preprocessing on train_raw only."""
    xtr = base_block(train_raw, block)
    xother = base_block(other_raw, block)
    if "missing" in block:
        xtr = add_missing(xtr, train_raw)
        xother = add_missing(xother, other_raw)
    if "freq" in block:
        maps = fit_freq(train_raw, cfg["frequency_columns"])
        xtr = add_freq(xtr, train_raw, maps, cfg["frequency_columns"])
        xother = add_freq(xother, other_raw, maps, cfg["frequency_columns"])
    schema = fit_schema(xtr, "lightgbm")
    return transform(xtr, schema), transform(xother, schema), schema


def transform_with_schema(train_raw, other_raw, block, cfg, schema, freq_maps=None):
    x = base_block(other_raw, block)
    if "missing" in block: x = add_missing(x, other_raw)
    if "freq" in block:
        if freq_maps is None: freq_maps = fit_freq(train_raw, cfg["frequency_columns"])
        x = add_freq(x, other_raw, freq_maps, cfg["frequency_columns"])
    return transform(x, schema)


def prepare_many(train_raw, frames, block, cfg):
    xtr = base_block(train_raw, block)
    if "missing" in block: xtr = add_missing(xtr, train_raw)
    maps = None
    if "freq" in block:
        maps = fit_freq(train_raw, cfg["frequency_columns"])
        xtr = add_freq(xtr, train_raw, maps, cfg["frequency_columns"])
    schema = fit_schema(xtr, "lightgbm")
    xtr = transform(xtr, schema)
    outs = []
    for raw in frames:
        x = base_block(raw, block)
        if "missing" in block: x = add_missing(x, raw)
        if "freq" in block: x = add_freq(x, raw, maps, cfg["frequency_columns"])
        outs.append(transform(x, schema))
    return xtr, outs, schema, maps


def threshold_grid(cfg): return np.linspace(cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_steps"])


def choose_threshold(y, p, cfg):
    grid = threshold_grid(cfg)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    ix = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    j = ix[len(ix)//2]
    return float(grid[j]), float(scores[j])


def nested_metrics(y, p, folds, used_folds, cfg):
    used = np.isin(folds, used_folds) & np.isfinite(p)
    hard = np.zeros(len(y), dtype=bool)
    fold_f1, ts = [], []
    for f in used_folds:
        valid = folds == f
        tune = used & (folds != f)
        t, _ = choose_threshold(y[tune], p[tune], cfg)
        ts.append(t)
        hard[valid] = p[valid] >= t
        fold_f1.append(float(f1_score(y[valid], hard[valid], zero_division=0)))
    return {
        "nested_f1": float(f1_score(y[used], hard[used], zero_division=0)),
        "thresholds": ts,
        "median_threshold": float(statistics.median(ts)),
        "fold_f1": fold_f1,
        "fold_std": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
    }


def model_from_params(params, cfg, seed, rounds=None):
    p = dict(params)
    p["n_estimators"] = int(rounds if rounds is not None else p.get("n_estimators", cfg["base_params"]["n_estimators"]))
    p.update({
        "objective": "binary", "verbosity": -1, "n_jobs": int(cfg["threads"]),
        "random_state": int(seed), "deterministic": True, "force_col_wise": True,
    })
    return LGBMClassifier(**p)


def fit_fold(params, block, cfg, train, test, y, tr_ids, va_ids, audit_ids, seed, early_stop=True, rounds=None):
    train_raw = train.iloc[tr_ids].reset_index(drop=True)
    va_raw = train.iloc[va_ids].reset_index(drop=True)
    audit_raw = train.iloc[audit_ids].reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    xtr, others, schema, maps = prepare_many(train_raw, [va_raw, audit_raw, test_raw], block, cfg)
    xva, xa, xt = others
    model = model_from_params(params, cfg, seed, rounds)
    started = time.perf_counter()
    if early_stop:
        model.fit(xtr, y[tr_ids], eval_set=[(xva, y[va_ids])], eval_metric="binary_logloss",
                  categorical_feature=schema["categorical"],
                  callbacks=[lgb.early_stopping(int(cfg["early_stopping_rounds"]), verbose=False), lgb.log_evaluation(0)])
        best_iter = int(model.best_iteration_ or model.n_estimators_)
    else:
        model.fit(xtr, y[tr_ids], categorical_feature=schema["categorical"], callbacks=[lgb.log_evaluation(0)])
        best_iter = int(rounds)
    sec = time.perf_counter() - started
    return model, schema, maps, model.predict_proba(xva)[:,1], model.predict_proba(xa)[:,1], model.predict_proba(xt)[:,1], best_iter, sec


def cv_recipe(label, block, params, cfg, train, test, y, dev, audit, folds, used_folds, early_stop=True, rounds=None, out_dir=None):
    if out_dir is not None:
        jp = out_dir / f"{label}.json"; npz = out_dir / f"{label}.npz"
        if jp.is_file() and npz.is_file(): return read_json(jp)
    ydev = y[dev]
    oof = np.full(len(dev), np.nan)
    audit_parts, test_parts, best_iters, seconds = [], [], [], []
    for f in used_folds:
        tr_ids = dev[folds != f]
        va_pos = np.flatnonzero(folds == f)
        va_ids = dev[va_pos]
        stable_offset = zlib.crc32(label.encode("utf-8")) % 10000
        _, _, _, pv, pa, pt, bi, sec = fit_fold(params, block, cfg, train, test, y, tr_ids, va_ids, audit, int(cfg["seed"]) + f + stable_offset, early_stop, rounds)
        oof[va_pos] = pv; audit_parts.append(pa); test_parts.append(pt); best_iters.append(bi); seconds.append(sec)
    mask = np.isfinite(oof)
    t, tf1 = choose_threshold(ydev[mask], oof[mask], cfg)
    nested = nested_metrics(ydev, oof, folds, used_folds, cfg)
    result = {
        "label": label, "feature_block": block, "params": params, "used_folds": list(used_folds),
        "nested": nested, "tuned_threshold": t, "tuned_f1": tf1,
        "auc": float(roc_auc_score(ydev[mask], oof[mask])),
        "logloss": float(log_loss(ydev[mask], oof[mask], labels=[0,1])),
        "best_iterations": best_iters, "median_best_iteration": int(statistics.median(best_iters)),
        "fold_seconds": seconds,
    }
    if len(used_folds) == 5:
        result["audit_f1"] = float(f1_score(y[audit], np.mean(audit_parts,axis=0) >= nested["median_threshold"], zero_division=0))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True); write_json(jp, result)
        np.savez_compressed(npz, oof=oof, audit=np.mean(audit_parts,axis=0), test=np.mean(test_parts,axis=0))
    return result


def ensure_study(cfg, allowed_blocks):
    import optuna
    OUT.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]), n_startup_trials=12)
    study = optuna.create_study(study_name="v19_lightgbm", storage=f"sqlite:///{STUDY_DB.as_posix()}", direction="maximize", sampler=sampler, load_if_exists=True)
    study.set_user_attr("allowed_blocks", list(allowed_blocks))
    return study


def objective_factory(cfg, allowed_blocks, train, test, y, dev, audit, folds):
    def objective(trial):
        block = trial.suggest_categorical("feature_block", list(allowed_blocks))
        params = {
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1,4,5,6,7]),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 180, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 0.05, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 60.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 1.0),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "subsample_freq": 1,
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
            "max_bin": trial.suggest_categorical("max_bin", [63,127,255]),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.2, log=True),
        }
        r = cv_recipe(f"screen_t{trial.number}", block, params, cfg, train, test, y, dev, audit, folds, cfg["hpo_screen_folds"], True, None, None)
        trial.set_user_attr("nested_f1", r["nested"]["nested_f1"])
        trial.set_user_attr("fold_std", r["nested"]["fold_std"])
        trial.set_user_attr("tuned_f1", r["tuned_f1"])
        trial.set_user_attr("auc", r["auc"])
        trial.set_user_attr("median_best_iteration", r["median_best_iteration"])
        return r["nested"]["nested_f1"] - 0.10 * r["nested"]["fold_std"]
    return objective


def completed_trials(study):
    import optuna
    return sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None], key=lambda t: float(t.value), reverse=True)


def params_from_trial(t):
    p = dict(t.params); p.pop("feature_block", None); p["n_estimators"] = 4000; p["subsample_freq"] = 1
    return p


def rank01(p): return pd.Series(p).rank(method="average", pct=True).to_numpy(float)
def blend(arrs, weights, kind):
    xs = [rank01(x) for x in arrs] if kind == "rank" else arrs
    return sum(float(w)*x for w,x in zip(weights,xs))


def ensemble_scan(cfg, ydev, folds, lgb_oof):
    v12 = pd.read_csv(ROOT / "artifacts/v12_exact_v5_fullstack/selected_oof.csv").set_index("row_id")
    man = pd.read_csv(PARENT / "split_manifest.csv")
    dev = man.loc[man.partition.eq("development"), "row_id"].to_numpy(int)
    v12p = v12.loc[dev, "probability"].to_numpy(float)
    xgbp = np.load(ROOT / "artifacts/v18_xgb_finalize_ensemble/fixed_round/trial59_r1367.npz")["oof"]
    rows = []
    def add(name, arrs, weights, kind):
        p = blend(arrs, weights, kind); n = nested_metrics(ydev,p,folds,list(range(5)),cfg); t,tf = choose_threshold(ydev,p,cfg)
        rows.append({"name":name,"kind":kind,"weights":weights,"nested":n,"tuned_threshold":t,"tuned_f1":tf,"auc":float(roc_auc_score(ydev,p))})
    for kind in ["prob","rank"]:
        for w in [0.5,0.6,0.7,0.8,0.9]: add(f"v12_lgb_{kind}_{int(w*100)}",[v12p,lgb_oof],[w,1-w],kind)
        for w in [0.5,0.6,0.7,0.8]: add(f"xgb_lgb_{kind}_{int(w*100)}",[xgbp,lgb_oof],[w,1-w],kind)
        for w12 in [0.4,0.5,0.6,0.7]:
            for wx in [0.1,0.2,0.3,0.4]:
                wl = round(1-w12-wx,10)
                if 0.1 <= wl <= 0.4: add(f"tri_{kind}_{int(w12*100)}_{int(wx*100)}_{int(wl*100)}",[v12p,xgbp,lgb_oof],[w12,wx,wl],kind)
    rows.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True)
    return rows, v12p, xgbp


def full_refit(best, cfg, train, test, y, sample):
    final = OUT / "final"; final.mkdir(parents=True, exist_ok=True)
    block, params, rounds = best["feature_block"], best["params"], int(best["rounds"])
    xall, others, schema, maps = prepare_many(train.reset_index(drop=True), [test.reset_index(drop=True)], block, cfg); xt = others[0]
    preds=[]; models=[]
    for seed in cfg["full_seeds"]:
        mp=final/f"lgb_full_seed{seed}.joblib"; pp=final/f"test_seed{seed}.npy"
        if mp.is_file() and pp.is_file(): p=np.load(pp); sec=None
        else:
            m=model_from_params(params,cfg,int(seed),rounds); st=time.perf_counter(); m.fit(xall,y,categorical_feature=schema["categorical"],callbacks=[lgb.log_evaluation(0)]); sec=time.perf_counter()-st; p=m.predict_proba(xt)[:,1]
            joblib.dump({"model":m,"schema":schema,"freq_maps":maps,"block":block,"params":params,"rounds":rounds},mp,compress=3); np.save(pp,p)
        preds.append(p); models.append({"seed":int(seed),"path":str(mp.relative_to(ROOT)).replace("\\","/"),"sha256":sha(mp),"seconds":sec})
    arr=np.vstack(preds); single=arr[0]; seed5=arr.mean(axis=0)
    pd.DataFrame({"Id":sample.Id,"probability":single}).to_csv(final/"test_single.csv",index=False)
    pd.DataFrame({"Id":sample.Id,"probability":seed5}).to_csv(final/"test_seed5.csv",index=False)
    return single,seed5,models


def make_candidate(sample, name, p, t):
    sub=sample.copy(); sub[TARGET]=(p>=t).astype(np.int64); path=ROOT/"submissions"/name; sub.to_csv(path,index=False); ck=check_submission(sample,pd.read_csv(path)); return {"path":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":sha(path),**ck}


def run(cfg):
    if (OUT/"results.json").is_file(): print("v19 already complete; immutable results preserved",flush=True); return
    OUT.mkdir(parents=True,exist_ok=True); write_json(OUT/"config.json",cfg)
    train,test,y,sample,man,dev,audit,folds=load_data(); ydev=y[dev]

    blocks=[]
    for b in cfg["feature_blocks"]:
        print(f"BLOCK {b}",flush=True); r=cv_recipe(f"block_{b}",b,cfg["base_params"],cfg,train,test,y,dev,audit,folds,list(range(5)),True,None,OUT/"blocks"); blocks.append(r)
    blocks.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True); write_json(OUT/"block_leaderboard.json",blocks)
    allowed=[r["feature_block"] for r in blocks[:int(cfg["hpo_feature_top_k"])]]

    study=ensure_study(cfg,allowed); done=completed_trials(study); remain=max(0,int(cfg["hpo_trials"])-len(done))
    if remain:
        print(f"OPTUNA completed={len(done)} remaining={remain} blocks={allowed}",flush=True)
        study.optimize(objective_factory(cfg,allowed,train,test,y,dev,audit,folds),n_trials=remain,timeout=int(cfg["hpo_timeout_seconds"]),gc_after_trial=True,show_progress_bar=False)
    done=completed_trials(study)
    pd.DataFrame([{"trial":t.number,"value":t.value,**t.params,**{f"attr_{k}":v for k,v in t.user_attrs.items()}} for t in done]).to_csv(OUT/"optuna_trials.csv",index=False)

    confirms=[]
    for t in done[:int(cfg["confirm_top_k"])]:
        p=params_from_trial(t); b=t.params["feature_block"]; print(f"CONFIRM trial={t.number} {b}",flush=True)
        r=cv_recipe(f"trial_{t.number}",b,p,cfg,train,test,y,dev,audit,folds,list(range(5)),True,None,OUT/"confirm"); r["trial"]=int(t.number); r["optuna_value"]=float(t.value); confirms.append(r)
    confirms.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True); write_json(OUT/"confirmation_leaderboard.json",confirms)

    fixed=[]
    for c in confirms[:int(cfg["fixed_top_k"])]:
        base=max(20,int(c["median_best_iteration"]))
        for mult in cfg["round_multipliers"]:
            rounds=max(20,int(round(base*float(mult)))); label=f"trial{c['trial']}_r{rounds}"; print(f"FIXED {label}",flush=True)
            r=cv_recipe(label,c["feature_block"],c["params"],cfg,train,test,y,dev,audit,folds,list(range(5)),False,rounds,OUT/"fixed"); r["trial"]=c["trial"]; r["rounds"]=rounds; fixed.append(r)
    fixed.sort(key=lambda r:(r["nested"]["nested_f1"],-r["nested"]["fold_std"],r["tuned_f1"],r["auc"]),reverse=True); write_json(OUT/"fixed_leaderboard.json",fixed)
    best=fixed[0]; write_json(OUT/"selection.json",{k:v for k,v in best.items() if k not in {"fold_seconds","best_iterations","used_folds"}})
    bp=np.load(OUT/"fixed"/f"{best['label']}.npz"); lgb_oof=bp["oof"]

    ensembles,v12p,xgbp=ensemble_scan(cfg,ydev,folds,lgb_oof); write_json(OUT/"ensemble_scan.json",ensembles); ens=ensembles[0]; write_json(OUT/"ensemble_selection.json",ens)
    corr={"lgb_v12":float(np.corrcoef(lgb_oof,v12p)[0,1]),"lgb_xgb":float(np.corrcoef(lgb_oof,xgbp)[0,1])}
    t_lgb=float(best["nested"]["median_threshold"]); p_lgb=lgb_oof>=t_lgb; p12=v12p>=.315; px=xgbp>=.36
    corr["lgb_v12_hard_disagreement"]=int(np.sum(p_lgb!=p12)); corr["lgb_xgb_hard_disagreement"]=int(np.sum(p_lgb!=px))

    single,seed5,models=full_refit(best,cfg,train,test,y,sample)
    standalone={"single":make_candidate(sample,"v19_lgbm_full_single.csv",single,t_lgb),"seed5":make_candidate(sample,"v19_lgbm_full_seed5.csv",seed5,t_lgb)}
    v12t=pd.read_csv(ROOT/"artifacts/v12_exact_v5_fullstack/test_probabilities.csv").probability.to_numpy(float)
    xgbt=pd.read_csv(ROOT/"artifacts/v18_xgb_finalize_ensemble/final_xgb/test_single.csv").probability.to_numpy(float)
    ens_cands={}
    for lname,lp in [("single",single),("seed5",seed5)]:
        name=ens["name"]; w=ens["weights"]; kind=ens["kind"]
        if name.startswith("v12_lgb"): arrs=[v12t,lp]
        elif name.startswith("xgb_lgb"): arrs=[xgbt,lp]
        else: arrs=[v12t,xgbt,lp]
        pt=blend(arrs,w,kind); th=float(ens["nested"]["median_threshold"])
        ens_cands[lname]=make_candidate(sample,f"v19_{name}_{lname}.csv",pt,th)

    result={"version":cfg["version"],"completed_utc":now(),"block_leaderboard":blocks,"allowed_hpo_blocks":allowed,"optuna_completed_trials":len(done),"confirmation_top":confirms,"fixed_top":fixed,"selected":{k:v for k,v in best.items() if k not in {"fold_seconds","best_iterations","used_folds"}},"diversity":corr,"ensemble_selected":ens,"full_refit":{"training_rows":len(train),"all_available_labels_used":True,"models":models,"standalone_candidates":standalone,"ensemble_candidates":ens_cands},"submitted":False}
    write_json(OUT/"results.json",result); print(json.dumps({"completed":True,"selected":result["selected"],"diversity":corr,"ensemble":ens,"candidates":result["full_refit"]},indent=2,ensure_ascii=False),flush=True)


def check(cfg):
    train,test,y,sample,man,dev,audit,folds=load_data(); shapes={}
    for b in cfg["feature_blocks"]:
        x=base_block(train.head(200),b)
        if "missing" in b: x=add_missing(x,train.head(200))
        if "freq" in b: x=add_freq(x,train.head(200),fit_freq(train.head(200),cfg["frequency_columns"]),cfg["frequency_columns"])
        shapes[b]=int(x.shape[1])
    payload={"check_passed":True,"lightgbm":lgb.__version__,"train":len(train),"dev":len(dev),"audit":len(audit),"test":len(test),"fold_counts":{str(f):int(np.sum(folds==f)) for f in range(5)},"feature_counts":shapes,"hpo_trials":cfg["hpo_trials"],"full_refit_required":True,"remote_submission_authorized":False}
    write_json(ROOT/"reports"/"v19_lightgbm_preflight.json",payload); print(json.dumps(payload,indent=2))


def smoke(cfg):
    train,test,y,sample,man,dev,audit,folds=load_data(); rows=[]
    for b in cfg["feature_blocks"]:
        tr_ids=dev[folds!=0][:8000]; va_ids=dev[folds==0][:2000]
        _,_,_,pv,_,_,bi,sec=fit_fold(cfg["base_params"],b,cfg,train,test,y,tr_ids,va_ids,audit,123,True,None)
        rows.append({"block":b,"auc":float(roc_auc_score(y[va_ids],pv)),"best_iteration":bi,"seconds":sec})
    payload={"smoke_passed":True,"rows":rows}; write_json(ROOT/"reports"/"v19_lightgbm_smoke.json",payload); print(json.dumps(payload,indent=2))


def status(cfg):
    nblocks=len(list((OUT/"blocks").glob("*.json"))) if (OUT/"blocks").exists() else 0
    nconf=len(list((OUT/"confirm").glob("*.json"))) if (OUT/"confirm").exists() else 0
    nfixed=len(list((OUT/"fixed").glob("*.json"))) if (OUT/"fixed").exists() else 0
    ntrials=0
    if STUDY_DB.is_file():
        try:
            import optuna; s=optuna.load_study(study_name="v19_lightgbm",storage=f"sqlite:///{STUDY_DB.as_posix()}"); ntrials=len(completed_trials(s))
        except Exception: pass
    print(json.dumps({"blocks":nblocks,"block_target":len(cfg["feature_blocks"]),"optuna_complete":ntrials,"optuna_target":cfg["hpo_trials"],"confirm":nconf,"confirm_target":cfg["confirm_top_k"],"fixed":nfixed,"fixed_target":cfg["fixed_top_k"]*len(cfg["round_multipliers"]),"results":(OUT/"results.json").is_file()},indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["check","smoke","run","status"]); ap.add_argument("--config",type=Path,default=ROOT/"configs"/"v19_lightgbm.json"); a=ap.parse_args(); cfg=read_json(a.config)
    if a.phase=="check": check(cfg)
    elif a.phase=="smoke": smoke(cfg)
    elif a.phase=="status": status(cfg)
    else: run(cfg)


if __name__=="__main__": main()
