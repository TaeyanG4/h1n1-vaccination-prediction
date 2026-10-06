"""v17: aggressive but validation-disciplined XGBoost Optuna search.

Protocol
--------
1. Search on frozen grouped development folds 0/1/2 only.
2. Objective uses cross-fold threshold transfer, not a threshold tuned on the
   same validation fold being scored.
3. Re-evaluate the top Optuna trials on all five frozen grouped dev folds.
4. Freeze one hyperparameter recipe using five-fold nested-threshold F1 only.
5. Re-check a small set of fixed boosting-round multipliers on all five folds.
6. Freeze a robust threshold from CV only.
7. Refit the frozen recipe on all 42,154 labeled rows for every final seed.
8. Emit single-seed and 5-seed full-refit submission candidates. Never submit.

The Optuna SQLite study is resumable.  Confirmation and round-selection
artifacts are also cached so a long interrupted run can continue safely.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import joblib
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder as CEOrdinalEncoder
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from baseline import check_submission
from v2_features import build_features
from v3_pipeline import fit_xgb_schema, transform_xgb


ROOT = Path(__file__).resolve().parents[1]
TARGET = "vacc_h1n1_f"
PARENT = ROOT / "artifacts" / "baseline_v1"
OUT = ROOT / "artifacts" / "v17_xgb_optuna"
STUDY_DB = OUT / "study.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_grid(cfg: dict) -> np.ndarray:
    return np.linspace(float(cfg["threshold_min"]), float(cfg["threshold_max"]), int(cfg["threshold_steps"]))


def select_threshold(y: np.ndarray, p: np.ndarray, cfg: dict) -> tuple[float, float]:
    grid = threshold_grid(cfg)
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])


def nested_threshold_metrics(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray, folds: list[int], cfg: dict) -> dict:
    hard = np.zeros(len(y), dtype=bool)
    used = np.zeros(len(y), dtype=bool)
    thresholds = []
    fold_f1 = []
    for fold in folds:
        valid = fold_ids == fold
        tune = np.isin(fold_ids, [f for f in folds if f != fold])
        t, _ = select_threshold(y[tune], p[tune], cfg)
        thresholds.append(float(t))
        hard[valid] = p[valid] >= t
        used[valid] = True
        fold_f1.append(float(f1_score(y[valid], hard[valid], zero_division=0)))
    return {
        "nested_f1": float(f1_score(y[used], hard[used], zero_division=0)),
        "thresholds": thresholds,
        "median_threshold": float(statistics.median(thresholds)),
        "fold_f1": fold_f1,
        "fold_std": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
    }


def load_data():
    raw = ROOT / "data" / "raw"
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")
    labels = pd.read_csv(raw / "train_labels.csv")
    sample = pd.read_csv(raw / "submission.csv")
    manifest = pd.read_csv(PARENT / "split_manifest.csv", dtype={"group_hash": str})
    if len(train) != len(labels) or not np.array_equal(manifest.row_id.to_numpy(), np.arange(len(train))):
        raise ValueError("Row identity mismatch")
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test schema mismatch")
    y = labels[TARGET].to_numpy(int)
    dev = manifest.loc[manifest.partition.eq("development"), "row_id"].to_numpy(int)
    audit = manifest.loc[manifest.partition.eq("audit"), "row_id"].to_numpy(int)
    folds = manifest.loc[dev, "dev_fold"].to_numpy(int)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Expected frozen dev folds 0..4")
    return train, test, y, sample, manifest, dev, audit, folds


def historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the useful row-local feature ideas from the 2023 notebook.

    The old notebook concatenated train/test before this function, but all
    operations below are row-local. We intentionally avoid carrying over its
    weaker validation protocol while preserving the representation hypothesis.
    """
    df = frame.copy()
    effective = {
        "Refused": 0, "Not At All Effective": 1, "Not Very Effective": 2,
        "Dont Know": 3, "Somewhat Effective": 4, "Very Effective": 5,
    }
    risk = {
        "Refused": 0, "Very Low": 1, "Somewhat Low": 2,
        "Dont Know": 3, "Somewhat High": 4, "Very High": 5,
    }
    sick = {
        "Refused": 0, "Not At All Worried": 1, "Not Very Worried": 2,
        "Dont Know": 3, "Somewhat Worried": 4, "Very Worried": 5,
    }
    age = {
        "6 Months - 9 Years": 1, "10 - 17 Years": 2, "18 - 34 Years": 3,
        "35 - 44 Years": 4, "45 - 54 Years": 5, "55 - 64 Years": 6,
        "65+ Years": 7,
    }
    msa = {"Non-MSA": 1, "MSA, Not Principle City": 2, "MSA, Principle City": 3}
    df["opinion_h1n1_vacc_effective"] = pd.to_numeric(df["opinion_h1n1_vacc_effective"].map(effective), errors="coerce")
    df["opinion_h1n1_risk"] = pd.to_numeric(df["opinion_h1n1_risk"].map(risk), errors="coerce")
    df["opinion_h1n1_sick_from_vacc"] = pd.to_numeric(df["opinion_h1n1_sick_from_vacc"].map(sick), errors="coerce")
    df["agegrp"] = pd.to_numeric(df["agegrp"].map(age), errors="coerce")
    df["census_msa"] = pd.to_numeric(df["census_msa"].map(msa), errors="coerce")
    df["rent_own_r"] = pd.to_numeric(df["rent_own_r"], errors="coerce").replace({77.0: 1.0, 99.0: 1.0})
    df["employment_status"] = (df["employment_status"] == "Employed").astype(float)
    other = [c for c in df.columns if any(k in c for k in ["doctor", "chronic", "child", "health"])]
    # Match old pandas behavior: row sum skips NaN and all-NaN rows become 0.
    df["historical_other"] = df[other].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    # In the 2023 pandas version, the mean dropped nuisance/string seasonal
    # opinion columns, so this effectively aggregated the three numeric H1N1
    # opinion columns after their explicit ordinal conversion.
    h1_op = ["opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc"]
    df["historical_opinion"] = df[h1_op].mean(axis=1)
    behavior = [c for c in df.columns if "behavioral" in c]
    df["historical_behaviorals"] = df[behavior].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    if "census_region" in df.columns:
        df = df.drop(columns=["census_region"])
    return df


def feature_cache(train: pd.DataFrame, test: pd.DataFrame, modes: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    cache = {}
    for mode in modes:
        if mode == "historical":
            cache[mode] = (historical_features(train), historical_features(test))
        elif mode == "legacy_ordinal":
            # The saved 2023 notebook's model cells use train[features] rather
            # than the engineered `raw` dataframe. This mode captures that
            # modeling path: original columns + fold-local OrdinalEncoder.
            cache[mode] = (train.copy(), test.copy())
        else:
            cache[mode] = (build_features(train.copy(), mode), build_features(test.copy(), mode))
    return cache


def fit_transform_representation(frame: pd.DataFrame, mode: str):
    if mode in {"historical", "legacy_ordinal"}:
        encoder = CEOrdinalEncoder(handle_unknown="value", handle_missing="value", return_df=True)
        transformed = encoder.fit_transform(frame)
        # XGBoost receives a pure numeric matrix just as in the historical notebook.
        transformed = transformed.apply(pd.to_numeric, errors="coerce").astype(float)
        return {"kind": "ordinal", "encoder": encoder}, transformed
    schema = fit_xgb_schema(frame)
    return {"kind": "native", "schema": schema}, transform_xgb(frame, schema)


def transform_representation(frame: pd.DataFrame, transformer: dict) -> pd.DataFrame:
    if transformer["kind"] == "ordinal":
        return transformer["encoder"].transform(frame).apply(pd.to_numeric, errors="coerce").astype(float)
    return transform_xgb(frame, transformer["schema"])


def model_from_params(params: dict, cfg: dict, seed: int, n_estimators: int, early_stopping: bool, mode: str) -> XGBClassifier:
    kwargs = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": cfg["device"],
        "enable_categorical": mode not in {"historical", "legacy_ordinal"},
        "n_estimators": int(n_estimators),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "reg_lambda": float(params["reg_lambda"]),
        "reg_alpha": float(params["reg_alpha"]),
        "gamma": float(params["gamma"]),
        "scale_pos_weight": float(params["scale_pos_weight"]),
        "max_delta_step": int(params["max_delta_step"]),
        "max_bin": int(params["max_bin"]),
        "max_cat_to_onehot": int(params["max_cat_to_onehot"]),
        "max_cat_threshold": int(params["max_cat_threshold"]),
        "grow_policy": params["grow_policy"],
        "eval_metric": "logloss",
        "random_state": int(seed),
        "n_jobs": int(cfg["threads"]),
        "verbosity": 0,
    }
    if params["grow_policy"] == "lossguide":
        kwargs["max_leaves"] = int(params["max_leaves"])
    if early_stopping:
        kwargs["early_stopping_rounds"] = int(cfg["early_stopping_rounds"])
    return XGBClassifier(**kwargs)


def sample_params(trial, cfg: dict) -> dict:
    grow = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    params = {
        "feature_mode": trial.suggest_categorical("feature_mode", cfg["feature_modes"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.3, 80.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.45, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.20, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.75, 4.50, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 4),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        "max_cat_to_onehot": trial.suggest_categorical("max_cat_to_onehot", [2, 4, 8, 16, 32]),
        "max_cat_threshold": trial.suggest_categorical("max_cat_threshold", [16, 32, 64, 128]),
        "grow_policy": grow,
        "max_leaves": trial.suggest_int("max_leaves", 15, 255, log=True) if grow == "lossguide" else 0,
    }
    return params


def params_from_trial(trial) -> dict:
    p = dict(trial.params)
    if p.get("grow_policy") != "lossguide":
        p["max_leaves"] = 0
    return p


def fit_fold(params: dict, cfg: dict, features: pd.DataFrame, y: np.ndarray, train_ids: np.ndarray, valid_ids: np.ndarray, seed: int, n_estimators: int, early_stopping: bool, mode: str):
    transformer, xtr = fit_transform_representation(features.iloc[train_ids], mode)
    xva = transform_representation(features.iloc[valid_ids], transformer)
    model = model_from_params(params, cfg, seed, n_estimators, early_stopping, mode)
    fit_kwargs = {"verbose": False}
    if early_stopping:
        fit_kwargs["eval_set"] = [(xva, y[valid_ids])]
    model.fit(xtr, y[train_ids], **fit_kwargs)
    p = model.predict_proba(xva)[:, 1]
    best_iter = int(getattr(model, "best_iteration", n_estimators - 1)) + 1 if early_stopping else int(n_estimators)
    return model, transformer, p, best_iter


def screen_objective_factory(cfg: dict, cache, y, dev, fold_ids):
    screen_folds = [int(x) for x in cfg["screen_folds"]]
    dev_y = y[dev]

    def objective(trial):
        params = sample_params(trial, cfg)
        mode = params.pop("feature_mode")
        features = cache[mode][0]
        oof = np.full(len(dev), np.nan, dtype=float)
        best_iters = []
        fold_times = []
        for fold in screen_folds:
            tr_ids = dev[fold_ids != fold]
            va_pos = np.flatnonzero(fold_ids == fold)
            va_ids = dev[va_pos]
            started = time.perf_counter()
            _, _, p, best_iter = fit_fold(
                params, cfg, features, y, tr_ids, va_ids,
                int(cfg["seed"]) + trial.number * 17 + fold,
                int(cfg["n_estimators_search"]), True, mode,
            )
            fold_times.append(time.perf_counter() - started)
            oof[va_pos] = p
            best_iters.append(best_iter)
        metric = nested_threshold_metrics(dev_y, oof, fold_ids, screen_folds, cfg)
        used = np.isin(fold_ids, screen_folds)
        auc = float(roc_auc_score(dev_y[used], oof[used]))
        tuned_t, tuned_f1 = select_threshold(dev_y[used], oof[used], cfg)
        trial.set_user_attr("feature_mode", mode)
        trial.set_user_attr("nested_f1", metric["nested_f1"])
        trial.set_user_attr("fold_std", metric["fold_std"])
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("tuned_f1", tuned_f1)
        trial.set_user_attr("tuned_threshold", tuned_t)
        trial.set_user_attr("median_best_iteration", int(statistics.median(best_iters)))
        trial.set_user_attr("best_iterations", best_iters)
        trial.set_user_attr("seconds", float(sum(fold_times)))
        # Small stability penalty prevents the search from chasing one lucky fold.
        return float(metric["nested_f1"] - 0.10 * metric["fold_std"])

    return objective


def ensure_study(cfg: dict):
    import optuna
    OUT.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{STUDY_DB.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]), n_startup_trials=30)
    study = optuna.create_study(
        study_name=cfg["study_name"], direction="maximize", storage=storage,
        sampler=sampler, load_if_exists=True,
    )
    if len(study.trials) == 0:
        # Known strong/manual anchors so TPE sees sensible regions immediately.
        anchors = [
            {"feature_mode":"survey","learning_rate":0.03,"max_depth":4,"min_child_weight":12.0,"subsample":0.90,"colsample_bytree":0.85,"colsample_bylevel":1.0,"reg_lambda":12.0,"reg_alpha":0.15,"gamma":0.0,"scale_pos_weight":1.0,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"raw","learning_rate":0.03,"max_depth":4,"min_child_weight":12.0,"subsample":0.90,"colsample_bytree":0.85,"colsample_bylevel":1.0,"reg_lambda":12.0,"reg_alpha":0.15,"gamma":0.0,"scale_pos_weight":1.0,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"survey","learning_rate":0.0929022135,"max_depth":7,"min_child_weight":0.8042,"subsample":0.90,"colsample_bytree":0.9090,"colsample_bylevel":1.0,"reg_lambda":10.0,"reg_alpha":0.0 + 1e-5,"gamma":0.0,"scale_pos_weight":1.0,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"legacy_ordinal","learning_rate":0.03,"max_depth":6,"min_child_weight":1.0,"subsample":1.0,"colsample_bytree":0.20,"colsample_bylevel":0.90,"reg_lambda":1.0,"reg_alpha":1e-5,"gamma":1.0,"scale_pos_weight":2.5714285714,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"legacy_ordinal","learning_rate":0.03,"max_depth":6,"min_child_weight":1.0,"subsample":0.50,"colsample_bytree":0.50,"colsample_bylevel":0.50,"reg_lambda":1.0,"reg_alpha":1e-5,"gamma":1.0,"scale_pos_weight":2.5714285714,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"legacy_ordinal","learning_rate":0.10,"max_depth":9,"min_child_weight":1.0,"subsample":1.0,"colsample_bytree":0.70,"colsample_bylevel":0.50,"reg_lambda":1.0,"reg_alpha":1e-5,"gamma":0.0,"scale_pos_weight":4.0,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
            {"feature_mode":"historical","learning_rate":0.03,"max_depth":6,"min_child_weight":1.0,"subsample":0.50,"colsample_bytree":0.50,"colsample_bylevel":0.50,"reg_lambda":1.0,"reg_alpha":1e-5,"gamma":1.0,"scale_pos_weight":2.5714285714,"max_delta_step":0,"max_bin":256,"max_cat_to_onehot":8,"max_cat_threshold":64,"grow_policy":"depthwise"},
        ]
        for a in anchors:
            study.enqueue_trial(a)
    return study


def completed_trials(study):
    import optuna
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    return sorted(trials, key=lambda t: float(t.value), reverse=True)


def confirm_trial(trial, cfg: dict, cache, train, test, y, dev, audit, fold_ids) -> dict:
    path = OUT / "confirm" / f"trial_{trial.number}.json"
    if path.is_file():
        return read_json(path)
    params_all = params_from_trial(trial)
    mode = params_all.pop("feature_mode")
    features, test_features = cache[mode]
    dev_y = y[dev]
    oof = np.full(len(dev), np.nan, dtype=float)
    audit_parts, test_parts, best_iters, fold_seconds = [], [], [], []
    for fold in range(5):
        tr_ids = dev[fold_ids != fold]
        va_pos = np.flatnonzero(fold_ids == fold)
        va_ids = dev[va_pos]
        started = time.perf_counter()
        model, transformer, p, best_iter = fit_fold(
            params_all, cfg, features, y, tr_ids, va_ids,
            int(cfg["seed"]) + trial.number * 31 + fold,
            int(cfg["n_estimators_search"]), True, mode,
        )
        oof[va_pos] = p
        audit_parts.append(model.predict_proba(transform_representation(features.iloc[audit], transformer))[:, 1])
        test_parts.append(model.predict_proba(transform_representation(test_features, transformer))[:, 1])
        best_iters.append(best_iter)
        fold_seconds.append(time.perf_counter() - started)
    nested = nested_threshold_metrics(dev_y, oof, fold_ids, list(range(5)), cfg)
    tuned_t, tuned_f1 = select_threshold(dev_y, oof, cfg)
    audit_p = np.mean(audit_parts, axis=0)
    test_p = np.mean(test_parts, axis=0)
    result = {
        "trial": int(trial.number), "optuna_value": float(trial.value), "feature_mode": mode,
        "params": params_all, "nested": nested, "tuned_threshold": tuned_t, "tuned_f1": tuned_f1,
        "auc": float(roc_auc_score(dev_y, oof)), "logloss": float(log_loss(dev_y, oof, labels=[0,1])),
        "best_iterations": best_iters, "median_best_iteration": int(statistics.median(best_iters)),
        "fold_seconds": fold_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    np.savez_compressed(OUT / "confirm" / f"trial_{trial.number}_predictions.npz", oof=oof, audit=audit_p, test=test_p)
    return result


def fixed_round_cv(label: str, params: dict, mode: str, rounds: int, cfg: dict, cache, y, dev, audit, fold_ids) -> dict:
    path = OUT / "round_selection" / f"{label}.json"
    pred_path = OUT / "round_selection" / f"{label}_predictions.npz"
    if path.is_file() and pred_path.is_file():
        return read_json(path)
    features = cache[mode][0]
    dev_y = y[dev]
    oof = np.full(len(dev), np.nan, dtype=float)
    audit_parts, fold_seconds = [], []
    for fold in range(5):
        tr_ids = dev[fold_ids != fold]
        va_pos = np.flatnonzero(fold_ids == fold)
        va_ids = dev[va_pos]
        started = time.perf_counter()
        model, transformer, p, _ = fit_fold(params, cfg, features, y, tr_ids, va_ids, int(cfg["seed"]) + 7000 + fold, rounds, False, mode)
        oof[va_pos] = p
        audit_parts.append(model.predict_proba(transform_representation(features.iloc[audit], transformer))[:, 1])
        fold_seconds.append(time.perf_counter() - started)
    nested = nested_threshold_metrics(dev_y, oof, fold_ids, list(range(5)), cfg)
    tuned_t, tuned_f1 = select_threshold(dev_y, oof, cfg)
    result = {
        "label": label, "rounds": int(rounds), "nested": nested,
        "tuned_threshold": tuned_t, "tuned_f1": tuned_f1,
        "auc": float(roc_auc_score(dev_y, oof)), "logloss": float(log_loss(dev_y, oof, labels=[0,1])),
        "fold_seconds": fold_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, result)
    np.savez_compressed(pred_path, oof=oof, audit=np.mean(audit_parts, axis=0))
    return result


def final_refit(selected: dict, cfg: dict, cache, train, test, y, sample) -> dict:
    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    mode = selected["feature_mode"]
    params = selected["params"]
    rounds = int(selected["rounds"])
    threshold = float(selected["threshold"])
    features, test_features = cache[mode]
    transformer, xall = fit_transform_representation(features, mode)
    xtest = transform_representation(test_features, transformer)
    seed_preds = []
    model_rows = []
    for seed in cfg["full_seeds"]:
        model_path = final_dir / f"xgb_full_seed{seed}.joblib"
        prob_path = final_dir / f"test_seed{seed}.npy"
        if model_path.is_file() and prob_path.is_file():
            p = np.load(prob_path)
            seconds = None
        else:
            model = model_from_params(params, cfg, int(seed), rounds, False, mode)
            started = time.perf_counter()
            model.fit(xall, y, verbose=False)
            seconds = time.perf_counter() - started
            p = model.predict_proba(xtest)[:, 1]
            joblib.dump({"model": model, "transformer": transformer, "feature_mode": mode, "params": params, "rounds": rounds}, model_path, compress=3)
            np.save(prob_path, p)
        seed_preds.append(p)
        model_rows.append({"seed": int(seed), "model_path": str(model_path.relative_to(ROOT)).replace("\\","/"), "sha256": sha(model_path), "seconds": seconds})
    arr = np.vstack(seed_preds)
    single = arr[0]
    mean5 = arr.mean(axis=0)
    pd.DataFrame({"Id": sample.Id, "probability": single}).to_csv(final_dir / "test_single.csv", index=False)
    pd.DataFrame({"Id": sample.Id, "probability": mean5}).to_csv(final_dir / "test_seed5.csv", index=False)
    candidates = {}
    for name, p in [("single", single), ("seed5", mean5)]:
        sub = sample.copy()
        sub[TARGET] = (p >= threshold).astype(np.int64)
        sub_path = ROOT / "submissions" / f"v17_xgb_optuna_full_{name}.csv"
        if sub_path.exists():
            existing = pd.read_csv(sub_path)
            check_submission(sample, existing)
        else:
            sub.to_csv(sub_path, index=False)
        checked = check_submission(sample, pd.read_csv(sub_path))
        candidates[name] = {"path": str(sub_path.relative_to(ROOT)).replace("\\","/"), "sha256": sha(sub_path), **checked}
    return {
        "models": model_rows,
        "seed_prediction_mean_std": float(np.mean(np.std(arr, axis=0))),
        "single_vs_seed5_corr": float(np.corrcoef(single, mean5)[0,1]),
        "single_vs_seed5_hard_disagreement": int(np.sum((single >= threshold) != (mean5 >= threshold))),
        "candidates": candidates,
    }


def check(cfg: dict) -> None:
    import optuna
    train, test, y, _, manifest, dev, audit, folds = load_data()
    cache = feature_cache(train, test, cfg["feature_modes"])
    payload = {
        "check_passed": True,
        "python_packages": {"optuna": optuna.__version__, "xgboost": importlib.metadata.version("xgboost")},
        "train_rows": len(train), "dev_rows": len(dev), "audit_rows": len(audit), "test_rows": len(test),
        "feature_shapes": {m: [int(cache[m][0].shape[0]), int(cache[m][0].shape[1])] for m in cache},
        "fold_counts": {str(f): int(np.sum(folds == f)) for f in range(5)},
        "device": cfg["device"], "n_trials": cfg["n_trials"], "screen_folds": cfg["screen_folds"],
        "full_refit_policy": "required; all 42154 labels; threshold frozen from grouped CV",
        "remote_submission_authorized": False,
    }
    write_json(ROOT / "reports" / "v17_xgb_optuna_preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def status(cfg: dict) -> None:
    if not STUDY_DB.is_file():
        print(json.dumps({"status":"not_started"}, indent=2)); return
    study = ensure_study(cfg)
    done = completed_trials(study)
    payload = {
        "completed_trials": len(done), "target_trials": int(cfg["n_trials"]),
        "best_value": float(done[0].value) if done else None,
        "best_trial": int(done[0].number) if done else None,
        "best_attrs": done[0].user_attrs if done else None,
        "confirm_completed": len(list((OUT / "confirm").glob("trial_*.json"))) if (OUT / "confirm").exists() else 0,
        "round_selection_completed": len(list((OUT / "round_selection").glob("*.json"))) if (OUT / "round_selection").exists() else 0,
        "final_results": (OUT / "results.json").is_file(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def run(cfg: dict) -> None:
    import optuna
    if (OUT / "results.json").is_file():
        print("v17 already complete; immutable results preserved", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    config_copy = OUT / "config.json"
    if config_copy.is_file() and read_json(config_copy) != cfg:
        # A changed contract must never be mixed into a real resumable study.
        # However, if no study/results exist yet, the prior config is only a
        # stale preflight artifact and can be archived safely.
        if not STUDY_DB.is_file() and not (OUT / "results.json").is_file():
            stale = OUT / "config.pre_historical.json"
            if not stale.is_file():
                config_copy.replace(stale)
            else:
                config_copy.unlink()
            write_json(config_copy, cfg)
            print(f"Archived stale pre-study config to {stale.name}; continuing with current config", flush=True)
        else:
            raise ValueError("v17 config differs from an existing resumable study; use a new version/study name")
    if not config_copy.is_file():
        write_json(config_copy, cfg)
    train, test, y, sample, manifest, dev, audit, folds = load_data()
    cache = feature_cache(train, test, cfg["feature_modes"])
    study = ensure_study(cfg)
    done_before = len(completed_trials(study))
    remaining = max(0, int(cfg["n_trials"]) - done_before)
    if remaining:
        print(f"OPTUNA resume/start: completed={done_before}, remaining={remaining}", flush=True)
        study.optimize(
            screen_objective_factory(cfg, cache, y, dev, folds),
            n_trials=remaining,
            timeout=int(cfg["study_timeout_seconds"]),
            gc_after_trial=True,
            show_progress_bar=True,
        )
    done = completed_trials(study)
    if not done:
        raise RuntimeError("No completed Optuna trials")
    pd.DataFrame([
        {"trial":t.number,"value":t.value,**t.params,**{f"attr_{k}":v for k,v in t.user_attrs.items() if not isinstance(v,(list,dict))}}
        for t in done
    ]).to_csv(OUT / "optuna_trials.csv", index=False)

    top_trials = done[:min(int(cfg["confirm_top_k"]), len(done))]
    confirms = []
    for rank, trial in enumerate(top_trials, 1):
        print(f"CONFIRM {rank}/{len(top_trials)} trial={trial.number} screen={trial.value:.6f}", flush=True)
        confirms.append(confirm_trial(trial, cfg, cache, train, test, y, dev, audit, folds))
    confirms.sort(key=lambda r: (r["nested"]["nested_f1"], r["tuned_f1"], r["auc"]), reverse=True)
    write_json(OUT / "confirmation_leaderboard.json", confirms)
    winner = confirms[0]

    base_rounds = max(50, int(winner["median_best_iteration"]))
    round_results = []
    for mult in cfg["round_multipliers"]:
        rounds = max(20, int(round(base_rounds * float(mult))))
        label = f"m{str(mult).replace('.','p')}_{rounds}"
        print(f"ROUND CHECK {label}", flush=True)
        round_results.append(fixed_round_cv(label, winner["params"], winner["feature_mode"], rounds, cfg, cache, y, dev, audit, folds))
    round_results.sort(key=lambda r: (r["nested"]["nested_f1"], r["tuned_f1"], r["auc"]), reverse=True)
    write_json(OUT / "round_selection.json", round_results)
    best_round = round_results[0]
    frozen_threshold = float(best_round["nested"]["median_threshold"])

    # Audit is diagnostic only and occurs after recipe/round/threshold selection.
    selected_pred = np.load(OUT / "round_selection" / f"{best_round['label']}_predictions.npz")
    audit_p = selected_pred["audit"]
    audit_metrics = {
        "f1": float(f1_score(y[audit], audit_p >= frozen_threshold, zero_division=0)),
        "auc": float(roc_auc_score(y[audit], audit_p)),
        "logloss": float(log_loss(y[audit], audit_p, labels=[0,1])),
    }
    selected = {
        "trial": winner["trial"], "feature_mode": winner["feature_mode"], "params": winner["params"],
        "rounds": int(best_round["rounds"]), "threshold": frozen_threshold,
        "confirm_nested_f1": winner["nested"]["nested_f1"],
        "fixed_round_nested_f1": best_round["nested"]["nested_f1"],
        "fixed_round_tuned_f1": best_round["tuned_f1"],
        "fixed_round_auc": best_round["auc"],
        "threshold_source": "median of five leave-one-fold CV thresholds after round selection",
    }
    write_json(OUT / "frozen_selection.json", selected)
    final = final_refit(selected, cfg, cache, train, test, y, sample)
    result = {
        "version": cfg["version"], "completed_utc": now(),
        "optuna_completed_trials": len(done), "top_confirmed": len(confirms),
        "selected": selected, "audit_diagnostic": audit_metrics,
        "full_refit": {"training_rows": len(train), "all_available_labels_used": True, **final},
        "limitations": [
            "Optuna search uses frozen development folds and can still overfit the validation scheme after many trials.",
            "The reused audit is diagnostic only and did not influence selection.",
            "Final threshold is frozen from grouped CV and is not retuned on all labels.",
        ],
        "submitted": False,
    }
    write_json(OUT / "results.json", result)
    with (ROOT / "kaggle_ops" / "experiments.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({"version":cfg["version"],"id":"xgb_optuna_fullrefit","parent":"v13_xgb_full","result":result,"artifact":"artifacts/v17_xgb_optuna","submitted":False},ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"completed":True,"selected":selected,"audit":audit_metrics,"candidates":final["candidates"]}, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check","run","status"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v17_xgb_optuna.json")
    args = ap.parse_args()
    cfg = read_json(args.config)
    if args.phase == "check": check(cfg)
    elif args.phase == "status": status(cfg)
    else: run(cfg)


if __name__ == "__main__":
    main()
