"""Leakage-aware binary F1 baseline with a locked audit holdout."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'vacc_h1n1_f'
NOMINAL = {'education_comp', 'raceeth4_i', 'sex_i', 'inc_pov', 'marital',
           'rent_own_r', 'census_region', 'hhs_region'}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Tune on development OOF only; coarse fixed grid limits overfitting."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError('Invalid threshold calibration predictions')
    grid = np.linspace(0.15, 0.60, 91)
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in grid])
    winners = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    idx = winners[len(winners) // 2]
    return float(grid[idx]), float(scores[idx])

def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(np.int8)
    return {'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'positive_prediction_rate': float(pred.mean())}

def prepare_features(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in categorical:
        result[col] = result[col].where(result[col].notna(), '__MISSING__').astype(str)
    for col in result.columns.difference(categorical):
        result[col] = pd.to_numeric(result[col], errors='raise').astype(float)
    return result

def make_model(family: str, categorical: list[str], numeric: list[str], config: dict, seed: int):
    if family == 'logistic':
        numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                                 ('scale', StandardScaler())])
        preprocess = ColumnTransformer([
            ('numeric', numeric_pipe, numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', min_frequency=5), categorical),
        ])
        return Pipeline([('preprocess', preprocess),
                         ('model', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=seed))])
    if family == 'catboost':
        return CatBoostClassifier(**config['catboost'], cat_features=categorical,
                                  random_seed=seed, thread_count=config['threads'],
                                  allow_writing_files=False, verbose=False)
    raise ValueError(f'Unknown family: {family}')

def check_submission(sample: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    if list(candidate.columns) != list(sample.columns) or len(candidate) != len(sample):
        raise ValueError('Submission schema/row count mismatch')
    if not candidate['Id'].equals(sample['Id']) or not candidate['Id'].is_unique:
        raise ValueError('Submission ID/order mismatch')
    pred = candidate[TARGET].to_numpy()
    if candidate.isna().any().any() or not np.isfinite(pred).all() or not set(np.unique(pred)) <= {0, 1}:
        raise ValueError('Submission must contain finite binary labels')
    return {'passed': True, 'rows': len(candidate), 'columns': list(candidate.columns),
            'positive_predictions': int(pred.sum()), 'positive_rate': float(pred.mean())}

def run(config: dict) -> dict:
    started = time.monotonic()
    raw = ROOT / 'data' / 'raw'
    x, test, labels, sample = [pd.read_csv(raw / name) for name in
        ('train.csv', 'test.csv', 'train_labels.csv', 'submission.csv')]
    if list(x.columns) != list(test.columns) or len(x) != len(labels):
        raise ValueError('Unexpected input schema')
    if list(sample.columns) != ['Id', TARGET] or len(sample) != len(test):
        raise ValueError('Unexpected official sample')
    if TARGET in x or 'vacc_seas_f' in x:
        raise ValueError('Training labels may not enter features')
    y = labels[TARGET].to_numpy()
    if labels[TARGET].isna().any() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid target')
    groups = pd.util.hash_pandas_object(x.fillna('__MISSING__').astype(str), index=False).astype(str).to_numpy()
    seed = int(config['seed'])
    audit_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, audit = next(audit_split.split(x, y, groups))
    folds = list(StratifiedGroupKFold(n_splits=config['folds'], shuffle=True, random_state=seed + 1)
                 .split(x.iloc[dev], y[dev], groups[dev]))
    assert not set(groups[dev]) & set(groups[audit])
    categorical = [c for c in x if x[c].dtype == 'object' or c in NOMINAL]
    numeric = [c for c in x if c not in categorical]
    x, test = prepare_features(x, categorical), prepare_features(test, categorical)
    run_id = config['run_id']
    out = ROOT / 'artifacts' / run_id
    if out.exists():
        raise FileExistsError(f'Immutable run exists: {out}; choose a new run_id')
    out.mkdir(parents=True)
    manifest = pd.DataFrame({'row_id': np.arange(len(x)), 'group_hash': groups, 'partition': 'audit', 'dev_fold': -1})
    manifest.loc[dev, 'partition'] = 'development'
    for fold, (tr, va) in enumerate(folds):
        assert not set(groups[dev[tr]]) & set(groups[dev[va]])
        manifest.loc[dev[va], 'dev_fold'] = fold
    manifest.to_csv(out / 'split_manifest.csv', index=False)
    save_json(out / 'config.json', config)
    print(f'SPLIT development={len(dev)}, audit={len(audit)}, features={len(x.columns)}, categorical={len(categorical)}', flush=True)
    results, candidates = {}, {}
    # Do not use audit labels until the model family and threshold are frozen.
    for family in config['families']:
        family_started = time.monotonic()
        oof = np.full(len(dev), np.nan)
        audit_p, test_p = np.zeros(len(audit)), np.zeros(len(test))
        fold_results = []
        for fold, (tr, va) in enumerate(folds):
            model = make_model(family, categorical, numeric, config, seed + fold)
            model.fit(x.iloc[dev[tr]], y[dev[tr]])
            oof[va] = model.predict_proba(x.iloc[dev[va]])[:, 1]
            audit_p += model.predict_proba(x.iloc[audit])[:, 1] / len(folds)
            test_p += model.predict_proba(test)[:, 1] / len(folds)
            score = metrics(y[dev[va]], oof[va], 0.5)
            fold_results.append(score)
            joblib.dump(model, out / f'{family}_fold{fold}.joblib', compress=3)
            print(f'{family} fold={fold + 1}/{len(folds)} F1@0.5={score["f1"]:.6f} AUC={score["roc_auc"]:.6f}', flush=True)
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete/nonfinite OOF predictions')
        threshold, tuned_f1 = select_threshold(y[dev], oof)
        results[family] = {
            'threshold': threshold, 'dev_oof_tuned_f1': tuned_f1,
            'dev_oof_at_05': metrics(y[dev], oof, 0.5),
            'dev_oof_at_selected_threshold': metrics(y[dev], oof, threshold),
            'fold_metrics_at_05': fold_results,
            'fold_f1_at_05_std': float(np.std([v['f1'] for v in fold_results], ddof=1)),
            'training_seconds': time.monotonic() - family_started,
        }
        pd.DataFrame({'row_id': dev, 'target': y[dev], 'probability': oof}).to_csv(out / f'{family}_oof.csv', index=False)
        pd.DataFrame({'row_id': audit, 'probability': audit_p}).to_csv(out / f'{family}_audit_probabilities.csv', index=False)
        candidates[family] = {'audit': audit_p, 'test': test_p}
        print(f'{family} DEV OOF tuned F1={tuned_f1:.6f}, threshold={threshold:.3f}', flush=True)
    champion = max(results, key=lambda name: results[name]['dev_oof_tuned_f1'])
    threshold = results[champion]['threshold']
    frozen = {'family': champion, 'threshold': threshold, 'selection_source': 'development OOF only; no audit feedback'}
    save_json(out / 'frozen_selection.json', frozen)
    for family in config['families']:
        results[family]['audit_at_05'] = metrics(y[audit], candidates[family]['audit'], 0.5)
        results[family]['audit_at_frozen_threshold'] = metrics(y[audit], candidates[family]['audit'], results[family]['threshold'])
        print(f'{family} LOCKED AUDIT F1={results[family]["audit_at_frozen_threshold"]["f1"]:.6f}', flush=True)
    # Audit above describes the development ensemble, not this full-data refit.
    model = make_model(champion, categorical, numeric, config, seed)
    model.fit(x, y)
    full_prob = model.predict_proba(test)[:, 1]
    model_path = out / f'{champion}_full.joblib'
    joblib.dump(model, model_path, compress=3)
    reloaded_prob = joblib.load(model_path).predict_proba(test)[:, 1]
    if not np.allclose(full_prob, reloaded_prob, atol=1e-12, rtol=0):
        raise ValueError('Reloaded inference changed')
    submission = sample.copy()
    submission[TARGET] = (full_prob >= threshold).astype(np.int64)
    validation = check_submission(sample, submission)
    submission_path = ROOT / 'submissions' / f'{run_id}_{champion}.csv'
    submission_path.parent.mkdir(exist_ok=True)
    if submission_path.exists():
        raise FileExistsError(submission_path)
    submission.to_csv(submission_path, index=False)
    check_submission(sample, pd.read_csv(submission_path))
    pd.DataFrame({'Id': sample.Id, 'probability': full_prob}).to_csv(out / 'test_probabilities.csv', index=False)
    pd.DataFrame({'Id': sample.Id, 'probability': candidates[champion]['test']}).to_csv(out / 'dev_ensemble_test_probabilities.csv', index=False)
    if champion == 'catboost':
        pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).to_csv(out / 'feature_importance.csv', index=False)
    summary = {
        'run_id': run_id, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'metric': 'binary F1, positive class 1', 'development_rows': len(dev), 'audit_rows': len(audit),
        'validation': '80/20 StratifiedGroupKFold holdout; 5-fold grouped development OOF; identical feature hashes stay together',
        'selection': frozen, 'results': results,
        'constant_baseline': {'all_zero_f1': 0.0, 'all_one_audit_f1': float(f1_score(y[audit], np.ones(len(audit))))},
        'submission': {'path': str(submission_path.relative_to(ROOT)), 'sha256': digest(submission_path), **validation},
        'final_model': str(model_path.relative_to(ROOT)),
        'limitations': ['Development tuned F1 is a threshold-selection score, not an unbiased estimate.',
                       'Audit scores describe the development CV ensemble, not the full-data refit.',
                       'No respondent/household identifier is provided; feature-hash grouping is a conservative proxy.',
                       'Group split assumes approximately IID respondents; temporal/geographic shift is not ruled out.'],
        'code_sha256': digest(Path(__file__)), 'split_sha256': digest(out / 'split_manifest.csv'),
        'raw_sha256': {p.name: digest(p) for p in sorted(raw.glob('*.csv'))},
        'package_versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib')},
        'elapsed_seconds': time.monotonic() - started,
    }
    save_json(out / 'metrics.json', summary)
    ledger = ROOT / 'kaggle_ops' / 'experiments.jsonl'
    ledger.parent.mkdir(exist_ok=True)
    with ledger.open('a', encoding='utf-8') as f:
        for family, result in results.items():
            record = {'run_id': run_id, 'family': family, 'parent': None,
                      'hypothesis': 'Nonlinear categorical interactions improve F1 over a regularized linear baseline' if family == 'catboost' else 'One-hot logistic regression validates preprocessing and threshold selection',
                      'validation': summary['validation'], 'result': result,
                      'decision': 'champion' if family == champion else 'baseline',
                      'artifact': str(out.relative_to(ROOT))}
            f.write(json.dumps(record, allow_nan=False) + '\n')
    champion_path = ROOT / 'CHAMPION.json'
    if champion_path.exists():
        raise FileExistsError('Existing champion protected; promote manually after comparison')
    save_json(champion_path, {'run_id': run_id, **frozen,
                             'metrics_path': str((out / 'metrics.json').relative_to(ROOT)),
                             'submission_path': str(submission_path.relative_to(ROOT)),
                             'submission_sha256': digest(submission_path),
                             'model_path': str(model_path.relative_to(ROOT)),
                             'columns': list(x.columns), 'categorical_columns': categorical})
    print(json.dumps({'champion': champion, 'threshold': threshold,
                      'audit_f1': results[champion]['audit_at_frozen_threshold']['f1'],
                      'submission': summary['submission'], 'elapsed_seconds': summary['elapsed_seconds']}, indent=2), flush=True)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'baseline.json')
    args = parser.parse_args()
    run(json.loads(args.config.read_text(encoding='utf-8')))

if __name__ == '__main__':
    main()
