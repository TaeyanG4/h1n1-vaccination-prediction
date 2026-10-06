"""v2 controlled experiments. Preserves v1; never calls a remote submission API."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from baseline import check_submission, metrics, select_threshold, prepare_features, make_model
from v2_features import build_features, fit_schema, transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'v2'
PARENT = ROOT / 'artifacts' / 'baseline_v1'
TARGET = 'vacc_h1n1_f'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def initialize(config: dict) -> None:
    if OUT.exists():
        raise FileExistsError('v2 already exists. Do not overwrite; use the documented resume phase.')
    # Record all original v1 code, models, predictions and receipt bytes.
    protected = list(PARENT.rglob('*')) + list((ROOT / 'data' / 'raw').glob('*.csv'))
    protected += [ROOT / p for p in ['src/baseline.py', 'configs/baseline.json', 'tests/test_baseline.py', 'scripts/audit_data.py', 'scripts/reproduce_submission.py', 'CHAMPION.json', 'submissions/baseline_v1_catboost.csv', 'submissions/baseline_v1_reproduced.csv', 'reports/submission_receipt.json']]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in protected if p.is_file()}
    expected = read_json(PARENT / 'metrics.json')
    if sha(ROOT / 'src' / 'baseline.py') != expected['code_sha256']:
        raise ValueError('v1 source hash mismatch')
    if sha(PARENT / 'split_manifest.csv') != expected['split_sha256']:
        raise ValueError('v1 split hash mismatch')
    for name, digest in expected['raw_sha256'].items():
        if sha(ROOT / 'data' / 'raw' / name) != digest:
            raise ValueError(f'Raw data changed: {name}')
    if sha(ROOT / 'submissions' / 'baseline_v1_catboost.csv') != expected['submission']['sha256']:
        raise ValueError('v1 submitted file changed')
    snapshot = ROOT / 'versions' / 'v1'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ['agents.md', 'discoveries.md', 'handoff.md', 'plan.md', 'README.md', 'CHAMPION.json']:
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copy2(ROOT / 'src' / 'baseline.py', snapshot / 'baseline.py')
    shutil.copy2(ROOT / 'configs' / 'baseline.json', snapshot / 'baseline.json')
    write_json(snapshot / 'manifest.json', {'version': 'v1', 'run_id': 'baseline_v1', 'created_utc': now(), 'sha256': hashes})
    OUT.mkdir(parents=True)
    write_json(OUT / 'config.json', config)
    shutil.copy2(PARENT / 'split_manifest.csv', OUT / 'split_manifest.csv')
    source_dir = OUT / 'source'
    source_dir.mkdir()
    for name in ['v2_features.py', 'v2_pipeline.py', 'baseline.py']:
        shutil.copy2(ROOT / 'src' / name, source_dir / name)
    write_json(OUT / 'provenance.json', {'created_utc': now(), 'code_sha256': {name: sha(source_dir / name) for name in ['v2_features.py', 'v2_pipeline.py', 'baseline.py']}, 'config_sha256': sha(ROOT / 'configs' / 'v2.json'), 'split_sha256': sha(OUT / 'split_manifest.csv'), 'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'joblib']}, 'submitted': False})

def verify_parent() -> dict:
    protected = read_json(ROOT / 'versions' / 'v1' / 'manifest.json')['sha256']
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    if changed:
        raise ValueError(f'Protected v1 files changed: {changed}')
    return {'passed': True, 'protected_files': len(protected)}

def load_data():
    raw = ROOT / 'data' / 'raw'
    x, test, labels, sample = [pd.read_csv(raw / name) for name in ['train.csv', 'test.csv', 'train_labels.csv', 'submission.csv']]
    manifest = pd.read_csv(PARENT / 'split_manifest.csv', dtype={'group_hash': str})
    if len(x) != len(labels) or not np.array_equal(manifest.row_id, np.arange(len(x))):
        raise ValueError('Row identity mismatch')
    if list(x.columns) != list(test.columns):
        raise ValueError('Train/test schemas differ')
    dev = manifest.loc[manifest.partition == 'development', 'row_id'].to_numpy()
    audit = manifest.loc[manifest.partition == 'audit', 'row_id'].to_numpy()
    groups = manifest.group_hash.to_numpy()
    if set(groups[dev]) & set(groups[audit]):
        raise ValueError('Duplicate groups cross audit boundary')
    return x, test, labels[TARGET].to_numpy(), sample, manifest, dev, audit

def model_for(spec: dict, config: dict, schema: dict, seed: int):
    if spec['family'] == 'catboost':
        return CatBoostClassifier(**config['catboost'], random_seed=seed, thread_count=config['threads'], cat_features=schema['categorical'], allow_writing_files=False, verbose=False)
    return LGBMClassifier(**config['lightgbm'], random_state=seed, n_jobs=config['threads'], objective='binary')

def train_fold(spec: dict, config: dict, fold: int, x: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, manifest: pd.DataFrame, dev: np.ndarray, audit: np.ndarray) -> None:
    directory = OUT / 'candidates' / spec['id']
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / f'fold{fold}_metrics.json'
    if receipt.exists():
        saved = read_json(receipt)
        for name, digest in saved['artifact_sha256'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Fold artifact changed: {name}')
        print(f'REUSE {spec["id"]} fold={fold}', flush=True)
        return
    if (directory / f'fold{fold}.joblib').exists():
        raise FileExistsError('Partial fold exists without its receipt; inspect it before retrying')
    fold_ids = manifest.loc[dev, 'dev_fold'].to_numpy()
    train_ids, valid_ids = dev[fold_ids != fold], dev[fold_ids == fold]
    if set(manifest.loc[train_ids, 'group_hash']) & set(manifest.loc[valid_ids, 'group_hash']):
        raise ValueError('Duplicate groups cross development fold')
    started = time.monotonic()
    features, test_features = build_features(x, spec['features']), build_features(test, spec['features'])
    schema = fit_schema(features.iloc[train_ids], spec['family'])
    model = model_for(spec, config, schema, config['seed'] + fold)
    model.fit(transform(features.iloc[train_ids], schema), y[train_ids])
    p = model.predict_proba(transform(features.iloc[valid_ids], schema))[:, 1]
    audit_p = model.predict_proba(transform(features.iloc[audit], schema))[:, 1]
    test_p = model.predict_proba(transform(test_features, schema))[:, 1]
    if not all(np.isfinite(a).all() for a in [p, audit_p, test_p]):
        raise ValueError('Nonfinite model probabilities')
    joblib.dump({'model': model, 'schema': schema, 'spec': spec}, directory / f'fold{fold}.joblib', compress=3)
    pd.DataFrame({'row_id': valid_ids, 'target': y[valid_ids], 'probability': p}).to_csv(directory / f'fold{fold}_oof.csv', index=False)
    np.savez_compressed(directory / f'fold{fold}_predictions.npz', audit=audit_p, test=test_p)
    result = metrics(y[valid_ids], p, 0.31)
    result.update({'fold': fold, 'candidate': spec['id'], 'features': len(features.columns), 'seconds': time.monotonic() - started, 'audit_labels_used': False})
    names = [f'fold{fold}.joblib', f'fold{fold}_oof.csv', f'fold{fold}_predictions.npz']
    result['artifact_sha256'] = {name: sha(directory / name) for name in names}
    write_json(receipt, result)
    print(f'{spec["id"]} fold={fold + 1}/5 F1@0.31={result["f1"]:.6f} AUC={result["roc_auc"]:.6f} seconds={result["seconds"]:.1f}', flush=True)

def summarize(y: np.ndarray, p: np.ndarray, fold_ids: np.ndarray) -> dict:
    t, f1 = select_threshold(y, p)
    fold_f1 = [float(f1_score(y[fold_ids == f], p[fold_ids == f] >= t)) for f in sorted(set(fold_ids))]
    nearby = {f'{z:.3f}': float(f1_score(y, p >= z)) for z in [t - .01, t - .005, t, t + .005, t + .01]}
    return {'threshold': t, 'dev_oof_tuned_f1': f1, 'dev_metrics': metrics(y, p, t), 'fold_f1': fold_f1, 'fold_f1_std': float(np.std(fold_f1, ddof=1)), 'threshold_neighborhood_f1': nearby}

def assemble(spec: dict, y: np.ndarray, manifest: pd.DataFrame, dev: np.ndarray) -> dict:
    directory = OUT / 'candidates' / spec['id']
    parts = [pd.read_csv(directory / f'fold{f}_oof.csv') for f in range(5)]
    oof = pd.concat(parts).set_index('row_id').loc[dev].reset_index()
    if not np.array_equal(oof.target, y[dev]) or oof.probability.isna().any():
        raise ValueError('OOF alignment failure')
    audit_p, test_p = [], []
    for f in range(5):
        with np.load(directory / f'fold{f}_predictions.npz') as data:
            audit_p.append(data['audit'])
            test_p.append(data['test'])
    oof.to_csv(directory / 'oof.csv', index=False)
    ap, tp = np.mean(audit_p, axis=0), np.mean(test_p, axis=0)
    np.savez_compressed(directory / 'ensemble_predictions.npz', audit=ap, test=tp)
    result = summarize(y[dev], oof.probability.to_numpy(), manifest.loc[dev, 'dev_fold'].to_numpy())
    result['training_seconds'] = sum(read_json(directory / f'fold{f}_metrics.json')['seconds'] for f in range(5))
    result['spec'] = spec
    write_json(directory / 'development_metrics.json', result)
    return {'oof': oof.probability.to_numpy(), 'audit': ap, 'test': tp, 'result': result, 'components': {spec['id']: 1.0}}

def bootstrap_delta(y: np.ndarray, parent_pred: np.ndarray, candidate_pred: np.ndarray, groups: np.ndarray, repeats: int, seed: int) -> dict:
    _, inverse = np.unique(groups, return_inverse=True)
    n = int(inverse.max()) + 1
    def counts(pred):
        return np.column_stack([np.bincount(inverse, weights=w, minlength=n) for w in [(y == 1) & pred, (y == 0) & pred, (y == 1) & ~pred]])
    base, cand = counts(parent_pred), counts(candidate_pred)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(repeats):
        ix = rng.integers(0, n, size=n)
        b, c = base[ix].sum(axis=0), cand[ix].sum(axis=0)
        bf = 2 * b[0] / max(1.0, 2 * b[0] + b[1] + b[2])
        cf = 2 * c[0] / max(1.0, 2 * c[0] + c[1] + c[2])
        deltas.append(cf - bf)
    return {'repeats': repeats, 'groups': n, 'delta_95pct_interval': np.quantile(deltas, [.025, .975]).tolist(), 'positive_fraction': float(np.mean(np.array(deltas) > 0)), 'limitation': 'Conditional on fixed trained models, selected candidates and thresholds; does not include model/selection uncertainty.'}

def develop(config: dict, resume: bool) -> None:
    if not resume:
        initialize(config)
    else:
        if read_json(OUT / 'config.json') != config:
            raise ValueError('Resume config mismatch')
        original = read_json(OUT / 'provenance.json')['code_sha256']
        if any(sha(ROOT / 'src' / n) != h for n, h in original.items()):
            raise ValueError('Source changed during experiment; do not silently resume')
    verify_parent()
    x, test, y, sample, manifest, dev, audit = load_data()
    fold_ids = manifest.loc[dev, 'dev_fold'].to_numpy()
    candidates = {}
    for family in ['catboost', 'logistic']:
        oof = pd.read_csv(PARENT / f'{family}_oof.csv').set_index('row_id').loc[dev]
        ap = pd.read_csv(PARENT / f'{family}_audit_probabilities.csv').set_index('row_id').loc[audit, 'probability'].to_numpy()
        name = f'{family}_v1'
        p = oof.probability.to_numpy()
        candidates[name] = {'oof': p, 'audit': ap, 'test': None, 'result': summarize(y[dev], p, fold_ids), 'components': {name: 1.0}}
    parent = candidates['catboost_v1']
    print(f'PARENT v1 OOF F1={parent["result"]["dev_oof_tuned_f1"]:.6f}; identical splits; v1 untouched', flush=True)
    screen_mask = np.isin(fold_ids, config['screen_folds'])
    ref_t, ref_f1 = select_threshold(y[dev][screen_mask], parent['oof'][screen_mask])
    ref_auc = roc_auc_score(y[dev][screen_mask], parent['oof'][screen_mask])
    screen_results = {}
    for spec in config['candidates']:
        for f in config['screen_folds']:
            train_fold(spec, config, f, x, test, y, manifest, dev, audit)
        part = pd.concat([pd.read_csv(OUT / 'candidates' / spec['id'] / f'fold{f}_oof.csv') for f in config['screen_folds']]).set_index('row_id').loc[dev[screen_mask]]
        t, score = select_threshold(part.target.to_numpy(), part.probability.to_numpy())
        auc = float(roc_auc_score(part.target, part.probability))
        passed = score >= ref_f1 - config['screen_max_f1_drop'] and auc >= ref_auc - config['screen_max_auc_drop']
        screen_results[spec['id']] = {'f1': score, 'threshold': t, 'auc': auc, 'f1_delta_vs_parent_same_rows': score - ref_f1, 'passed': bool(passed)}
        print(f'SCREEN {spec["id"]} F1={score:.6f} delta={score-ref_f1:+.6f} continue={passed}', flush=True)
        write_json(OUT / 'screening.json', {'reference_f1': ref_f1, 'reference_auc': ref_auc, 'folds': config['screen_folds'], 'candidates': screen_results, 'limitation': 'Two-fold screening is noisy and is not promotion evidence.'})
    for spec in config['candidates']:
        if not screen_results[spec['id']]['passed']:
            continue
        for f in range(5):
            if f not in config['screen_folds']:
                train_fold(spec, config, f, x, test, y, manifest, dev, audit)
        candidates[spec['id']] = assemble(spec, y, manifest, dev)
        print(f'DEV COMPLETE {spec["id"]} F1={candidates[spec["id"]]["result"]["dev_oof_tuned_f1"]:.6f}', flush=True)
    blends = [('blend_cat_logistic', 'logistic_v1')]
    eligible_lgb = [s['id'] for s in config['candidates'] if s['family'] == 'lightgbm' and s['id'] in candidates]
    if eligible_lgb:
        best_lgb = max(eligible_lgb, key=lambda n: candidates[n]['result']['dev_oof_tuned_f1'])
        blends.append(('blend_cat_lgb', best_lgb))
    if 'cat_survey' in candidates:
        blends.append(('blend_cat_survey', 'cat_survey'))
    for name, other in blends:
        p = .5 * (parent['oof'] + candidates[other]['oof'])
        candidates[name] = {'oof': p, 'audit': .5 * (parent['audit'] + candidates[other]['audit']), 'test': None, 'result': summarize(y[dev], p, fold_ids), 'components': {'catboost_v1': .5, other: .5}}
    chosen = max(candidates, key=lambda n: candidates[n]['result']['dev_oof_tuned_f1'])
    rows = []
    for name, obj in candidates.items():
        r = obj['result']
        deltas = np.asarray(r['fold_f1']) - np.asarray(parent['result']['fold_f1'])
        r.update({'candidate': name, 'dev_delta_vs_v1': r['dev_oof_tuned_f1'] - parent['result']['dev_oof_tuned_f1'], 'paired_fold_deltas': deltas.tolist(), 'positive_folds': int((deltas > 0).sum()), 'oof_correlation_v1': float(np.corrcoef(parent['oof'], obj['oof'])[0, 1]), 'components': obj['components']})
        rows.append({k: r[k] for k in ['candidate', 'dev_oof_tuned_f1', 'threshold', 'dev_delta_vs_v1', 'positive_folds', 'oof_correlation_v1']})
    selection = {'candidate': chosen, 'components': candidates[chosen]['components'], 'threshold': candidates[chosen]['result']['threshold'], 'frozen_utc': now(), 'selection_source': 'Development OOF only. Reused audit not scored in v2 before this file was written.', 'audit_reuse_warning': 'This holdout was already inspected in v1; it is diagnostic, not a pristine confirmation set.'}
    frozen_path = OUT / 'frozen_selection.json'
    if frozen_path.exists() and read_json(frozen_path)['candidate'] != chosen:
        raise ValueError('Frozen selection changed')
    write_json(frozen_path, selection)
    # Only the chosen recipe and reference are now scored on the reused holdout.
    winner = candidates[chosen]
    ap, bp = winner['audit'], parent['audit']
    audit_score = metrics(y[audit], ap, selection['threshold'])
    audit_base = metrics(y[audit], bp, parent['result']['threshold'])
    dev_gain = winner['result']['dev_delta_vs_v1']
    audit_gain = audit_score['f1'] - audit_base['f1']
    bootstrap = bootstrap_delta(y[audit], bp >= parent['result']['threshold'], ap >= selection['threshold'], manifest.loc[audit, 'group_hash'].to_numpy(), config['bootstrap_repeats'], 20261005)
    ready = chosen != 'catboost_v1' and dev_gain >= config['minimum_dev_gain'] and audit_gain >= -config['maximum_audit_drop'] and winner['result']['positive_folds'] >= 3
    pd.DataFrame(rows).sort_values('dev_oof_tuned_f1', ascending=False).to_csv(OUT / 'comparison.csv', index=False)
    pd.DataFrame({'row_id': dev, 'target': y[dev], 'probability': winner['oof']}).to_csv(OUT / 'selected_oof.csv', index=False)
    pd.DataFrame({'row_id': audit, 'target': y[audit], 'candidate_probability': ap, 'v1_probability': bp}).to_csv(OUT / 'selected_audit.csv', index=False)
    result = {'version': 'v2', 'completed_utc': now(), 'selection': selection, 'candidates': {n: obj['result'] for n, obj in candidates.items()}, 'screening': screen_results, 'selected_audit': audit_score, 'v1_audit': audit_base, 'audit_delta': audit_gain, 'audit_paired_group_bootstrap': bootstrap, 'candidate_ready_for_finalization': bool(ready), 'v1_integrity': verify_parent(), 'submitted': False, 'limitations': ['OOF thresholds/model family/blend are selected on the same development OOF, so scores are selection-biased.', 'The v1 audit was already inspected; v2 uses it only as a frozen-recipe diagnostic.', 'Bootstrap intervals condition on this split, fitted models and selected thresholds; they do not establish unconditional significance.', 'Audit evaluates development fold-averaged models. Final full-data refits will be a different fitted predictor.', 'No final champion promotion or remote submission occurs in this script.']}
    write_json(OUT / 'results.json', result)
    ledger = ROOT / 'kaggle_ops' / 'experiments.jsonl'
    with ledger.open('a', encoding='utf-8') as f:
        for name, obj in candidates.items():
            if not name.endswith('_v1'):
                f.write(json.dumps({'version': 'v2', 'id': name, 'parent': 'baseline_v1', 'timestamp_utc': now(), 'validation': 'Exact v1 development folds; grouped; fixed training rounds; OOF threshold selection', 'result': obj['result'], 'decision': 'provisional_candidate' if name == chosen and ready else 'diagnostic', 'artifact': 'artifacts/v2'}, allow_nan=False) + '\n')
    print(json.dumps({'chosen': chosen, 'dev_f1': winner['result']['dev_oof_tuned_f1'], 'dev_gain': dev_gain, 'audit_f1': audit_score['f1'], 'audit_gain': audit_gain, 'audit_delta_95pct': bootstrap['delta_95pct_interval'], 'ready': ready, 'v1_preserved': True}, indent=2), flush=True)

def finalize(config: dict) -> None:
    result = read_json(OUT / 'results.json')
    if not result['candidate_ready_for_finalization']:
        raise ValueError('Candidate did not pass the predeclared gates; preserve v1 and diagnose')
    verify_parent()
    target_dir = OUT / 'final'
    target_dir.mkdir(exist_ok=False)
    x, test, y, sample, manifest, dev, audit = load_data()
    selection = result['selection']
    probability = np.zeros(len(test))
    components = []
    for name, weight in selection['components'].items():
        if name == 'catboost_v1':
            champion = read_json(ROOT / 'CHAMPION.json')
            model_path = ROOT / champion['model_path']
            model = joblib.load(model_path)
            p = model.predict_proba(prepare_features(test, champion['categorical_columns']))[:, 1]
            info = {'name': name, 'weight': weight, 'path': str(model_path.relative_to(ROOT)).replace('\\', '/'), 'format': 'v1_catboost', 'categorical': champion['categorical_columns']}
        elif name == 'logistic_v1':
            champion = read_json(ROOT / 'CHAMPION.json')
            categorical = champion['categorical_columns']
            numeric = [c for c in x if c not in categorical]
            legacy_config = read_json(ROOT / 'configs' / 'baseline.json')
            model = make_model('logistic', categorical, numeric, legacy_config, config['seed'])
            model.fit(prepare_features(x, categorical), y)
            p = model.predict_proba(prepare_features(test, categorical))[:, 1]
            model_path = target_dir / 'logistic_v1_full.joblib'
            joblib.dump(model, model_path, compress=3)
            info = {'name': name, 'weight': weight, 'path': str(model_path.relative_to(ROOT)).replace('\\', '/'), 'format': 'v1_logistic', 'categorical': categorical}
        else:
            spec = next(s for s in config['candidates'] if s['id'] == name)
            features = build_features(x, spec['features'])
            schema = fit_schema(features, spec['family'])
            model = model_for(spec, config, schema, config['seed'])
            model.fit(transform(features, schema), y)
            p = model.predict_proba(transform(build_features(test, spec['features']), schema))[:, 1]
            model_path = target_dir / f'{name}_full.joblib'
            joblib.dump({'model': model, 'schema': schema, 'spec': spec}, model_path, compress=3)
            info = {'name': name, 'weight': weight, 'path': str(model_path.relative_to(ROOT)).replace('\\', '/'), 'format': 'v2_bundle'}
            pd.DataFrame({'feature': features.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).to_csv(target_dir / f'{name}_importance.csv', index=False)
        info['sha256'] = sha(model_path)
        components.append(info)
        probability += weight * p
    submission = sample.copy()
    submission[TARGET] = (probability >= selection['threshold']).astype(np.int64)
    checked = check_submission(sample, submission)
    path = ROOT / 'submissions' / f'v2_{selection["candidate"]}.csv'
    if path.exists():
        raise FileExistsError(path)
    submission.to_csv(path, index=False)
    pd.DataFrame({'Id': sample.Id, 'probability': probability}).to_csv(target_dir / 'test_probabilities.csv', index=False)
    candidate = {'version': 'v2', 'candidate': selection['candidate'], 'threshold': selection['threshold'], 'components': components, 'input_columns': list(test.columns), 'test_sha256': sha(ROOT / 'data' / 'raw' / 'test.csv'), 'sample_sha256': sha(ROOT / 'data' / 'raw' / 'submission.csv'), 'submission_path': str(path.relative_to(ROOT)).replace('\\', '/'), 'submission_sha256': sha(path), 'validation': checked, 'dev_oof_f1': result['candidates'][selection['candidate']]['dev_oof_tuned_f1'], 'reused_audit_ensemble_f1': result['selected_audit']['f1'], 'submitted': False, 'public_score': None, 'private_score': None, 'v1_preserved': verify_parent(), 'interpretation': 'Provisional v2 candidate, not a proven replacement for v1. Root CHAMPION.json is intentionally unchanged.'}
    write_json(OUT / 'CANDIDATE.json', candidate)
    print(json.dumps(candidate, indent=2), flush=True)

def reproduce(output: Path) -> None:
    candidate = read_json(OUT / 'CANDIDATE.json')
    verify_parent()
    raw = ROOT / 'data' / 'raw'
    if sha(raw / 'test.csv') != candidate['test_sha256'] or sha(raw / 'submission.csv') != candidate['sample_sha256']:
        raise ValueError('Inference input changed')
    test, sample = pd.read_csv(raw / 'test.csv'), pd.read_csv(raw / 'submission.csv')
    if list(test.columns) != candidate['input_columns']:
        raise ValueError('Inference schema changed')
    p = np.zeros(len(test))
    for component in candidate['components']:
        path = ROOT / component['path']
        if sha(path) != component['sha256']:
            raise ValueError('Model artifact changed')
        bundle = joblib.load(path)
        if component['format'] in {'v1_catboost', 'v1_logistic'}:
            pred = bundle.predict_proba(prepare_features(test, component['categorical']))[:, 1]
        else:
            pred = bundle['model'].predict_proba(transform(build_features(test, bundle['spec']['features']), bundle['schema']))[:, 1]
        p += component['weight'] * pred
    sample[TARGET] = (p >= candidate['threshold']).astype(np.int64)
    check = check_submission(pd.read_csv(raw / 'submission.csv'), sample)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    check.update({'sha256': sha(output), 'exact_match': sha(output) == candidate['submission_sha256'], 'v1_integrity': verify_parent()})
    if not check['exact_match']:
        raise ValueError('Reproduced submission hash differs')
    write_json(OUT / 'reproduction_check.json', check)
    print(json.dumps(check, indent=2), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['develop', 'finalize', 'reproduce', 'verify'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'v2.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'submissions' / 'v2_reproduced.csv')
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == 'develop':
        develop(config, args.resume)
    elif args.phase == 'finalize':
        finalize(config)
    elif args.phase == 'reproduce':
        reproduce(args.output)
    else:
        print(json.dumps(verify_parent(), indent=2))

if __name__ == '__main__':
    main()
