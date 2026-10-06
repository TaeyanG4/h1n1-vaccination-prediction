"""Audit official, immutable competition files without changing them."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    train, test, labels, sample = [pd.read_csv(RAW / f) for f in
        ('train.csv', 'test.csv', 'train_labels.csv', 'submission.csv')]
    assert list(train.columns) == list(test.columns), 'Train/test column mismatch'
    assert len(train) == len(labels), 'Label row count mismatch'
    assert list(sample.columns) == ['Id', 'vacc_h1n1_f'], 'Unexpected output contract'
    assert len(test) == len(sample), 'Sample/test row mismatch'
    assert sample.Id.is_unique and sample.Id.notna().all()
    y = labels.vacc_h1n1_f
    assert y.notna().all() and set(y.unique()) == {0, 1}
    assert 'vacc_h1n1_f' not in train and 'vacc_seas_f' not in train
    combined = pd.concat([train, test], ignore_index=True)
    hashes = pd.util.hash_pandas_object(combined.fillna('__MISSING__').astype(str), index=False).astype(str)
    a, b = hashes.iloc[:len(train)], hashes.iloc[len(train):]
    dup_labels = pd.DataFrame({'hash': a.to_numpy(), 'y': y.to_numpy()}).groupby('hash').y.nunique()
    versions = {}
    for package in ('pandas', 'numpy', 'scikit-learn', 'catboost', 'joblib', 'kaggle'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        'train_shape': list(train.shape), 'test_shape': list(test.shape),
        'target': 'vacc_h1n1_f', 'ignored_label': 'vacc_seas_f',
        'label_counts': {str(k): int(v) for k, v in y.value_counts().items()},
        'positive_rate': float(y.mean()), 'duplicate_train_rows': int(a.duplicated().sum()),
        'duplicate_test_rows': int(b.duplicated().sum()),
        'train_test_feature_overlap_unique': int(len(set(a) & set(b))),
        'conflicting_label_duplicate_groups': int((dup_labels > 1).sum()),
        'train_missing_fraction': {k: float(v) for k, v in train.isna().mean().items()},
        'test_missing_fraction': {k: float(v) for k, v in test.isna().mean().items()},
        'train_dtypes': train.dtypes.astype(str).to_dict(),
        'nunique_train': {k: int(v) for k, v in train.nunique().items()},
        'unseen_test_categories': {c: int(len(set(test[c].dropna().astype(str)) - set(train[c].dropna().astype(str)))) for c in train},
        'raw_sha256': {p.name: sha256(p) for p in sorted(RAW.glob('*.csv'))},
        'python': sys.version, 'platform': platform.platform(), 'package_versions': versions,
    }
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    (out / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k, v in versions.items() if v) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('train_missing_fraction', 'test_missing_fraction', 'train_dtypes', 'nunique_train', 'unseen_test_categories')}, indent=2))

if __name__ == '__main__':
    main()
