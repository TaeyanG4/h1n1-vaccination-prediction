"""Regenerate the champion CSV from its saved model; never retrain or submit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline import check_submission, digest, prepare_features

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'submissions' / 'reproduced.csv')
    args = parser.parse_args()
    champion = json.loads((ROOT / 'CHAMPION.json').read_text(encoding='utf-8'))
    summary = json.loads((ROOT / champion['metrics_path']).read_text(encoding='utf-8'))
    raw = ROOT / 'data' / 'raw'
    for name, expected in summary['raw_sha256'].items():
        if digest(raw / name) != expected:
            raise ValueError(f'Raw data fingerprint changed: {name}')
    test = pd.read_csv(raw / 'test.csv')
    sample = pd.read_csv(raw / 'submission.csv')
    if list(test.columns) != champion['columns']:
        raise ValueError('Inference feature schema changed')
    features = prepare_features(test, champion['categorical_columns'])
    model = joblib.load(ROOT / champion['model_path'])
    probability = model.predict_proba(features)[:, 1]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid inference probabilities')
    candidate = sample.copy()
    candidate['vacc_h1n1_f'] = (probability >= champion['threshold']).astype(np.int64)
    report = check_submission(sample, candidate)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must remain inside the project')
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output, index=False)
    actual = digest(output)
    if actual != champion['submission_sha256']:
        raise ValueError(f'Reproduced file hash differs: {actual}')
    print(json.dumps({**report, 'sha256': actual, 'matches_champion': True, 'path': str(output)}, indent=2))

if __name__ == '__main__':
    main()
