"""Rebuild v2 in a new isolated directory using a frozen v1 parent; no network."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    allowed = (ROOT / 'reproduction_runs').resolve()
    destination = (args.destination or allowed / f'v2_{stamp}').resolve()
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError('Destination must be a new child of this project/reproduction_runs')
    if destination.exists():
        raise FileExistsError('Never overwrite a previous reproduction run')
    parent_manifest = read_json(ROOT / 'versions' / 'v1' / 'manifest.json')
    provenance = read_json(ROOT / 'artifacts' / 'v2' / 'provenance.json')
    for relative, expected in parent_manifest['sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Protected v1 artifact changed: {relative}')
    for name, expected in provenance['code_sha256'].items():
        if sha(ROOT / 'artifacts' / 'v2' / 'source' / name) != expected:
            raise ValueError(f'Frozen v2 source changed: {name}')
    destination.mkdir(parents=True)
    (destination / 'kaggle_ops').mkdir()
    def copy(source: Path, relative: str):
        target = destination / relative
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Invalid relative path in manifest')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Only an explicit, hash-verified artifact whitelist is copied. No credentials.
    for relative in parent_manifest['sha256']:
        copy(ROOT / relative, relative)
    for name in ['agents.md', 'discoveries.md', 'handoff.md', 'plan.md', 'README.md']:
        copy(ROOT / 'versions' / 'v1' / name, name)
    for name in provenance['code_sha256']:
        copy(ROOT / 'artifacts' / 'v2' / 'source' / name, f'src/{name}')
    copy(ROOT / 'artifacts' / 'v2' / 'config.json', 'configs/v2.json')
    copy(ROOT / 'tests' / 'test_v2.py', 'tests/test_v2.py')
    print(f'ISOLATED_RETRAIN_ROOT={destination}', flush=True)
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        [sys.executable, '-u', 'src/v2_pipeline.py', 'develop'],
        [sys.executable, '-u', 'src/v2_pipeline.py', 'finalize'],
        [sys.executable, '-u', 'src/v2_pipeline.py', 'reproduce'],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=destination, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
        output = completed.stdout + completed.stderr
        with (destination / 'retrain.log').open('a', encoding='utf-8') as log:
            log.write('\nCOMMAND: ' + repr(command) + '\n' + output)
        print(output, flush=True)
        if completed.returncode:
            raise RuntimeError(f'Reproduction command failed with code {completed.returncode}; inspect {destination / "retrain.log"}')
    expected = read_json(ROOT / 'artifacts' / 'v2' / 'CANDIDATE.json')
    actual = read_json(destination / 'artifacts' / 'v2' / 'CANDIDATE.json')
    exact_csv = actual['submission_sha256'] == expected['submission_sha256']
    if not exact_csv or actual['candidate'] != expected['candidate'] or actual['threshold'] != expected['threshold']:
        raise ValueError('Full retraining failed to reproduce the selected recipe and exact submission CSV')
    for relative, expected_hash in parent_manifest['sha256'].items():
        if sha(ROOT / relative) != expected_hash:
            raise ValueError(f'Original v1 changed during reproduction: {relative}')
    receipt = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'v2 full retraining with frozen v1 reference artifacts', 'isolated_root': str(destination), 'candidate': actual['candidate'], 'threshold': actual['threshold'], 'submission_sha256': actual['submission_sha256'], 'exact_submission_match': exact_csv, 'v1_protected_files_verified': len(parent_manifest['sha256']), 'commands_passed': len(commands), 'remote_calls': 0}
    (destination / 'retraining_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    receipt_dir = ROOT / 'reports' / 'v2_retraining'
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / f'{stamp}.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('FULL_RETRAIN_VERIFIED=' + json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
