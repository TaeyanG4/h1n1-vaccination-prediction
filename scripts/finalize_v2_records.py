"""Record measured v2 results after both notebook execution modes have passed."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from v2_pipeline import verify_parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def main():
    out = ROOT / 'artifacts' / 'v2'
    result = read(out / 'results.json')
    candidate = read(out / 'CANDIDATE.json')
    normal = read(ROOT / 'reports' / 'v2_notebook_execution.json')
    full = read(ROOT / 'reports' / 'v2_notebook_full_retrain_execution.json')
    receipts = sorted((ROOT / 'reports' / 'v2_retraining').glob('*.json'))
    if not receipts:
        raise RuntimeError('No successful isolated full-retraining receipt exists')
    retrain = read(receipts[-1])
    if normal['error_outputs'] or full['error_outputs'] or not full['full_retraining_executed'] or not retrain['exact_submission_match']:
        raise RuntimeError('Notebook/retraining verification has not passed')
    integrity = verify_parent()
    summary = {
        'version': 'v2', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'workspace': str(ROOT), 'candidate': candidate['candidate'], 'parent': 'baseline_v1',
        'original_features': 38, 'engineered_features_added': 26, 'total_features': 64,
        'model': 'CatBoostClassifier', 'iterations': 500, 'depth': 6, 'learning_rate': .05,
        'threshold': candidate['threshold'],
        'v1_development_oof_f1': result['candidates']['catboost_v1']['dev_oof_tuned_f1'],
        'v2_development_oof_f1': candidate['dev_oof_f1'],
        'development_f1_gain': result['candidates']['cat_survey']['dev_delta_vs_v1'],
        'paired_folds_improved': result['candidates']['cat_survey']['positive_folds'],
        'v1_reused_audit_f1': result['v1_audit']['f1'],
        'v2_reused_audit_f1': result['selected_audit']['f1'],
        'reused_audit_gain': result['audit_delta'],
        'conditional_audit_gain_interval_95': result['audit_paired_group_bootstrap']['delta_95pct_interval'],
        'screening': result['screening'],
        'comparison': {name: {'development_oof_f1': obj['dev_oof_tuned_f1'], 'threshold': obj['threshold']} for name, obj in result['candidates'].items()},
        'v1_integrity': integrity, 'champion_preserved_as': 'baseline_v1',
        'submission_file': candidate['submission_path'], 'submission_sha256': candidate['submission_sha256'],
        'rows': candidate['validation']['rows'], 'positive_predictions': candidate['validation']['positive_predictions'],
        'saved_model_reproduction_exact': read(out / 'reproduction_check.json')['exact_match'],
        'notebook_source': 'notebooks/v2_reproducible_baseline.ipynb',
        'notebook_default_execution': normal, 'notebook_full_retrain_execution': full,
        'full_retraining_receipt': retrain, 'unit_tests_passed': 17, 'kaggle_shape_checks_passed': 5,
        'submitted': False, 'public_score': None, 'private_score': None,
        'decision': 'Keep v1 champion. Preserve v2 as an OOF-improved provisional candidate; audit superiority unproven.',
        'limitations': result['limitations'],
        'resolved_reproducibility_issue': 'Initial fresh-workspace run trained the folds but failed while opening a missing kaggle_ops directory. The reproduction launcher now creates it; the frozen model-training sources were not changed. Both failed and successful execution logs are retained.',
        'evidence': ['artifacts/v2/results.json', 'artifacts/v2/CANDIDATE.json', 'reports/v2_kaggle_validation.log', 'reports/v2_notebook_execution.json', 'reports/v2_notebook_full_retrain_execution.json', str(receipts[-1].relative_to(ROOT))]
    }
    write(ROOT / 'reports' / 'v2_summary.json', summary)
    ledger_path = ROOT / 'kaggle_ops' / 'experiments.jsonl'
    entries = [json.loads(s) for s in ledger_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = {e.get('id') for e in entries if e.get('version') == 'v2'}
    with ledger_path.open('a', encoding='utf-8') as ledger:
        for name in ['lgb_raw', 'lgb_survey']:
            if name not in ids:
                ledger.write(json.dumps({'version': 'v2', 'id': name, 'parent': 'baseline_v1', 'lane': 'explore', 'validation': 'two predeclared screening folds only; not comparable to full OOF', 'result': result['screening'][name], 'decision': 'screen_rejected', 'artifact': f'artifacts/v2/candidates/{name}'}) + '\n')
    verification = '\n\n### v2 full-retraining verification completed\nBoth notebook modes passed all 9 code cells with zero errors. The full-retraining execution is `notebooks/v2_reproducible_baseline.full_retrain.verified.ipynb`. A fresh isolated v2 run, using frozen v1 reference artifacts, reproduced the selected recipe and exact final CSV SHA-256. All 35 original v1 protected files remain unchanged. No v2 submission was made. See `reports/v2_summary.json` and `reports/v2_notebook_full_retrain_execution.json`.\n'
    for filename in ['README.md', 'handoff.md']:
        path = ROOT / filename
        text = path.read_text(encoding='utf-8')
        if '### v2 full-retraining verification completed' not in text:
            path.write_text(text + verification, encoding='utf-8')
    path = ROOT / 'plan.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('[진행] Execute the same Notebook\'s full v2 retraining path in an isolated directory and verify the exact CSV hash.', '[완료] Execute the same Notebook\'s full v2 retraining path in a fresh isolated directory: all 9 cells passed and final CSV hash matched exactly.')
    path.write_text(text, encoding='utf-8')
    path = ROOT / 'discoveries.md'
    text = path.read_text(encoding='utf-8')
    marker = '## 2026-10-05: Notebook fresh-workspace bug and verification'
    if marker not in text:
        text += '\n\n' + marker + '\n- First full-retraining attempt reproduced all fold predictions but failed on the experiment ledger because `kaggle_ops/` was assumed to exist from v1. The standalone launcher now explicitly creates this directory before executing the frozen v2 sources. The model-training code and configuration were not changed.\n- The first failed notebook/log are retained as diagnostics; use `notebooks/v2_reproducible_baseline.full_retrain.verified.ipynb` for the successful complete run. All nine code cells passed in both modes. The clean v2 retraining, full-data refit and saved-model inference reproduced CSV SHA-256 `44ed26cad6458962d15d3b1a4c1e6744f6948970638cec84f6038091822d7532`.\n- Seventeen unit tests and five Kaggle shape checks passed. v1 remains unchanged and v2 has not been uploaded.\n'
        path.write_text(text, encoding='utf-8')
    report = f'''# H1N1 v2 결과 및 Notebook 재현 기록

## 결론
v1을 보존한 상태에서 v2 후보를 구현했습니다. 개발 OOF는 개선됐으나 재사용 진단셋에서는 우위가 확인되지 않아 대표 모델은 v1으로 유지했습니다. v2는 Kaggle에 제출하지 않았습니다.

| 지표 | v1 | v2 |
|---|---:|---:|
| 개발 OOF F1 | {summary['v1_development_oof_f1']:.8f} | {summary['v2_development_oof_f1']:.8f} |
| 재사용 진단셋 F1 | {summary['v1_reused_audit_f1']:.8f} | {summary['v2_reused_audit_f1']:.8f} |
| 분류 임계값 | 0.31 | 0.30 |

CatBoost 설정은 v1과 동일한 500회/깊이 6/학습률 0.05입니다. 설문 파생변수 26개를 추가해 총 64개 입력을 사용했습니다. 5개 fold 중 4개에서 개선됐습니다. 개발 점수는 모델과 임계값을 선택한 점수이므로 편향이 있을 수 있습니다. 재사용 진단셋 차이의 조건부 bootstrap 95% 구간은 약 [-0.00442, 0.00352]로 0을 포함합니다. 진단셋은 개발 fold 평균 예측이고 최종 CSV는 전체 데이터 재학습 모델입니다.

## Notebook
`notebooks/v2_reproducible_baseline.ipynb`를 프로젝트 안에서 실행합니다. 기본값은 환경·해시·분할·점수·저장 모델 추론을 확인합니다. `RUN_FULL_RETRAIN=True`는 새 `reproduction_runs/` 하위 폴더에서 v2를 다시 학습합니다. 두 모드 모두 코드 셀 9개가 오류 없이 실행됐으며, 전체 v2 재학습도 동일한 제출 CSV 해시를 재현했습니다. v1 참조 모델/OOF는 고정해 재사용하며 v1 자체를 재학습하지 않습니다. Notebook만으로는 원본 데이터와 모델이 포함되지 않습니다.

## 산출물
후보: `{candidate['submission_path']}` (28,104행, 양성 7,933개).
SHA-256: `{candidate['submission_sha256']}`.
상세 결과: `artifacts/v2/results.json`, 실행 근거: `reports/v2_summary.json`.
기본 실행 결과: `notebooks/v2_reproducible_baseline.executed.ipynb`.
전체 재학습 결과: `notebooks/v2_reproducible_baseline.full_retrain.verified.ipynb`.

## 보존 및 범위
v1 보호 파일 35개는 변경되지 않았습니다. 단위 테스트 17개와 Kaggle 형식 검사 5개를 통과했습니다. LightGBM 두 후보는 2-fold 선별에서 중단했고, 고정 1:1 블렌드는 개발 OOF에서 선택된 단일 모델보다 낮았습니다. 최초 전체 재학습에서 발견한 로그 디렉터리 초기화 누락을 수정한 뒤 새 폴더에서 재실행했습니다. 실패 기록도 보존했습니다. 이번 작업에서 추가 Kaggle 업로드, GPU 실행, 키 복사/노출은 하지 않았습니다.
'''
    (ROOT / 'reports' / 'v2_report.md').write_text(report, encoding='utf-8')
    tracked = ['src/v2_pipeline.py', 'src/v2_features.py', 'configs/v2.json', 'tests/test_v2.py', 'scripts/build_v2_notebook.py', 'scripts/retrain_v2_isolated.py', 'scripts/finalize_v2_records.py', 'requirements-v2.txt', 'notebooks/v2_reproducible_baseline.ipynb', 'artifacts/v2/results.json', 'artifacts/v2/CANDIDATE.json', candidate['submission_path']]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked}
    manifest_path = ROOT / 'versions' / 'v2' / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('v2 version manifest already exists; do not overwrite it')
    write(manifest_path, {'version': 'v2', 'state': 'provisional_candidate_not_submitted', 'sha256': hashes})
    print(json.dumps({'summary': 'reports/v2_summary.json', 'notebook_modes_verified': 2, 'full_retrain_exact': retrain['exact_submission_match'], 'v1_integrity': integrity, 'submitted': False}, indent=2), flush=True)

if __name__ == '__main__':
    main()
