"""Build and optionally execute the reproducibility notebook without remote actions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def build():
    cells = []
    def add(kind, source):
        source = textwrap.dedent(source).strip()
        fn = nbformat.v4.new_code_cell if kind == 'code' else nbformat.v4.new_markdown_cell
        cell = fn(source)
        cell['id'] = hashlib.sha256((str(len(cells)) + source).encode('utf-8')).hexdigest()[:12]
        cells.append(cell)
    add('markdown', '''
    # H1N1 vaccination — v2 재현 Notebook

    **v1을 보존하고, v2 실험·검증·제출 파일을 재현합니다.**

    이 Notebook은 프로젝트의 Python 소스와 저장된 공식 데이터·모델을 사용합니다.
    단독 파일만으로는 학습 데이터나 모델이 포함되지 않습니다.

    기본 실행은 환경/해시 확인, 실험 결과 재계산, 저장 모델 추론 재현입니다.
    `RUN_FULL_RETRAIN=True`로 바꾸면 새로운 격리 폴더에서 v2 전체 실험과 최종 모델을 다시 학습합니다.
    이때 v1 기준 모델/OOF는 고정된 참조로 재사용하며, v1 자체를 재학습하지 않습니다.
    **어떤 모드에서도 Kaggle 제출·다운로드·인증 변경을 하지 않습니다.**
    ''')
    add('code', '''
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib
    import importlib.metadata
    import json
    import os
    import subprocess
    import sys
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown

    RUN_FULL_RETRAIN = False
    # 다른 위치로 옮긴 경우 H1N1_PROJECT_ROOT 환경변수 또는 아래 탐색 경로를 지정합니다.
    roots = []
    if os.environ.get('H1N1_PROJECT_ROOT'):
        roots.append(Path(os.environ['H1N1_PROJECT_ROOT']))
    roots.extend([Path.cwd(), *Path.cwd().parents])
    ROOT = next((p.resolve() for p in roots if (p / 'configs' / 'v2.json').is_file() and (p / 'artifacts' / 'v2').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('프로젝트 안에서 실행하거나 H1N1_PROJECT_ROOT를 설정하세요.')
    OUT = ROOT / 'artifacts' / 'v2'
    sys.path.insert(0, str(ROOT / 'src'))
    import v2_pipeline as vp
    from v2_features import build_features, fit_schema, transform
    from sklearn.metrics import f1_score

    def read_json(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    print('프로젝트:', ROOT)
    print('Python:', sys.version.split()[0])
    print('전체 v2 재학습:', RUN_FULL_RETRAIN)
    ''')
    add('markdown', '''
    ## 1. 환경과 원본 보존 확인

    정확한 재현은 기록된 패키지 버전과 동일한 데이터/소스가 전제입니다.
    `requirements-v2.txt`는 설치 참조용이며 이 Notebook에서 패키지를 자동 설치하지 않습니다.
    모델은 이 프로젝트가 만든 신뢰할 수 있는 파일만 불러옵니다.
    ''')
    add('code', '''
    provenance = read_json(OUT / 'provenance.json')
    versions = pd.DataFrame([
        {'package': name, 'expected': version, 'installed': importlib.metadata.version(name)}
        for name, version in provenance['versions'].items()
    ])
    display(versions)
    assert (versions['expected'] == versions['installed']).all(), '패키지 버전이 다릅니다.'
    for name, expected in provenance['code_sha256'].items():
        assert vp.sha(ROOT / 'src' / name) == expected, f'소스가 변경됨: {name}'
    integrity = vp.verify_parent()
    print('v1 원본 보존:', integrity)
    assert vp.sha(OUT / 'split_manifest.csv') == provenance['split_sha256']
    ''')
    add('markdown', '''
    ## 2. 데이터와 동일한 검증 분할

    예측 대상은 `vacc_h1n1_f`입니다. 테스트에서 알 수 없는 정답 `vacc_seas_f`는 입력에서 제외합니다.
    동일 입력 행의 해시 그룹이 학습과 검증에 함께 들어가지 않도록 합니다.
    v1의 개발 33,722행/진단 8,432행 및 개발용 5-fold를 그대로 사용합니다.
    **진단셋은 v1에서 이미 점수를 확인했으므로 새로운 독립 검증셋이 아닙니다.**
    ''')
    add('code', '''
    x, test, y, sample, manifest, dev, audit = vp.load_data()
    assert len(x) == len(y) == len(manifest)
    assert {'vacc_h1n1_f', 'vacc_seas_f'}.isdisjoint(x.columns)
    assert set(manifest.loc[dev, 'group_hash']).isdisjoint(manifest.loc[audit, 'group_hash'])
    for fold in range(5):
        train_groups = set(manifest.loc[dev[manifest.loc[dev, 'dev_fold'].to_numpy() != fold], 'group_hash'])
        valid_groups = set(manifest.loc[dev[manifest.loc[dev, 'dev_fold'].to_numpy() == fold], 'group_hash'])
        assert train_groups.isdisjoint(valid_groups)
    display(pd.DataFrame({'rows': [len(x), len(test), len(dev), len(audit)]}, index=['train', 'test', 'development', 'reused audit']))
    display(manifest.groupby(['partition', 'dev_fold']).size().rename('rows').to_frame())
    display(x.isna().mean().sort_values(ascending=False).head(8).rename('missing_fraction').to_frame())
    ''')
    add('markdown', '''
    ## 3. v2의 변화: 설문 파생변수

    결측 개수/구간별 결측, 무응답, 예방행동 요약, 의사 권고 조합,
    백신 효과·위험 인식의 순서형 표현 및 차이, 가구 내 아동 비율을 추가합니다.
    모든 파생변수는 **해당 행의 입력만 사용**하며 정답이나 전체 데이터 통계를 사용하지 않습니다.
    LightGBM 범주 목록은 각 fold의 학습 데이터에서만 정의합니다.
    원래 의견 범주도 함께 보존하며, `Dont Know`를 낮은 위험 등의 숫자로 임의 대체하지 않습니다.
    ''')
    add('code', '''
    survey = build_features(x, 'survey')
    added = [c for c in survey if c not in x]
    pd.testing.assert_frame_equal(build_features(x.iloc[[0]], 'survey'), survey.iloc[[0]])
    assert list(x.columns) == list(test.columns)
    print(f'원본 {len(x.columns)}개 -> 파생변수 포함 {len(survey.columns)}개, 추가 {len(added)}개')
    display(pd.DataFrame({'added_feature': added}))
    config = read_json(OUT / 'config.json')
    display(pd.DataFrame(config['candidates']))
    print('CatBoost:', config['catboost'])
    print('LightGBM:', config['lightgbm'])
    ''')
    add('markdown', '''
    ## 4. 선택 사항: v2 전체 재학습

    첫 코드 셀의 `RUN_FULL_RETRAIN=True`를 사용합니다. 별도의 `reproduction_runs/v2_<시각>/`를 만들고
    해시로 확인된 v1 참조·원본 데이터·고정된 v2 소스만 복사합니다. 키/인증 폴더는 복사하지 않습니다.
    2-fold 후보 선별 -> 통과 후보 5-fold -> 고정 1:1 블렌드 비교 -> 모델/임계값 고정 ->
    기존 진단셋 평가 -> 전체 데이터 재학습 -> 제출 CSV 해시 비교를 수행합니다.
    v1/v2 원본 결과를 덮어쓰지 않고, 오류를 무시하지 않습니다.
    ''')
    add('code', '''
    if RUN_FULL_RETRAIN:
        completed = subprocess.run(
            [sys.executable, '-u', str(ROOT / 'scripts' / 'retrain_v2_isolated.py')],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False
        )
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'v2 재학습 실패: exit={completed.returncode}')
        assert 'FULL_RETRAIN_VERIFIED=' in completed.stdout
    else:
        print('전체 재학습은 생략했습니다. 아래에서 저장 모델 추론을 실제로 재현합니다.')
    ''')
    add('markdown', '''
    ## 5. 실험 결과: 선별 점수와 최종 OOF를 구분

    LightGBM의 2-fold 선별 결과는 전체 5-fold 점수와 직접 비교하지 않습니다.
    최종 비교는 동일한 개발 데이터의 OOF입니다. F1 임계값은 0.15~0.60의 사전 고정 격자에서 선택합니다.
    모델·임계값을 이 OOF로 선택했으므로 튜닝된 OOF 점수에는 선택 편향이 있습니다.
    ''')
    add('code', '''
    screening = read_json(OUT / 'screening.json')
    print('2-fold 기준 v1 F1:', screening['reference_f1'])
    display(pd.DataFrame(screening['candidates']).T)
    comparison = pd.read_csv(OUT / 'comparison.csv')
    display(comparison)
    result = read_json(OUT / 'results.json')
    selection = result['selection']
    print('OOF로 고정한 후보:', selection['candidate'], 'threshold:', selection['threshold'])
    print('고정 시각 UTC:', selection['frozen_utc'])
    ''')
    add('markdown', '''
    ## 6. 저장된 예측으로 점수 재계산

    아래는 이미 고정된 임계값으로 점수를 다시 계산하는 검증 단계입니다.
    진단셋에서 임계값이나 모델을 다시 고르지 않습니다.
    F1은 정확도(accuracy)가 아니며, 서로 다른 평가 데이터의 점수 차이를 개선량으로 해석하지 않습니다.
    ''')
    add('code', '''
    oof = pd.read_csv(OUT / 'selected_oof.csv')
    audit_predictions = pd.read_csv(OUT / 'selected_audit.csv')
    threshold = selection['threshold']
    recomputed_oof = f1_score(oof.target, oof.probability >= threshold)
    recomputed_audit = f1_score(audit_predictions.target, audit_predictions.candidate_probability >= threshold)
    chosen_metrics = result['candidates'][selection['candidate']]
    assert abs(recomputed_oof - chosen_metrics['dev_oof_tuned_f1']) < 1e-12
    assert abs(recomputed_audit - result['selected_audit']['f1']) < 1e-12
    display(pd.DataFrame({
        'v1': [result['candidates']['catboost_v1']['dev_oof_tuned_f1'], result['v1_audit']['f1']],
        'v2': [recomputed_oof, recomputed_audit],
    }, index=['development OOF (selected)', 'reused audit (fold ensemble)']))
    display(pd.DataFrame({'fold': range(1, 6), 'F1_delta_v2_minus_v1': chosen_metrics['paired_fold_deltas']}))
    display(pd.Series(chosen_metrics['threshold_neighborhood_f1'], name='development_F1').to_frame())
    print('그룹 bootstrap 진단:', result['audit_paired_group_bootstrap'])
    ''')
    add('markdown', '''
    ## 7. 최종 저장 모델로 제출 CSV 재생성

    전체 학습 데이터로 적합한 최종 모델을 다시 불러오고, 원본 테스트 순서로 0/1을 예측합니다.
    행 수, ID/열 순서, 이진값, 결측값, 원본 파일·모델 해시를 확인한 후
    생성한 CSV가 동결된 후보와 **바이트 단위 SHA-256까지 같은지** 확인합니다.
    이 단계는 재학습이 아니라 저장 모델 추론 재현입니다.
    ''')
    add('code', '''
    candidate = read_json(OUT / 'CANDIDATE.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = OUT / 'notebook_runs' / stamp / 'submission.csv'
    vp.reproduce(output)
    assert vp.sha(output) == candidate['submission_sha256']
    print('Notebook 생성 CSV:', output)
    print('SHA-256:', vp.sha(output))
    print('행 수/형식 검사:', candidate['validation'])
    assert read_json(ROOT / 'CHAMPION.json')['run_id'] == 'baseline_v1'
    print('기존 CHAMPION은 v1으로 유지:', vp.verify_parent())
    ''')
    add('markdown', '''
    ## 8. 상태와 해석

    v2는 개발 OOF에서 개선된 **잠정 후보**입니다. 재사용 진단셋에서 우위가 입증됐다는 뜻은 아닙니다.
    그룹 bootstrap 구간은 현재 분할·학습모델·선택된 임계값에 조건부이며, 선택 과정 전체의 불확실성을 포함하지 않습니다.
    또한 진단 점수는 개발 fold 평균 예측이고, 제출 CSV는 전체 데이터 재학습 모델의 예측입니다.
    Kaggle에 제출하지 않은 v2에는 Public/Private 점수가 없습니다. 이 Notebook은 업로드하지 않습니다.
    ''')
    add('code', '''
    display(pd.DataFrame([
        {'version': 'v1', 'status': 'previously submitted', 'public_F1': 0.63655, 'private_F1': 0.63036},
        {'version': 'v2', 'status': 'local candidate, not submitted', 'public_F1': None, 'private_F1': None},
    ]))
    print('모든 Notebook 재현 검사를 통과했습니다.')
    ''')
    add('markdown', '''
    ## 근거 파일과 참고 문서

    로컬 측정: `artifacts/v2/results.json`, `screening.json`, `comparison.csv`, `CANDIDATE.json`,
    `versions/v1/manifest.json`, `reports/submission_receipt.json`.

    구현 참고: [LightGBM 4.6.0 범주형/결측 처리](https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html),
    [CatBoost 공식 파라미터 설명](https://catboost.ai/docs/en/concepts/parameter-tuning),
    [Jupyter NBClient 실행 문서](https://nbclient.readthedocs.io/en/latest/client.html).
    외부 문서의 일반적 조언과 이 데이터에서 측정한 실제 개선 여부는 구분합니다.
    ''')
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3 (H1N1 v2)', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12.6'}, 'h1n1': {'version': 'v2', 'default_mode': 'saved_model_reproduction', 'submits_to_kaggle': False}}
    nbformat.validate(notebook)
    return notebook

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--full-retrain', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'notebooks'
    directory.mkdir(exist_ok=True)
    path = directory / 'v2_reproducible_baseline.ipynb'
    notebook = build()
    if not path.exists():
        nbformat.write(notebook, path)
    else:
        old = nbformat.read(path, as_version=4)
        if [c.source for c in old.cells] != [c.source for c in notebook.cells]:
            raise FileExistsError('Notebook source changed; preserve it before replacing')
    dependencies = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'lightgbm', 'joblib', 'nbformat', 'nbclient', 'ipykernel', 'jupyter-client']
    lock = ROOT / 'requirements-v2.txt'
    if not lock.exists():
        lock.write_text('# Python 3.12.6; measured Windows environment\n' + '\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in dependencies) + '\n', encoding='utf-8')
    print(f'NOTEBOOK_CREATED={path}', flush=True)
    if not args.execute:
        return
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    if args.full_retrain:
        notebook.cells[1].source = notebook.cells[1].source.replace('RUN_FULL_RETRAIN = False', 'RUN_FULL_RETRAIN = True')
    suffix = '.full_retrain.verified.ipynb' if args.full_retrain else '.executed.ipynb'
    executed_path = directory / ('v2_reproducible_baseline' + suffix)
    if executed_path.exists():
        raise FileExistsError(executed_path)
    # Pin the kernel executable to the interpreter that trained these models.
    km = KernelManager(kernel_name='python3')
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=km, timeout=1800, allow_errors=False, resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        nbformat.write(notebook, executed_path)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    errors = [out for c in code_cells for out in c.get('outputs', []) if out.output_type == 'error']
    if errors or any(c.execution_count is None for c in code_cells):
        raise RuntimeError('Notebook contains errors or unexecuted code cells')
    receipt = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'notebook': str(executed_path.relative_to(ROOT)), 'code_cells_executed': len(code_cells), 'error_outputs': len(errors), 'full_retraining_executed': args.full_retrain, 'kernel_python': sys.executable, 'notebook_sha256': hashlib.sha256(executed_path.read_bytes()).hexdigest()}
    name = 'v2_notebook_full_retrain_execution.json' if args.full_retrain else 'v2_notebook_execution.json'
    (ROOT / 'reports' / name).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)

if __name__ == '__main__':
    main()
