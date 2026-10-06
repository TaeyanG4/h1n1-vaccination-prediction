# H1N1 Vaccination Prediction

Reproducible tabular-ML research for the Kaggle competition **Prediction of H1N1 vaccination**.

The project started from a CatBoost baseline and evolved through grouped validation, AutoGluon stacking, XGBoost/LightGBM optimization, TabM, EBM, RealMLP, and constrained ensemble research. The final result came from combining several individually different error patterns rather than from one dominant standalone model.

> Kaggle competition: `prediction-of-h1n1-vaccination`
>
> Metric: binary **F1** for `vacc_h1n1_f`

## Final result

| Item | Result |
|---|---:|
| Final candidate | `v21_base_ebm_15_realmlp_10_single` |
| Grouped 5-fold nested OOF F1 | **0.639157** |
| Frozen threshold | **0.325** |
| Kaggle public F1 | **0.63267** |
| Kaggle private F1 | **0.63545** |
| Kaggle submission ref | `56876458` |

The competition ended in 2023. These leaderboard values come from accepted late submissions made during this reproducibility/research campaign in 2026, so they should **not** be interpreted as a historical competition rank.

The final private score improved over the previous best result found during this project (`0.63483`) by `+0.00062`.

## Final ensemble

The champion is a probability blend of five model families:

| Component | Effective weight |
|---|---:|
| v12 AutoGluon exact-44 full stack | 45.0% |
| v18 XGBoost | 22.5% |
| v19 LightGBM | 7.5% |
| v21 EBM | 15.0% |
| v21 RealMLP-TD | 10.0% |

The last two models were weak as standalone classifiers, but their prediction errors were sufficiently different from the stronger tree/stack models to improve the final blend.

The frozen threshold is `0.325`. It was identical across all five outer validation folds for the selected blend.

## Validation design

The training set contains **42,154 rows** and 38 input columns. The positive-class rate is about **23.9%**.

Validation was designed around duplicate-profile leakage risk:

- identical feature rows are grouped by an exact feature hash;
- 33,722 rows form the development set;
- 8,432 rows form a locked audit set;
- model selection on development data uses five stratified group folds;
- thresholds are selected from OOF predictions only;
- serious submission candidates are refit on **all 42,154 labeled rows**;
- the auxiliary target `vacc_seas_f` is never used as a feature.

The dataset contains exact duplicate feature profiles and some duplicate groups with conflicting labels, which is why ordinary random K-fold validation was intentionally avoided.

## Experiment progression

| Stage | Main idea | Private F1 / decision |
|---|---|---:|
| v1 | CatBoost baseline | 0.63036 |
| v12 | AutoGluon exact-44 full stack | 0.63376 |
| v18 | v12 + XGBoost rank blend | 0.63439 |
| v19 | v12 + XGB + LightGBM rank blend | 0.63483 |
| v20 | TabM diversity blend | 0.63372 |
| v21 | base + 15% RealMLP | 0.63534 |
| **v21** | **base + 15% EBM + 10% RealMLP** | **0.63545** |
| v22 | nested logistic stack + slice gating | rejected locally |

The exact OOF-best v21 blend was **not** the private-LB best. This was an important stopping signal: local validation was useful for identifying a promising blend region, but not reliable enough to justify fine-grained leaderboard weight tuning.

## What worked

- Grouping duplicate feature profiles before cross-validation.
- Preserving OOF predictions from every serious model family.
- Searching for **diversity**, not only standalone F1.
- Full-label refitting after model/threshold selection.
- Simple low-dimensional probability/rank blends before flexible stackers.
- Keeping single-seed and multi-seed variants separate; seed averaging was not universally beneficial.

## What did not work well

- TE/WOE expansion for XGBoost/LightGBM.
- TabICLv2 on the tested representation.
- Standalone EBM and RealMLP as final classifiers.
- The v20 TabM 15% probability blend.
- The final v22 logistic stack and slice-aware gating experiment.

The v22 experiment used nested meta-model selection and a predeclared promotion gate. Its parent scored `0.639157` nested F1, while the pure stack scored `0.636470`; the best parent/stack mixture scored `0.637378` and improved only 2/5 folds. No v22 Kaggle submission was made.

## Repository layout

```text
configs/                  experiment configurations
src/                      training, validation, HPO and ensemble pipelines
scripts/                  Windows runners, checks and reproducibility helpers
notebooks/                reproducible notebook entry points
research/                 experiment planning and model-family research notes
tests/                    baseline metric/submission tests
CHAMPION.json             final frozen champion metadata
requirements-v2.txt       core reproducibility environment
requirements-modern.txt   later XGB/LGBM/modern-tabular packages
```

Large or competition-sensitive files are intentionally excluded from Git:

- raw Kaggle data;
- trained models and OOF arrays;
- submission CSVs;
- local Kaggle/account logs;
- virtual environments;
- private working notes and session handoff files.

## Setup

Python 3.12 was used for most of the campaign.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-v2.txt
pip install -r requirements-modern.txt
```

PyTorch/CUDA should be installed separately for the local CUDA version before running TabM/RealMLP experiments. AutoGluon and TabICL were tested in separate environments because of dependency conflicts.

Download the competition data from Kaggle into `data/raw/`. The repository does not redistribute competition data.

## Reproducibility notes

The project contains the complete experiment code, configs, frozen champion metadata, and the major research decisions. Binary model artifacts and raw OOF/test arrays are not committed.

For a lightweight pipeline sanity check:

```powershell
python -m unittest discover -s tests -v
python src\v2_pipeline.py verify
```

Later experiments depend on artifacts produced by earlier stages. The main lineage is approximately:

```text
baseline/v2
  -> v3 XGBoost
  -> v5/v12 AutoGluon full stack
  -> v17/v18 XGBoost HPO/finalization
  -> v19 LightGBM
  -> v20 TabM
  -> v21 EBM + RealMLP
  -> v22 final stack/gating audit
```

See [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md) for the final campaign summary and interpretation.

## Data and competition policy

No raw competition files, credentials, Kaggle API tokens, or model binaries are included in this repository. Users must obtain the data through Kaggle and comply with the competition's terms.

## License

No open-source license has been selected yet. Until a license is added, the repository remains copyrighted by its owner with no implied redistribution rights.

