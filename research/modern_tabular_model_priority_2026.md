# 2024-2026 modern tabular model priority for H1N1 competition

This roadmap uses the user-provided 2024-2026 model list as a hypothesis pool.
Model/version availability, licenses, checkpoints, row/context limits, and local
runtime requirements are not assumed; verify them immediately before an actual
experiment. Scientific selection remains based on the frozen grouped OOF
protocol, not model recency.

## Competition-specific geometry

- Binary F1 task, 42,154 labeled rows, 28,104 test rows.
- Only 38 raw columns; mixed low/high-cardinality categorical and numeric survey
  variables; heavy but structured missingness.
- Exact duplicate feature profiles exist and must remain grouped in validation.
- Current strongest family is tree/AutoML based: v12 exact44 stack plus v18 raw
  XGBoost. Current strongest private 2026 result is v18 v12/XGB rank60 single.
- Therefore a new family is useful primarily when it supplies complementary
  errors, not merely another highly correlated score around the same optimum.

## Tier A - highest-value next independent families

### 1. TabM

Why:
- Parameter-efficient MLP ensemble is structurally different from CatBoost/XGB.
- Medium-size 42k x ~40-80 feature geometry is appropriate for a serious neural
  baseline.
- Cheap enough to evaluate on exact grouped folds compared with large TFMs.

Experiment:
- raw/survey numeric+categorical encoding variants;
- 2-fold smoke -> grouped5 OOF if competitive;
- save OOF and measure correlation/error rescue vs v12 and v18 XGB;
- only full-refit if nested F1 or ensemble gain justifies it.

### 2. RealMLP

Why:
- Strong modern MLP baseline and very different inductive bias from GBDTs.
- Particularly valuable even if standalone F1 is slightly weaker when error
  correlation is lower.

Experiment:
- compact survey representation first;
- native/ordinal categorical preprocessing ablation;
- small seed ensemble only after one strong recipe is found.

### 3. ModernNCA

Why:
- Learned representation + nearest-neighbor behavior creates a highly different
  decision mechanism from trees and stacked boosting.
- Potentially useful on repeated/near-repeated survey response patterns.

Experiment:
- normalize numeric/ordinal features;
- compare raw semantic ordinal vs survey64 compact features;
- explicitly inspect performance on exact/near duplicate profiles and hard v12
  disagreement rows.

### 4. TabR

Why:
- Retrieval-augmented neural prediction could exploit local survey-profile
  neighborhoods and provide diversity.
- More setup/runtime than TabM/RealMLP, so run after those cheap gates.

Experiment:
- grouped validation must prevent duplicate leakage;
- neighbor retrieval must use training-fold rows only for validation rows;
- test full retrieval index is built from all 42,154 labeled rows only after the
  recipe is frozen.

## Tier A/B - cheap classical diversity controls

### ExtraTrees

- Very cheap and structurally different enough to be worth an OOF control.
- Use ordinal/frequency encoded categoricals; missing indicators preserved.
- Strong candidate for a 5-15% ensemble weight if correlation is low.

### EBM / Explainable Boosting Machine

- Additive/interpretable bias can capture stable monotonic-ish survey effects
  while avoiding tree-ensemble interaction complexity.
- Standalone score may be weaker, but disagreement quality can justify inclusion.

### HistGradientBoosting

- Cheap benchmark only; lower priority if LGBM already covers similar residuals.

## Tier B - Foundation / In-Context models worth checking after cheap neural lanes

The user-provided list includes TabPFN-3.5/3/2.6, LimiX-2, TabFM, Mitra-v2,
TabICLv2, EXAONE Tabular, Causilo, TabDPT-Turbo, and related variants.

Before running any of them verify:

1. public/local checkpoint or package actually available;
2. license permits competition use;
3. 42k-row context or approved subsampling/retrieval path;
4. mixed categorical/high-cardinality support;
5. inference time and VRAM fit RTX 4070 Ti SUPER 16GB;
6. deterministic or cacheable grouped-fold inference;
7. full-refit/full-context inference can be reproduced for test.

### TabPFN family

- High priority foundation-family check if a current locally usable version fits
  42k rows or supports retrieval/subsampling cleanly.
- Use as a distinct probability/rank source, not necessarily standalone champion.
- Do not tune public leaderboard against context/subsample choices.

### TabICLv2

- Already tested once in this project; OOF F1 was about 0.631 on the prior
  representation, so it is not an immediate rerun priority.
- Revisit only if a meaningfully different preprocessing/context strategy exists.

### LimiX / Mitra / TabDPT families

- Interesting because their ICL/retrieval priors differ from both trees and MLPs.
- Priority depends on actual accessible implementation and dataset-size support.
- A one-fold/2-fold falsification gate should precede any expensive full OOF run.

### TabFM / EXAONE Tabular / Causilo

- Treat as exploratory until local availability and reproducibility are verified.
- Keep them behind TabM/RealMLP/ModernNCA because infrastructure uncertainty is
  higher and we already have strong tree baselines.

## Tier C - lower fit for this specific single-table task

### Relational models: Relatron, SAP-RPT-1

- Low priority because the competition is a single flat table, not a relational
  multi-table database problem.

### CausalFM

- Low priority because the metric is predictive F1, not causal effect inference.

### Regression-centered models: Nori, Xiaomi-TabLDM variants

- Low priority unless a classification-capable implementation is clearly
  supported. Do not force a regression-centric model onto this binary task.

### Semantic/LLM-heavy models: TabSTAR, CARTE, TabuLa, FeatLLM, TabFM-Auto

- Interesting but lower ROI here because feature names/categories already have
  compact semantics and the dataset is small enough for strong GBDT/neural
  methods.
- More useful as feature-generation hypotheses than as first-line predictors.

### Mamba/Transformer/NAS families

- Mambular, MambaTab, ExcelFormer, AMFormer, pTNAS and similar models remain
  exploratory diversity candidates, behind stronger tabular-specific baselines.

## Ensemble policy for modern models

Do not build a large "model zoo average". The current evidence says correlated
equal averaging can hurt. Use greedy OOF admission:

1. anchor = v12 exact44 and v18 XGB;
2. for each new family, require valid grouped OOF and full inference artifact;
3. measure probability/rank correlation and hard-error rescue;
4. test small weights first (5%, 10%, 15%, 20%);
5. admit the family only if nested OOF improves or fold stability materially
   improves without a large score penalty;
6. compare probability and rank blending because neural/TFM calibration may
   differ strongly from tree models;
7. use a tiny logistic/meta stack only after 3+ genuinely diverse eligible
   families survive simple blending.

## Execution order after v19 LightGBM

1. TabM quick grouped-fold lane.
2. RealMLP quick grouped-fold lane.
3. ModernNCA quick grouped-fold lane.
4. ExtraTrees + EBM cheap diversity controls in parallel.
5. TabR if the first neural/retrieval results justify more setup.
6. Verify availability/license/context constraints for current TabPFN/LimiX/
   Mitra/TabDPT-family models and run only the best feasible 1-2 foundation
   families.
7. Greedy simple OOF ensemble against v12 + v18 XGB + best LGBM.
8. Only then consider optimized blend/stacking.

Every serious candidate still follows the same rule: grouped CV chooses recipe
and threshold; the threshold is frozen; all 42,154 labels are used for final
refit/context/index construction before submission generation.
