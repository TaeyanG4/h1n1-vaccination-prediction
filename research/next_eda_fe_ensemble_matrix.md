# Next experiment matrix: recovered 2023 ideas x current strongest pipeline

## Guiding principle

The 2023 notebook contributes three valuable mechanisms: ordinal representation,
compact row-level feature engineering, and strong class-weight / sparse-column
XGBoost. The current campaign contributes better validation, exact-duplicate
grouping, explicit missingness analysis, full-label refit, and a strong v12
44-base stack. The next experiments should combine mechanisms, not copy the old
validation scheme.

## EDA that can directly generate model hypotheses

### E1. v12 error slices vs historical-XGB error slices
- On the same development rows, compare FP/FN sets for v12 OOF and the best
  historical/legacy XGB OOF.
- Slice disagreements by doctor recommendation, H1N1 opinion/risk, age,
  health-worker status, employment structural missingness, state/industry/
  occupation, and missingness pattern.
- Goal: identify features where XGB reliably rescues v12, or vice versa. These
  slices should drive ensemble/cascade logic rather than blind averaging.

### E2. Target-rate tables for old engineered blocks
- doctor_recc_h1n1 x opinion_h1n1_risk
- doctor_recc_h1n1 x opinion_h1n1_vacc_effective
- h1n1_concern x h1n1_knowledge
- health_worker x doctor_recc_h1n1
- agegrp x chronic_med_condition
- employment_status x occupation/industry missingness
- opinion/refused/dont-know counts x target
- Evaluate support count + target rate + v12/XGB error rate per cell.

### E3. High-cardinality EDA without premature target encoding
- For state, employment_industry, employment_occupation, hhs_region:
  frequency, target-rate shrinkage, missingness, v12 error rate, historical-XGB
  error rate.
- Use this to decide whether frequency/grouping features are justified; do not
  feed raw target rates directly unless encoded fold-locally.

### E4. Missing-pattern and special-code analysis
- Preserve the old notebook's refusal/dont-know/special-code awareness and the
  current structural-missing findings.
- Analyze joint patterns rather than one-column missing flags only: employment
  block, insurance/health block, doctor recommendation block, H1N1 opinion
  block, seasonal opinion block.

### E5. Duplicate/profile analysis as a separate finalization hypothesis
- Quantify exact train-test profile overlaps that have unanimous vs conflicting
  train labels.
- Test a conservative final-only probability blend for unanimous overlap
  profiles; do not contaminate grouped CV by splitting identical profiles.
- Treat this as a special transductive hypothesis because ordinary grouped CV
  cannot estimate the benefit on profiles deliberately kept out of training.

## Feature-engineering lanes

### F1. Historical FE + modern structural missingness (high priority)
Base: recovered 40-feature historical representation.
Add only ~8-12 compact modern missing-pattern features:
- employment pair/status missing
- doctor any/both missing
- H1N1 opinion all-missing
- seasonal opinion all-missing
- demographic missing count
- health missing count
- refusal/dont-know counts
Rationale: historical FE already gives +~0.005 nested F1 vs raw legacy;
missingness is independently predictive and was not explicitly represented in
the old pipeline.

### F2. Historical FE + selected v2 survey features (high priority)
Do not append all survey64 blindly. Add only mechanisms not already duplicated:
- doctor recommendation sum/difference
- H1N1 vs seasonal opinion deltas
- effectiveness x risk
- children fraction / any-child
- compact behavior mean/sum if not equivalent
Goal: ~45-50 features, not another v8-style feature explosion.

### F3. Old ordinal core + dual representation (high priority)
Keep each informative ordinal variable twice:
- original categorical/string representation for native XGB/CatBoost
- explicit semantic ordinal numeric version
Especially H1N1 opinion, seasonal opinion, age, census MSA.
Rationale: trees can choose between nominal and ordered interpretations.

### F4. Explicit seasonal analogue block (medium-high)
The old notebook explicitly ordinal-mapped only H1N1 opinion variables while
seasonal opinion stayed categorical. Add semantic seasonal maps and cross-vaccine
deltas/ratios. This is a plausible missing piece because seasonal attitudes are
strongly related to H1N1 vaccine behavior without using the forbidden seasonal
target.

### F5. Frequency encoding only where it complements ordinal coding (medium)
Add target-free frequency for state/industry/occupation/hhs_region to the
legacy/historical ordinal representation. Test strict fold-train frequency and
train+test target-free frequency as separate ablations.
Do not assume v14's standalone failure transfers to XGB ordinal representation.

### F6. Compact interaction block (medium)
Only a few interpretable interactions, selected from E2:
- doctor_recc_h1n1 x risk
- doctor_recc_h1n1 x effective
- concern x knowledge
- health_worker x doctor_recc_h1n1
- age x chronic
- H1N1 effective x risk
Reject broad polynomial/cross-product expansion; v8 already showed that broad
interaction growth can hurt.

### F7. Duplicate-group weighting (exploratory)
Test sample_weight = 1 / exact-profile group-size so repeated survey profiles do
not dominate tree fitting. Preserve labels including conflicting groups.
Compare against ordinary weighting on identical grouped folds.

## XGBoost / runtime lanes

### X1. Current XGB 3.3 Optuna
Continue v17 with four representations: raw native, survey native,
legacy_ordinal, historical. Wide class-weight and column-sampling space.

### X2. Two-stage Optuna around surviving historical regions
After broad v17, launch a narrower second study centered on the top 10 trials,
with repeated/five-fold confirmation as the selection surface. Search fewer
parameters more densely rather than endlessly expanding the broad study.

## Ensemble ladder (ensemble first, optimized blend later)

### A1. v12 exact44 + best historical XGB, 50/50 probability
First diversity test. v12 is current private leader; historical XGB has a
different representation and class boundary.

### A2. v12 + historical XGB, 50/50 rank average
High priority because strong class weighting shifts XGB calibration. Rank
averaging tests diversity without requiring comparable probability scales.

### A3. Conservative 75/25 and 80/20 v12:XGB
If equal averaging loses precision, preserve v12 as anchor and use XGB only as
diversity. Compare nested threshold stability, not just tuned full-OOF F1.

### A4. Seed-ensemble XGB then combine with v12
Average five full-refit XGB seeds first; only then combine with v12. This
separates seed variance reduction from cross-family diversity.

### A5. Boundary-rescue cascade
Use v12 as primary classifier. Only for rows in a narrow uncertainty band around
the v12 threshold, allow best historical XGB to decide/rescue. Outside the band,
keep v12 decision unchanged.
This is attractive if E1 shows XGB's advantage is concentrated near the v12
boundary.

### A6. Confidence rescue instead of global average
- Positive if v12 is positive, OR XGB is extremely confident positive.
- Optional symmetric negative rescue if v12 is barely positive but XGB is very
  confident negative.
Tune only a very small predeclared grid under nested folds.

### A7. v12 + XGB + one truly different weak model
Only if error-overlap analysis justifies it: TabICL/RealMLP/another tabular
family can receive 5-15% rank weight. Do not add correlated CatBoost variants
just to increase model count; v16 showed correlated equal averaging can hurt.

## Blend / stack phase after simple ensemble evidence

### B1. Coarse constrained OOF weight scan
Nonnegative weights summing to one, coarse increments (e.g. .05/.10), nested
weight selection. Start with v12 + best XGB only; add a third family only if it
adds error diversity.

### B2. Fold-local calibration then blend
For class-weighted XGB, compare Platt/isotonic calibration fitted within outer
folds, then average with v12. This may outperform raw probability blending even
when rank average is strong.

### B3. Tiny meta-model stack
Inputs: only OOF probabilities/ranks from v12, XGB historical, optionally one
third diverse family. Meta: logistic regression or shallow CatBoost. No large
feature set. Full test meta-features must come from full-refit base models.

## Prioritized execution order

1. Finish/resume v17 broad Optuna.
2. Run E1 disagreement/error-slice analysis using v12 and v17 top historical XGB.
3. F1 historical + structural missingness.
4. F2 historical + selected v2 survey block.
5. Simple ensembles A1/A2/A3/A5.
6. F4 seasonal analogue and F5 frequency supplements.
7. Narrow second-stage Optuna around surviving historical regions.
8. Only after simple ensembles survive: B1/B2/B3 optimized blending/stacking.

Every serious final candidate follows the current policy: recipe/threshold
selected from leakage-safe grouped CV, then all 42,154 labels full-refit before
submission generation.
