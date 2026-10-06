# Public notebook / related-competition intelligence — 2026-10-05

## Goal
Collect transferable ideas for the current positive-class F1 H1N1 competition without copying weak validation or tuning against the public leaderboard.

## Highest-value sources

### 1. CG DSA ML Cup 2026 — near-exact F1 analogue
Source: https://www.kaggle.com/competitions/cg-dsa-ml-cup-2026

- Same NHFS/H1N1-style respondent features and H1N1 binary target.
- Submission is hard 0/1 labels and metric is positive-class F1.
- Official guidance explicitly says probabilities can be trained internally and the binary threshold should be chosen on the participant's own validation data.
- Final visible leaderboard top score was about 0.60535. This is useful only as evidence that threshold/imbalance handling matter; its split is different from the current competition, so the score is not a target for us.

Transfer decision: HIGH for metric/threshold lessons; LOW for absolute score comparison.

### 2. Direct current-competition notebook — Seungtae Moon
Source: https://www.kaggle.com/code/conanmoon/predicting-vaccinated-individuals-seungtae-moon

- Uses target name `vacc_h1n1_f` and current competition file paths.
- Adds a total behavioral count.
- Removes categorical features with cardinality over 30, including `state`.
- Author reports the resulting submission scored only about 0.33 and explicitly notes that an earlier model retaining all/high-cardinality features performed better.

Transfer decision: HIGH negative evidence. Do not drop `state`, occupation, industry, or other high-cardinality survey fields merely because of cardinality. Current v2 correctly retains them.

### 3. DrivenData Flu Shot Learning — original NHFS problem
Source: https://www.drivendata.org/competitions/66/flu-shot-learning/

- Same National 2009 H1N1 Flu Survey source and nearly identical feature semantics.
- Original task predicts H1N1 + seasonal probabilities and scores mean ROC AUC, so ranking/feature/model lessons transfer but hard-label threshold conclusions do not.

Transfer decision: HIGH for representation/model discovery; MEDIUM/LOW for post-processing because current metric is F1.

### 4. Prasad Huddar public solution — author reports #48 / 8,080
Source: https://github.com/Prasad-Huddar/Flu-Shot

Author-reported design:
- 15-fold stratified OOF.
- Two LightGBM + two XGBoost + two CatBoost variants.
- Isotonic probability calibration.
- Fold-local target encoding and WOE encoding.
- Differential-evolution optimized nonnegative model weights.
- H1N1 stored OOF AUC 0.87301, log loss 0.33863.

Feature block in source:
- `doctor_recc_both`, `doctor_recc_sum`.
- H1N1 risk/sick-from-vaccine ratio and net opinion.
- seasonal equivalents.
- cross-vaccine effectiveness product.
- total/mean/variance of opinion fields.
- total/protective/risky behavioral counts.
- ordinal age + senior/young flags.
- ordinal education + college flag.
- income ordinal.
- health-risk count.
- doctor × age, doctor × chronic-condition interactions.
- knowledge × behavior interaction.

Encoding block:
- category labels, plus fold-train TargetEncoder(smoothing=1.0), plus WOEEncoder.

Important reproducibility / validation cautions:
- Current notebook source defines `lgb`, `xgb`, `cat` but later refers to `lgb1/lgb2/xgb1/xgb2/cat1/cat2`; those six definitions are absent from the published source although outputs exist. Treat the exact six-model recipe as incompletely reproducible.
- Differential-evolution weights are fitted on the same validation fold whose AUC is reported. That makes the fold score optimistically selected. Do not copy this evaluation design.
- Isotonic calibration does not inherently improve AUC/ranking; for our F1 task its main potential value is threshold stability, which must be tested nested/leakage-safe.

Transfer decision: HIGH for feature + encoding hypotheses; MEDIUM for calibration; LOW for copied optimized blending protocol.

### 5. Darkknight98 — Flu Shot Prediction Complete EDA and HPO
Source: https://www.kaggle.com/code/darkknight98/flu-shot-prediction-complete-eda-and-hpo

Useful feature ideas:
- behavioral `cleanliness` sum.
- total opinion score.
- H1N1 opinion = effectiveness + risk - sick-from-vaccine.
- concern >= 2 flag, high-risk flag, high-knowledge flag.
- concern + knowledge interaction.
- age-squared representation.

The notebook also drops many variables and uses old random-split / accuracy-style comparisons in places, so claimed model comparisons are weaker than our frozen grouped CV.

Transfer decision: MEDIUM for feature hypotheses only.

### 6. DAC FIND IT 2023 — BukanRISTEK
Source: https://www.kaggle.com/code/edwardsalim/dac-findit-2023-bukanristek

Same NHFS source; original competition metric was ROC AUC.

Useful data/feature ideas:
- explicitly preserve missing/unknown meaning rather than blindly imputing.
- opinion missing can represent a "Don't know" category.
- employment occupation/industry missingness is structurally related to employment status.
- `good_behavioral_count` and `has_doc_recc`.
- mix ordinal encoding for ordered variables with one-hot for nominal variables.

Interesting CatBoost H1N1 parameter hypothesis reported by notebook:
- iterations 600
- learning_rate ~0.0286
- grow_policy Lossguide
- depth 6
- min_data_in_leaf 18
- l2_leaf_reg ~30.38
- max_bin 21
- bagging_temperature 2
- auto_class_weights Balanced

Caution: their custom CV calls ROC AUC on hard class predictions, so their reported HPO gains are not trustworthy evidence. Parameters are hypotheses only.

Transfer decision: MEDIUM for missingness/encoding and a cheap CatBoost configuration gate.

### 7. Other same-data public notebooks
- Eric Layer: imputation, MI/chi2 feature screening, RF/ET/SVM/XGB and multi-label techniques. Useful mainly as background; no stronger validation evidence than ours.
- Apache Naren: RFE, RF grid search, CatBoost/XGB experiments. Old random-split style; low evidence.
- Aysenur XGBoost: fits ordinal encoders separately to train and test, which can produce inconsistent category mappings. Do not copy.
- Abhishek Gupta: drops `doctor_recc_h1n1` because of correlation with seasonal recommendation. This is inappropriate for our H1N1-only target given doctor recommendation is a high-signal variable. Do not copy.

## What is already in our v2 survey features

Already covered:
- missing count and opinion/behavior missing count.
- explicit unknown-answer count.
- selected missing indicators.
- protective behavior sum/mean.
- doctor recommendation sum/difference.
- H1N1 and seasonal opinion ordinal maps.
- H1N1/seasonal effectiveness × risk benefit features.
- cross-vaccine opinion deltas.
- children fraction / any-children flag.

Therefore simply copying `behavioral sum`, `doctor sum`, or basic opinion ordinals is unlikely to add much.

## Truly new blocks worth testing

### Priority A — new row-local interaction feature block
Cheap and leakage-safe:
- H1N1 risk / sick-from-vaccine ratio.
- H1N1 net opinion = risk - sick-from-vaccine.
- overall opinion total / mean / variance.
- cross-vaccine effectiveness product.
- doctor recommendation product (`doctor_recc_both`).
- health-risk count across chronic condition / young child / health worker / insurance.
- age ordinal + senior/young flags.
- doctor-H1N1 × age.
- knowledge × behavior interaction.
- separate protective vs risky behavior totals.

Why first: these are not present in v2, cost almost nothing, preserve all high-cardinality raw fields, and several independent public solutions converge on opinion/doctor/behavior interactions.

### Priority B — leakage-safe supervised categorical encodings for XGB/LGBM
Use fold-train-only smoothed target encoding and WOE features for categorical columns, evaluated on the exact frozen grouped folds.

Why: AutoML already tested XGB/LGBM as families, but not necessarily with this representation. A representation change can make an already-tested family materially different and more competitive/diverse.

Guardrail: encoders must be fitted inside each training fold. Do not fit on all development labels before OOF scoring.

### Priority C — nested probability calibration / threshold stability
Test Platt and isotonic calibration on strong CatBoost/v5 probabilities under an outer-fold protocol, with threshold chosen only on inner/training folds.

Why: current metric is F1 and recent exact analogue explicitly requires validation-chosen thresholds. v7 showed tiny same-OOF threshold/HPO gains do not reliably transfer.

Guardrail: do not calibrate and score on the same held-out labels.

### Priority D — CatBoost structural parameter gate
Only a cheap gate for materially different tree geometry:
- Lossguide vs symmetric tree.
- high L2 regularization.
- min_data_in_leaf / max_bin.
- Balanced or SqrtBalanced class weights.

Why below A-C: v7 already showed ordinary local HPO gains can be selection noise. This should be a representation/geometry test, not another broad HPO sweep.

## Things not worth prioritizing
- Another generic AutoML sweep of CatBoost/LGBM/XGB/RF/ET/NN: already covered by v5.
- Blindly adding more equal-weight models: v5 leaderboard evidence showed dilution.
- Per-fold differential-evolution blend weights: optimistic if selected on the scored fold.
- Dropping `state`/employment high-cardinality variables: direct current-competition notebook reports this hurt badly.
- More fine HPO around the current CatBoost optimum without a representation change: v7 local +0.0011 did not improve leaderboard.
- Using `vacc_seas_f` as a feature: prohibited by our project contract and would be target leakage.

## Recommended experiment order
1. v8 row-local interaction feature block on the exact frozen folds, first with a strong simple CatBoost candidate and optionally v5-compatible scoring.
2. If v8 is neutral/positive, build leakage-safe TE+WOE XGB/LGBM candidates on the same folds and measure F1 + correlation/error overlap versus v5 CatBoost.
3. Test nested Platt/isotonic calibration and threshold stability on the strongest 1–2 models.
4. Only if these fail, run a cheap CatBoost Lossguide/Balanced structural gate or move to a genuinely new family (TabPFN/TabDPT/TabM/RealMLP).

## Historical-score connection
Old account submissions still outperform the new v5 scout on private leaderboard (`submission3.csv` 0.63475; XGB16/22 0.63421 vs v5 0.63268). No original source code for those historical files was found in the current workspace, and the workspace is not a Git repository, so their exact pipelines cannot currently be recovered from local history. The external findings make a representation hypothesis (especially categorical encoding / interaction features) more valuable than another generic model-family search.
