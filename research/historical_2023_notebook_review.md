# Recovered 2023 H1N1 notebook review

Source: user-provided `H1N1_vaccination_2023_01_14_0_0_1.ipynb`.

## What the notebook clearly did

- Installed XGBoost 1.7.3 and `category_encoders`.
- Loaded train features, H1N1 labels, test, and sample submission.
- Added `istest` and concatenated train/test into `raw` for inspection / row-local engineering.
- Considered duplicate removal and dropping `rent_own_r > 2`, but those lines are commented out and therefore are not evidence that rows were actually removed.
- Implemented row-local engineering:
  - ordinal numeric mappings for the three H1N1 opinion variables;
  - ordinal age-group mapping;
  - ordinal census-MSA mapping;
  - recoded `rent_own_r` 77/99 to 1;
  - binarized employment status as Employed vs other;
  - added a doctor/chronic/child/health aggregate;
  - added an opinion mean;
  - added a behavioral sum;
  - dropped `census_region`.
- Inspected per-column cardinality (`raw.nunique`) and explicitly inspected the values of the high-cardinality employment industry and occupation columns.
- The active XGB preprocessing pipeline used `category_encoders.OrdinalEncoder`; TargetEncoder, OneHotEncoder, and SimpleImputer were present but commented out.
- The broad XGB grid used F1 scoring, 2-fold `GridSearchCV`, early stopping, class weighting around 2.57, low column-sampling values, depth 6/7, learning rate 0.01-0.03, gamma 0.5/1, and subsample 0.5/1.
- One recorded grid result reports validation AUC 0.8605 and F1 0.6482, with best grid parameters `colsample_bylevel=.5`, `colsample_bytree=.5`, `gamma=1`, `learning_rate=.03`, `max_depth=6`, `subsample=.5`.
- A later explicit model used 484 estimators, `scale_pos_weight=2.5714`, depth 6, learning rate .03, gamma 1, tree column sample .2, level column sample .9, subsample 1. Its recorded validation F1 is 0.64205.
- A commented alternate recipe used 338 estimators, class weight 4, depth 9, learning rate .1, tree sample .7 and level sample .5.
- The generated submission's positive share was 29.1524%, so the old workflow paid attention to the final class balance / decision boundary.

## Important ambiguities / weaknesses in the saved notebook

- The cell that defines `val` is absent, so the exact historical holdout split cannot be recovered from this file.
- `raw = engineer(raw)` is created, but later active model cells use `train[features]` / `val[features]`. Therefore the saved notebook does not prove that the engineered `raw` dataframe fed the strongest model.
- The feature-drop list is commented out, so there is no evidence that the long manual drop list was active.
- Duplicate removal and `rent_own_r` row deletion are commented out; do not import those actions as defaults.
- No active SimpleImputer is used. This supports treating missingness as signal rather than blindly filling every value.
- The old validation/GridSearch setup is much weaker than the current grouped OOF contract and can be optimistic. We should transfer representation / imbalance / search ideas, not the validation methodology.
- Parameters such as `criterion`, `max_features`, and `min_samples_leaf` in the old `XGBClassifier` call are sklearn-style names and are not the main XGBoost tree controls. Modern reproduction should search actual XGBoost controls such as `min_child_weight`, regularization, sampling, depth/leaves, and categorical handling.

## What is being absorbed into the modern pipeline

1. Keep the current exact-duplicate grouped validation; do not restore the old holdout protocol.
2. Add `legacy_ordinal`: original 38 features + fold-local `category_encoders.OrdinalEncoder`.
3. Keep `historical`: the recovered row-local feature-engineering hypothesis as a separate representation.
4. Widen Optuna search to include the old high `scale_pos_weight` and very low column-sampling regions.
5. Seed Optuna with the recorded old recipes, but let modern grouped OOF decide whether they survive.
6. Keep missing values / missing-state information; do not turn on blanket imputation.
7. Keep high-cardinality industry/occupation/state rather than dropping them by default.
8. Track predicted-positive rate alongside F1 because the old strong recipe intentionally moved the decision boundary through class weighting.
9. Final candidates still require all-42,154-label full refit and CV-frozen threshold.

## Modern reproduction probe

Using frozen grouped dev folds with XGBoost 3.3.0:

- `legacy_final_484` reproduces an OOF positive rate of ~29.08% at threshold .5, strikingly close to the notebook submission's 29.1524% positive rate. This is strong behavioral evidence that the old OrdinalEncoder + class-weight recipe has been reconstructed plausibly.
- With a corrected threshold grid (.15-.85), `legacy_final_484` reaches grouped nested F1 0.62997; the old grid-best style recipe reaches 0.62876; the depth-9/SPW4 alternate reaches 0.60297.
- The recovered row-local historical feature engineering is materially better than raw legacy ordinal encoding: `historical_fe_484` reaches grouped nested F1 0.63494 at threshold ~0.54 and audit F1 0.64599. This is about +0.005 nested F1 over the otherwise similar raw `legacy_final_484` recipe.
- The recovered recipes remain below the notebook's recorded holdout F1, reinforcing the decision to keep the old representation / imbalance ideas but not the old validation protocol.
- Initial v17 threshold grid (max .45) was invalid for high-class-weight recipes because .5 already outperformed the capped tuned threshold. The v17 grid is now .15-.85 in .005 increments.
