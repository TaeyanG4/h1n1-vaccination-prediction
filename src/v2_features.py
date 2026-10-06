"""Deterministic row-local features and fold-fitted categorical schemas for v2."""
from __future__ import annotations
import numpy as np
import pandas as pd

NOMINAL = {'education_comp', 'raceeth4_i', 'sex_i', 'inc_pov', 'marital', 'rent_own_r', 'census_region', 'hhs_region'}
EFFECTIVE = {'Not At All Effective': 1, 'Not Very Effective': 2, 'Somewhat Effective': 3, 'Very Effective': 4}
RISK = {'Very Low': 1, 'Somewhat Low': 2, 'Moderate': 3, 'Somewhat High': 4, 'Very High': 5}
WORRY = {'Not At All Worried': 1, 'Not Very Worried': 2, 'Somewhat Worried': 3, 'Very Worried': 4}

def build_features(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    """No target, fitting, population aggregates, or train/test concatenation."""
    if {'vacc_h1n1_f', 'vacc_seas_f'} & set(raw.columns):
        raise ValueError('Target information must not enter features')
    if mode not in {'raw', 'survey'}:
        raise ValueError(f'Unknown feature mode: {mode}')
    x = raw.copy()
    if mode == 'raw':
        return x
    opinions = [c for c in raw if c.startswith('opinion_')]
    behaviors = [c for c in raw if c.startswith('behavioral_')]
    x['v2_missing_total'] = raw.isna().sum(axis=1).astype(float)
    for name, cols in [('opinion', opinions), ('behavior', behaviors)]:
        x[f'v2_{name}_missing'] = raw[cols].isna().sum(axis=1).astype(float)
    x['v2_unknown_answers'] = raw[opinions].isin(['Dont Know', "Don't Know", 'Refused']).sum(axis=1).astype(float)
    for col in ['health_insurance', 'employment_industry', 'education_comp', 'h1n1_concern', 'child_under_6_months']:
        x[f'v2_missing_{col}'] = raw[col].isna().astype(float)
    protective = [c for c in behaviors if c != 'behavioral_antiviral_meds']
    x['v2_protective_sum'] = raw[protective].sum(axis=1, min_count=1)
    x['v2_protective_mean'] = raw[protective].mean(axis=1)
    x['v2_doctor_sum'] = raw['doctor_recc_h1n1'] + raw['doctor_recc_seasonal']
    x['v2_doctor_difference'] = raw['doctor_recc_h1n1'] - raw['doctor_recc_seasonal']
    for season in ['h1n1', 'seas']:
        for suffix, mapping in [('vacc_effective', EFFECTIVE), ('risk', RISK), ('sick_from_vacc', WORRY)]:
            x[f'v2_{season}_{suffix}_ordinal'] = raw[f'opinion_{season}_{suffix}'].map(mapping).astype(float)
        x[f'v2_{season}_benefit'] = x[f'v2_{season}_vacc_effective_ordinal'] * x[f'v2_{season}_risk_ordinal']
    for suffix in ['vacc_effective', 'risk', 'sick_from_vacc']:
        x[f'v2_opinion_delta_{suffix}'] = x[f'v2_h1n1_{suffix}_ordinal'] - x[f'v2_seas_{suffix}_ordinal']
    adults = pd.to_numeric(raw['n_adult_r'], errors='raise')
    children = pd.to_numeric(raw['household_children'], errors='raise')
    x['v2_children_fraction'] = children / (adults + children).replace(0, np.nan)
    x['v2_any_children'] = children.gt(0).astype(float).where(children.notna())
    return x

def categorical_columns(x: pd.DataFrame) -> list[str]:
    return [c for c in x if x[c].dtype == 'object' or c in NOMINAL]

def fit_schema(x: pd.DataFrame, family: str) -> dict:
    if family not in {'catboost', 'lightgbm'}:
        raise ValueError(family)
    cat = categorical_columns(x)
    levels = {c: sorted(x[c].dropna().astype(str).unique().tolist()) for c in cat} if family == 'lightgbm' else {}
    return {'family': family, 'columns': list(x.columns), 'categorical': cat, 'levels': levels}

def transform(x: pd.DataFrame, schema: dict) -> pd.DataFrame:
    if list(x.columns) != schema['columns']:
        raise ValueError('Feature schema/order mismatch')
    result = x.copy()
    cat = set(schema['categorical'])
    for col in result:
        if col not in cat:
            result[col] = pd.to_numeric(result[col], errors='raise').astype(float)
        elif schema['family'] == 'catboost':
            result[col] = result[col].where(result[col].notna(), '__MISSING__').astype(str)
        else:
            # Categories are learned from this model's training rows only.
            # Unseen and missing categories become NaN, handled natively by LightGBM.
            values = result[col].where(result[col].notna(), np.nan)
            values = values.map(lambda v: str(v) if pd.notna(v) else np.nan)
            result[col] = pd.Categorical(values, categories=schema['levels'][col])
    return result
