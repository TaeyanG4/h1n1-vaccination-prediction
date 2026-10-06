"""Leakage-free row-local interaction features discovered from public H1N1 solutions.

This module deliberately keeps all original/high-cardinality variables and layers new
numeric interactions on top of the proven v2 `survey` representation. It never uses
labels, population aggregates, train/test concatenation, or `vacc_seas_f`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2_features import build_features as build_v2_features


AGE_ORDINAL = {
    "6 Months - 9 Years": 0,
    "10 - 17 Years": 1,
    "18 - 34 Years": 2,
    "35 - 44 Years": 3,
    "45 - 54 Years": 4,
    "55 - 64 Years": 5,
    "65+ Years": 6,
}


def _opinion_block(x: pd.DataFrame) -> pd.DataFrame:
    h_eff = x["v2_h1n1_vacc_effective_ordinal"]
    h_risk = x["v2_h1n1_risk_ordinal"]
    h_sick = x["v2_h1n1_sick_from_vacc_ordinal"]
    s_eff = x["v2_seas_vacc_effective_ordinal"]
    s_risk = x["v2_seas_risk_ordinal"]
    s_sick = x["v2_seas_sick_from_vacc_ordinal"]

    x["v8_h1n1_risk_sick_ratio"] = (h_risk + 1.0) / (h_sick + 1.0)
    x["v8_h1n1_risk_minus_sick"] = h_risk - h_sick
    x["v8_h1n1_net_opinion"] = h_eff + h_risk - h_sick
    x["v8_seas_risk_sick_ratio"] = (s_risk + 1.0) / (s_sick + 1.0)
    x["v8_seas_risk_minus_sick"] = s_risk - s_sick
    x["v8_effectiveness_product"] = h_eff * s_eff

    opinion = pd.concat([h_eff, h_risk, h_sick, s_eff, s_risk, s_sick], axis=1)
    x["v8_opinion_total"] = opinion.sum(axis=1, min_count=1)
    x["v8_opinion_mean"] = opinion.mean(axis=1)
    x["v8_opinion_variance"] = opinion.var(axis=1, ddof=0)
    x["v8_h1n1_effective_high"] = h_eff.ge(4).astype(float).where(h_eff.notna())
    x["v8_h1n1_risk_high"] = h_risk.ge(4).astype(float).where(h_risk.notna())
    return x


def _context_block(raw: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    behaviors = [
        "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
        "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
        "behavioral_touch_face",
    ]
    protective = [
        "behavioral_antiviral_meds", "behavioral_avoidance",
        "behavioral_face_mask", "behavioral_wash_hands",
    ]
    exposure = ["behavioral_large_gatherings", "behavioral_outside_home", "behavioral_touch_face"]

    x["v8_doctor_both"] = raw["doctor_recc_h1n1"] * raw["doctor_recc_seasonal"]
    health_cols = ["chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance"]
    x["v8_health_context_sum"] = raw[health_cols].sum(axis=1, min_count=1)

    x["v8_behavior_total"] = raw[behaviors].sum(axis=1, min_count=1)
    x["v8_behavior_protective"] = raw[protective].sum(axis=1, min_count=1)
    x["v8_behavior_exposure"] = raw[exposure].sum(axis=1, min_count=1)
    x["v8_behavior_balance"] = x["v8_behavior_protective"] - x["v8_behavior_exposure"]

    age = raw["agegrp"].map(AGE_ORDINAL).astype(float)
    x["v8_age_ordinal"] = age
    x["v8_is_pediatric"] = raw["agegrp"].isin(["6 Months - 9 Years", "10 - 17 Years"]).astype(float)
    x["v8_is_18_34"] = raw["agegrp"].eq("18 - 34 Years").astype(float)
    x["v8_is_senior"] = raw["agegrp"].eq("65+ Years").astype(float)
    x["v8_age_squared"] = age.pow(2)

    x["v8_doctor_h1n1_x_age"] = raw["doctor_recc_h1n1"] * age
    x["v8_doctor_h1n1_x_chronic"] = raw["doctor_recc_h1n1"] * raw["chronic_med_condition"]
    x["v8_knowledge_x_behavior"] = raw["h1n1_knowledge"] * x["v8_behavior_total"]
    x["v8_concern_plus_knowledge"] = raw["h1n1_concern"] + raw["h1n1_knowledge"]
    x["v8_concern_x_knowledge"] = raw["h1n1_concern"] * raw["h1n1_knowledge"]
    x["v8_concern_high"] = raw["h1n1_concern"].ge(2).astype(float).where(raw["h1n1_concern"].notna())
    x["v8_knowledge_high"] = raw["h1n1_knowledge"].ge(2).astype(float).where(raw["h1n1_knowledge"].notna())
    return x


def build_features(raw: pd.DataFrame, block: str) -> pd.DataFrame:
    if {"vacc_h1n1_f", "vacc_seas_f"} & set(raw.columns):
        raise ValueError("Target information must not enter features")
    if block not in {"opinion", "context", "all"}:
        raise ValueError(f"Unknown v8 block: {block}")
    x = build_v2_features(raw, "survey")
    if block in {"opinion", "all"}:
        x = _opinion_block(x)
    if block in {"context", "all"}:
        x = _context_block(raw, x)
    return x
