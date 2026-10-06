from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v2_features import build_features, fit_schema, transform
from v2_pipeline import bootstrap_delta

class V2Tests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'health_insurance': [1., np.nan], 'employment_industry': ['A', None], 'education_comp': [1., np.nan], 'h1n1_concern': [2., np.nan], 'child_under_6_months': [0., np.nan], 'behavioral_wash_hands': [1., np.nan], 'behavioral_antiviral_meds': [0., np.nan], 'doctor_recc_h1n1': [1., 0.], 'doctor_recc_seasonal': [0., 1.], 'opinion_h1n1_vacc_effective': ['Very Effective', 'Dont Know'], 'opinion_seas_vacc_effective': ['Somewhat Effective', None], 'opinion_h1n1_risk': ['Very High', 'Dont Know'], 'opinion_seas_risk': ['Very Low', None], 'opinion_h1n1_sick_from_vacc': ['Not At All Worried', None], 'opinion_seas_sick_from_vacc': ['Very Worried', None], 'n_adult_r': [1., 0.], 'household_children': [1., 0.]})
    def test_raw_not_mutated(self):
        x = self.frame(); original = x.copy(deep=True)
        build_features(x, 'survey')
        pd.testing.assert_frame_equal(x, original)
    def test_row_local(self):
        x = self.frame()
        pd.testing.assert_frame_equal(build_features(x, 'survey').iloc[[0]], build_features(x.iloc[[0]], 'survey'))
    def test_target_rejected(self):
        x = self.frame(); x['vacc_seas_f'] = 1
        with self.assertRaises(ValueError): build_features(x, 'survey')
    def test_missing_not_zero(self):
        x = build_features(self.frame(), 'survey')
        self.assertTrue(np.isnan(x.loc[1, 'v2_protective_sum']))
        self.assertTrue(np.isnan(x.loc[1, 'v2_children_fraction']))
    def test_ordinal_and_benefit(self):
        x = build_features(self.frame(), 'survey')
        self.assertEqual(x.loc[0, 'v2_h1n1_benefit'], 20)
        self.assertTrue(np.isnan(x.loc[1, 'v2_h1n1_vacc_effective_ordinal']))
    def test_unknown_category_not_fitted(self):
        x = pd.DataFrame({'cat': ['A', 'B'], 'number': [1., 2.]})
        schema = fit_schema(x.iloc[[0]], 'lightgbm')
        transformed = transform(x, schema)
        self.assertTrue(pd.isna(transformed.loc[1, 'cat']))
        self.assertEqual(schema['levels']['cat'], ['A'])
    def test_catboost_missing_string(self):
        x = pd.DataFrame({'cat': ['A', None], 'number': [1., np.nan]})
        self.assertEqual(transform(x, fit_schema(x, 'catboost')).loc[1, 'cat'], '__MISSING__')
    def test_schema_order_rejected(self):
        x = self.frame(); schema = fit_schema(x, 'catboost')
        with self.assertRaises(ValueError): transform(x[x.columns[::-1]], schema)
    def test_bootstrap_identical_zero(self):
        y = np.array([0, 1, 0, 1]); p = np.array([False, True, True, False])
        r = bootstrap_delta(y, p, p, np.array(['a', 'b', 'c', 'd']), 30, 42)
        self.assertEqual(r['delta_95pct_interval'], [0., 0.])
    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError): build_features(self.frame(), 'unsupported')

if __name__ == '__main__': unittest.main()
