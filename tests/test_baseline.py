from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline import check_submission, select_threshold, metrics, prepare_features

class BaselineTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(metrics(np.array([0, 0, 1, 1]), np.array([.01, .02, .98, .99]), .5)['f1'], 1.0)
    def test_threshold_helps(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.01, .10, .35, .40])
        threshold, score = select_threshold(y, p)
        self.assertEqual(score, 1.0)
        self.assertLess(threshold, .4)
    def test_valid_binary(self):
        sample = pd.DataFrame({'Id': [0, 1], 'vacc_h1n1_f': [0, 0]})
        self.assertTrue(check_submission(sample, sample)['passed'])
    def test_id_reorder_rejected(self):
        sample = pd.DataFrame({'Id': [0, 1], 'vacc_h1n1_f': [0, 0]})
        with self.assertRaises(ValueError): check_submission(sample, sample.iloc[::-1])
    def test_probability_rejected(self):
        sample = pd.DataFrame({'Id': [0, 1], 'vacc_h1n1_f': [0, 0]})
        bad = sample.copy(); bad['vacc_h1n1_f'] = [.2, .8]
        with self.assertRaises(ValueError): check_submission(sample, bad)
    def test_nan_rejected(self):
        sample = pd.DataFrame({'Id': [0, 1], 'vacc_h1n1_f': [0, 0]})
        bad = sample.copy(); bad['vacc_h1n1_f'] = [np.nan, 0]
        with self.assertRaises(ValueError): check_submission(sample, bad)
    def test_missing_category(self):
        frame = pd.DataFrame({'cat': ['a', None], 'num': [1, np.nan]})
        self.assertEqual(prepare_features(frame, ['cat']).loc[1, 'cat'], '__MISSING__')

if __name__ == '__main__': unittest.main()
