"""Executed-output checks: split provenance, benchmark fairness, interval bounds."""
import unittest,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class ResultsTests(unittest.TestCase):
 def test_split_and_metrics(self):
  a=json.loads((ROOT/'output/manifest.json').read_text())
  self.assertEqual(a['split'],{'train':1,'calibration':2,'test':3})
  m=pd.read_csv(ROOT/'output/metrics.csv')
  self.assertEqual(len(m),10)
  self.assertTrue(m.groupby('horizon_s').n_test.nunique().eq(1).all())
  self.assertTrue(m.test_coverage.between(0,1).all())
  self.assertTrue(m.mean_error_m.ge(0).all())
