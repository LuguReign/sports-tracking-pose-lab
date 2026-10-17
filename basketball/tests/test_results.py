from pathlib import Path
import unittest,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
class ResultsTests(unittest.TestCase):
 def test_source_accounting_and_primary_session(self):
  a=json.loads((R/'output/audit.json').read_text());self.assertEqual(a['source_trials'],583);self.assertEqual(a['eligible_trials']+len(a['excluded_trials']),583)
  f=pd.read_csv(R/'output/trial_features.csv');self.assertFalse(f.duplicated(['session','participant','trial']).any());self.assertEqual(f.participant.nunique(),5)
 def test_later_shots_held_out(self):
  f=pd.read_csv(R/'output/trial_features.csv');f=f[f.session=='2025-12-18'];p=pd.read_csv(R/'output/outcome_predictions.csv');p=p[(p.design=='Chronological 70/30')&(p.model=='Pose logistic')]
  for person,g in f.groupby('participant'):
   order=g.sort_values('trial_order');expected=set(order.iloc[int(np.floor(.7*len(order))):].trial)
   self.assertEqual(expected,set(p[p.participant==person].trial))
 def test_whole_athlete_folds(self):
  p=pd.read_csv(R/'output/outcome_predictions.csv');p=p[p.design=='Leave one athlete out'];self.assertTrue((p.fold==p.participant).all())
  for model,g in p.groupby('model'):self.assertFalse(g.duplicated(['participant','trial']).any())
 def test_models_and_intervals_are_valid(self):
  v=pd.read_csv(R/'output/variance_components.csv');self.assertTrue(v.converged.all());self.assertTrue(v.conditional_icc.between(0,1).all())
  p=pd.read_csv(R/'output/outcome_predictions.csv');self.assertTrue(p.probability.between(0,1).all());self.assertEqual(p.made.nunique(),2)
