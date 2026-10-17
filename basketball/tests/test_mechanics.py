import unittest
import numpy as np
from src.mechanics import joint_angle,repair_short_gaps,extract,FEATURES
class MechanicsTests(unittest.TestCase):
 def test_angle_geometry_and_translation(self):
  a=np.array([[1.,0,0]]);b=np.array([[0.,0,0]]);c=np.array([[0.,1,0]])
  np.testing.assert_allclose(joint_angle(a,b,c),90)
  np.testing.assert_allclose(joint_angle(a+7,b+7,c+7),90)
 def test_degenerate_angle_is_missing(self):
  a=np.zeros((1,3));self.assertTrue(np.isnan(joint_angle(a,a,a)[0]))
 def test_only_short_internal_gaps_are_repaired(self):
  x=np.arange(12,dtype=float)[:,None];x[2:4]=np.nan;x[6:10]=np.nan;x[0]=np.nan
  y=repair_short_gaps(x);np.testing.assert_allclose(y[2:4,0],[2,3]);self.assertTrue(np.isnan(y[6:10]).all());self.assertTrue(np.isnan(y[0]).all())
 def test_outcome_variables_cannot_enter_feature_list(self):
  self.assertTrue(set(FEATURES).isdisjoint({'made','result','landing_x','landing_y','entry_angle','participant','session'}))
