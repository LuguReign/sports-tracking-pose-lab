import unittest
import numpy as np
from src.data import samples
class FeatureTests(unittest.TestCase):
 def game(self):
  t=np.arange(150)/25
  pos=np.stack([np.c_[20+t,20+t*0],np.c_[30+t,30+t*0]],axis=1)
  return dict(pos=pos,ball=np.zeros((150,2)),period=np.ones(150,int),frame=np.arange(150),names=['a','b'],teams=np.array(['A','B']))
 def test_velocity_and_targets(self):
  x,y,m=samples(self.game(),25)
  np.testing.assert_allclose(x[:,2],1);np.testing.assert_allclose(y[:,0],1)
  self.assertEqual(x.shape[1],38)
 def test_future_cannot_change_features(self):
  a=self.game();x,y,m=samples(a,25);a['pos'][50]+=10
  xx,yy,mm=samples(a,25)
  np.testing.assert_allclose(x[m[:,0]==25],xx[mm[:,0]==25]);self.assertFalse(np.allclose(y[0],yy[0]))
 def test_no_period_boundary_crossing(self):
  a=self.game();a['period'][60:]=2;x,y,m=samples(a,25)
  self.assertTrue(all(a['period'][int(t)-25]==a['period'][int(t)+25] for t in m[:,0]))
 def test_player_dropout_excluded(self):
  a=self.game();a['pos'][25,0]=np.nan;x,y,m=samples(a,25)
  self.assertFalse(any((m[:,0]==25)&(m[:,1]==0)))
if __name__=='__main__':unittest.main()

class BlackoutTests(unittest.TestCase):
 def test_paired_eligibility_and_linear_prediction(self):
  from src.blackout import evaluate_blackout
  g=FeatureTests().game();g['pos'][45,0]=np.nan
  result=evaluate_blackout(g)
  self.assertEqual(result.n.nunique(),1)
  np.testing.assert_allclose(result.mean_error_m,0,atol=1e-12)

class EPTSParserTests(unittest.TestCase):
 def test_changing_channels_preserve_player_identity(self):
  import tempfile
  from pathlib import Path
  from src.data import read_game
  with tempfile.TemporaryDirectory() as temp:
   folder=Path(temp)/'data/Sample_Game_3';folder.mkdir(parents=True)
   xml='''<main><GlobalConfig><ProviderGlobalParameters><ProviderParameter><Name>first_half_end</Name><Value>1</Value></ProviderParameter></ProviderGlobalParameters></GlobalConfig><Players><Player id="A" teamId="H"/><Player id="B" teamId="V"/></Players><PlayerChannels><PlayerChannel id="a_x" playerId="A"/><PlayerChannel id="a_y" playerId="A"/><PlayerChannel id="b_x" playerId="B"/><PlayerChannel id="b_y" playerId="B"/></PlayerChannels><DataFormatSpecification startFrame="1" endFrame="1"><PlayerChannelRef playerChannelId="a_x"/><PlayerChannelRef playerChannelId="a_y"/><PlayerChannelRef playerChannelId="b_x"/><PlayerChannelRef playerChannelId="b_y"/></DataFormatSpecification><DataFormatSpecification startFrame="2" endFrame="2"><PlayerChannelRef playerChannelId="b_x"/><PlayerChannelRef playerChannelId="b_y"/><PlayerChannelRef playerChannelId="a_x"/><PlayerChannelRef playerChannelId="a_y"/></DataFormatSpecification></main>'''
   (folder/'Sample_Game_3_metadata.xml').write_text(xml)
   (folder/'Sample_Game_3_tracking.txt').write_text('1:0.1,0.2;0.7,0.8:0.5,0.5\n2:0.71,0.81;0.11,0.21:0.5,0.5\n')
   g=read_game(temp,3)
   np.testing.assert_allclose(g['pos'][:,0,0],[10.5,11.55])
   np.testing.assert_allclose(g['pos'][:,1,0],[73.5,74.55])
   self.assertEqual(g['period'].tolist(),[1,2])
