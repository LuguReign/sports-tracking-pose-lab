import numpy as np
import pandas as pd
def evaluate_blackout(g):
 gaps=[0,5,10,20,25];errors=[[] for _ in gaps]
 for t in range(50,len(g['pos'])-25,25):
  if g['period'][t-30]!=g['period'][t+25]:continue
  truth=g['pos'][t+25];predictions=[]
  for gap in gaps:
   a=t-gap;p=g['pos'][a];v=(p-g['pos'][a-5])/.2
   predictions.append(p+v*(1+gap/25))
  valid=np.isfinite(truth).all(1)
  for p in predictions:valid &= np.isfinite(p).all(1)
  for bucket,p in zip(errors,predictions):bucket.extend(np.linalg.norm(p[valid]-truth[valid],axis=1))
 return pd.DataFrame([dict(blackout_s=gap/25,n=len(e),mean_error_m=np.mean(e),median_error_m=np.median(e)) for gap,e in zip(gaps,errors)])
