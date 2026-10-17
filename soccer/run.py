from pathlib import Path
import sys,json,hashlib,subprocess,platform
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.data import read_game,samples
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
source=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'data/metrica'
if not source.exists() and (ROOT/'data/metrica-source.zip').exists():
 import zipfile
 with zipfile.ZipFile(ROOT/'data/metrica-source.zip') as archive: archive.extractall(source)
games={i:read_game(source,i) for i in [1,2,3]}
audit={str(i):{'frames':len(g['frame']),'player_ids':len(g['names']),'periods':np.unique(g['period']).tolist(),'missing_player_coordinate_fraction':float(np.isnan(g['pos']).mean()),'missing_ball_fraction':float(np.isnan(g['ball']).mean())} for i,g in games.items()}
metrics=[];predrows=[];blockrows=[];fitted={};intervals={};rng=np.random.default_rng(20261017)
for seconds in [1,2]:
 data={i:samples(g,25*seconds) for i,g in games.items()};X,y,m=data[1];xc,yc,mc=data[2];xt,yt,mt=data[3]
 audit[f'{seconds}s_samples']={str(i):len(v[0]) for i,v in data.items()}
 models={'Stationary':None,'Constant velocity':None,'Ridge history':make_pipeline(StandardScaler(),Ridge(alpha=100.)), 'Boosted trees history':MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=7)), 'Boosted trees interactions':MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=7))}
 for name,model in models.items():
  columns=38 if name.endswith('interactions') else 10
  if model is not None:
   model.fit(X[:,:columns],y);pc=model.predict(xc[:,:columns]);pt=model.predict(xt[:,:columns])
  elif name=='Stationary':pc=np.zeros_like(yc);pt=np.zeros_like(yt)
  else:pc=xc[:,2:4]*seconds;pt=xt[:,2:4]*seconds
  ec=np.linalg.norm(pc-yc,axis=1);et=np.linalg.norm(pt-yt,axis=1)
  q=float(np.quantile(ec,.9,method='higher'));intervals[(seconds,name)]=q
  metrics.append(dict(horizon_s=seconds,model=name,n_test=len(et),mean_error_m=et.mean(),median_error_m=np.median(et),p90_error_m=np.quantile(et,.9),calibration_radius_m=q,test_coverage=np.mean(et<=q)))
  if seconds==1:fitted[name]=pt
  for row,e in zip(mt,et):predrows.append([seconds,name,*row,e])
  block=(mt[:,3]/25//60).astype(int);blocks=np.unique(block);sums=np.array([et[block==b].sum() for b in blocks]);counts=np.array([(block==b).sum() for b in blocks]);draw=rng.integers(0,len(blocks),size=(1000,len(blocks)));means=sums[draw].sum(1)/counts[draw].sum(1)
  blockrows.append(dict(horizon_s=seconds,model=name,ci_low=np.quantile(means,.025),ci_high=np.quantile(means,.975),blocks=len(blocks)))
  print(seconds,name,round(et.mean(),3),flush=True)
 # Evaluation strata are descriptive; no claim that nearby opponents cause errors.
 if seconds==1:
  animated=[]
  for k in np.argsort(mt[:,4])[:200]:
   t,j=mt[k,:2].astype(int)
   if len(animated)>=6:break
   if any(abs(t-a['origin_index'])<250 for a in animated):continue
   frames=games[3]['pos'][t-25:t+26:2];ball=games[3]['ball'][t-25:t+26:2]
   animated.append(dict(origin_index=int(t),frame=int(mt[k,3]),player=int(j),team=games[3]['teams'].tolist(),positions=np.where(np.isfinite(frames),frames,np.nan).tolist(),ball=ball.tolist(),origin=xt[k,:2].tolist(),truth=(xt[k,:2]+yt[k]).tolist(),predictions={name:(xt[k,:2]+pt[k]).tolist() for name,pt in fitted.items()},radius=intervals[(1,'Boosted trees interactions')]))
  # JSON null replaces missing positions for browser safety.
  (OUT/'animation.json').write_text(json.dumps(animated).replace('NaN','null'))
pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(blockrows).to_csv(OUT/'block_bootstrap.csv',index=False)
pd.DataFrame(predrows,columns=['horizon_s','model','origin_index','player_index','period','frame','nearest_opponent_m','speed_mps','error_m']).to_csv(OUT/'test_predictions.csv.gz',index=False)
# Paired eligibility across every simulated blackout condition.
from src.blackout import evaluate_blackout
evaluate_blackout(games[3]).to_csv(OUT/'blackout_sensitivity.csv',index=False)
manifest={'source':'https://github.com/metrica-sports/sample-data','commit':((source/'SOURCE_COMMIT.txt').read_text().strip() if (source/'SOURCE_COMMIT.txt').exists() else subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()),'python':platform.python_version(),'split':{'train':1,'calibration':2,'test':3},'seed':20261017,'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'data').rglob('*') if p.is_file()}}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2));(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Finished',flush=True)
