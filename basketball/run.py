from pathlib import Path
import json,hashlib,warnings,platform
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,accuracy_score
from src.mechanics import extract,FEATURES,GRID
ROOT=Path(__file__).resolve().parent;O=ROOT/'output';O.mkdir(exist_ok=True)
index=json.loads((ROOT/'data/index.json').read_text());joints=index['joints'];bundles={};rows=[];curves=[];excluded=[];animations=[];rawmissing=[]
for meta in index['trials']:
 b=bundles.setdefault(meta['bundle'],np.load(ROOT/'data'/meta['bundle'])) if meta['bundle'] not in bundles else bundles[meta['bundle']]
 trial=meta['trial_id'];pose=b[trial+'_pose'];time=b[trial+'_time'];rawmissing.append(float(np.isnan(pose).mean()))
 try:feature,curve,anchor,sm,t=extract(pose,time,joints,meta['sampling_rate'])
 except ValueError as e:excluded.append(dict(session=meta['trial_date'],participant=meta['participant_id'],trial=trial,reason=str(e)));continue
 row=dict(session=meta['trial_date'],participant=meta['participant_id'],trial=trial,trial_order=int(trial[1:]),made=int(meta['result']=='made'),sampling_rate=meta['sampling_rate'],anchor_time_s=float(t[anchor]),raw_missing_fraction=rawmissing[-1],**feature);rows.append(row);curves.append(curve)
 # One clearly identified example per participant/session, independent of result.
 if not any(a['participant']==row['participant'] and a['session']==row['session'] for a in animations):
  sel=np.where((t-t[anchor]>=-.6)&(t-t[anchor]<=.4))[0][::2]
  center=(sm[anchor,joints.index('LEFT_HIP')]+sm[anchor,joints.index('RIGHT_HIP')])/2;center[2]=0
  animations.append(dict(participant=row['participant'],session=row['session'],trial=trial,made=row['made'],joints=joints,time=(t[sel]-t[anchor]).tolist(),pose=(sm[sel]-center).tolist(),features=feature))
df=pd.DataFrame(rows);df.to_csv(O/'trial_features.csv',index=False);pd.DataFrame(excluded,columns=['session','participant','trial','reason']).to_csv(O/'excluded_trials.csv',index=False)
C=np.stack(curves);np.savez_compressed(O/'aligned_curves.npz',curves=C,grid=GRID)
primary=df[df.session=='2025-12-18'].copy();primary['progress']=primary.groupby('participant').trial_order.transform(lambda s:(s-s.min())/(s.max()-s.min()))
summary=df.groupby(['session','participant']).agg(trials=('made','size'),made=('made','sum'),made_rate=('made','mean'),wrist_speed_mean=('wrist_speed_mps','mean'),wrist_speed_sd=('wrist_speed_mps','std'),knee_angle_mean=('knee_min_deg','mean'),elbow_angle_mean=('elbow_angle_deg','mean')).reset_index();summary.to_csv(O/'participant_summary.csv',index=False)
variance=[];log=[]
for feature in FEATURES[:4]:
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');fit=smf.mixedlm(feature+' ~ progress',primary,groups=primary.participant).fit(reml=True,method='powell',maxiter=1000,disp=False)
  log.extend([str(w.message) for w in caught])
 between=float(fit.cov_re.iloc[0,0]);within=float(fit.scale)
 variance.append(dict(feature=feature,between_variance=between,within_variance=within,conditional_icc=between/(between+within),progress_coefficient=float(fit.fe_params['progress']),converged=bool(fit.converged),participants=primary.participant.nunique(),trials=len(primary)))
pd.DataFrame(variance).to_csv(O/'variance_components.csv',index=False);(O/'mixed_model_warnings.txt').write_text('\n'.join(log))
# Descriptive PCA; never used as a fitted predictor in the held-out experiments.
P=C[df.session.to_numpy()=='2025-12-18'];mean=P.mean(axis=(0,1));scale=P.std(axis=(0,1));Z=((P-mean)/scale).reshape(len(P),-1);pca=PCA(n_components=3,svd_solver='full');scores=pca.fit_transform(Z)
embedding=primary[['participant','trial','made']].copy();embedding[['PC1','PC2','PC3']]=scores;embedding.to_csv(O/'trajectory_pca_scores.csv',index=False);(O/'pca.json').write_text(json.dumps({'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'channel_order':['elbow angle degrees','knee angle degrees','wrist above hip metres'],'grid_s':GRID.tolist(),'scope':'Descriptive fit to all eligible 2025 trials; not used in supervised evaluation.'},indent=2))
# Freeze simple settings. Split whole trials; never mix frames from the same shot.
X=primary[FEATURES].to_numpy();y=primary.made.to_numpy();ids=primary.participant.to_numpy();predictions=[]
splits=[]
train=np.zeros(len(primary),bool)
for player in np.unique(ids):
 ix=np.where(ids==player)[0];order=ix[np.argsort(primary.iloc[ix].trial_order.to_numpy())];train[order[:int(np.floor(.7*len(order)))]]=True
splits.append(('Chronological 70/30',train,~train,'all'))
for player in np.unique(ids):splits.append(('Leave one athlete out',ids!=player,ids==player,player))
for design,tr,te,fold in splits:
 models={'Pooled rate':None,'Pose logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=17)),'Pose random forest':RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=8,random_state=17,n_jobs=2)}
 for name,model in models.items():
  if model is None:prob=np.full(te.sum(),y[tr].mean())
  else:model.fit(X[tr],y[tr]);prob=model.predict_proba(X[te])[:,1]
  for idx,p in zip(np.where(te)[0],prob):predictions.append(dict(design=design,fold=fold,model=name,participant=ids[idx],trial=primary.iloc[idx].trial,made=int(y[idx]),probability=float(p)))
 # Known-athlete baseline only has meaning for chronological evaluation.
 if design=='Chronological 70/30':
  for idx in np.where(te)[0]:predictions.append(dict(design=design,fold=fold,model='Athlete training rate',participant=ids[idx],trial=primary.iloc[idx].trial,made=int(y[idx]),probability=float(y[tr&(ids==ids[idx])].mean())))
pred=pd.DataFrame(predictions);pred.to_csv(O/'outcome_predictions.csv',index=False)
metrics=[]
for (design,model),g in pred.groupby(['design','model']):
 metrics.append(dict(design=design,model=model,n=len(g),brier=brier_score_loss(g.made,g.probability),log_loss=log_loss(g.made,np.clip(g.probability,1e-8,1-1e-8)),auc=roc_auc_score(g.made,g.probability),accuracy=accuracy_score(g.made,g.probability>=.5)))
pd.DataFrame(metrics).to_csv(O/'outcome_metrics.csv',index=False)
# Descriptive smoothing sensitivity on all original trials; same eligibility is recorded.
sensitivity=[]
for width in [.1,.25]:
 for meta in index['trials']:
  if meta['trial_date']!='2025-12-18':continue
  b=bundles[meta['bundle']];trial=meta['trial_id']
  try:f,_,_,_,_=extract(b[trial+'_pose'],b[trial+'_time'],joints,meta['sampling_rate'],width)
  except ValueError:continue
  sensitivity.append(dict(smoothing_s=width,participant=meta['participant_id'],trial=trial,**f))
pd.DataFrame(sensitivity).to_csv(O/'smoothing_sensitivity.csv',index=False)
(O/'animation.json').write_text(json.dumps(animations,allow_nan=False))
audit={'source_trials':len(index['trials']),'eligible_trials':len(df),'excluded_trials':excluded,'primary_trials':len(primary),'unique_participants_all':df.participant.nunique(),'session_counts':df.groupby('session').size().to_dict(),'source_outcome_labels':sorted(set(m['result'] for m in index['trials'])),'units':'Input feet; analysis metres; time input milliseconds; analysis seconds; angles degrees','phase_anchor':'Peak upward velocity of smoothed bilateral mean wrist height, with elevated arms and complete -0.6 to +0.4 s window; proxy, not verified ball release','primary_design':'2025 five participants; 2024 P0001 is separate and cannot identify a session effect independent of measurement-system differences','input_bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.npz')}}
(O/'audit.json').write_text(json.dumps(audit,indent=2));print('Eligible',len(df),'primary',len(primary),'excluded',len(excluded));print(pd.DataFrame(variance).to_string(index=False));print(pd.DataFrame(metrics).to_string(index=False))
