import numpy as np
from scipy.signal import savgol_filter
FEATURES=['wrist_speed_mps','knee_min_deg','elbow_angle_deg','wrist_hip_height_m','knee_extension_time_s','elbow_extension_time_s','coordination_lag_s']
GRID=np.linspace(-.6,.4,101)
def joint_angle(a,b,c):
 u=a-b;v=c-b;den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
 cos=np.divide(np.sum(u*v,axis=-1),den,out=np.full_like(den,np.nan),where=den>1e-10)
 return np.degrees(np.arccos(np.clip(cos,-1,1)))
def repair_short_gaps(values,max_gap=3):
 x=values.copy();shape=x.shape;x=x.reshape(len(x),-1)
 for j in range(x.shape[1]):
  valid=np.isfinite(x[:,j]);bad=np.where(~valid)[0]
  for run in np.split(bad,np.where(np.diff(bad)>1)[0]+1):
   if len(run) and len(run)<=max_gap and run[0]>0 and run[-1]<len(x)-1 and valid[run[0]-1] and valid[run[-1]+1]:
    x[run,j]=np.interp(run,[run[0]-1,run[-1]+1],x[[run[0]-1,run[-1]+1],j])
 return x.reshape(shape)
def extract(pose_feet,time_ms,joints,rate,smoothing_s=.15):
 time=np.asarray(time_ms)/1000;pose=repair_short_gaps(np.asarray(pose_feet)*.3048)
 if len(time)<20 or not np.all(np.diff(time)>0):raise ValueError('Invalid timestamps or short trial')
 if not np.isfinite(pose).all():raise ValueError('Unrepaired marker gaps')
 win=max(5,int(round(rate*smoothing_s))|1);win=min(win,(len(time)-1)//2*2+1)
 sm=savgol_filter(pose,win,2,axis=0);lookup={j:i for i,j in enumerate(joints)}
 def xyz(name):return sm[:,lookup[name]]
 wrists=(xyz('LEFT_WRIST')+xyz('RIGHT_WRIST'))/2;hips=(xyz('LEFT_HIP')+xyz('RIGHT_HIP'))/2;shoulders=(xyz('LEFT_SHOULDER')+xyz('RIGHT_SHOULDER'))/2
 velocity=np.gradient(wrists[:,2],time)
 eligible=(time>=time[0]+.6)&(time<=time[-1]-.4)&(wrists[:,2]>shoulders[:,2]-.15)
 if not eligible.any():raise ValueError('No elevated-arm phase anchor')
 anchor=np.where(eligible,velocity,-np.inf).argmax()
 if velocity[anchor]<=0:raise ValueError('No upward wrist movement')
 relative=time-time[anchor]
 elbow=np.mean([joint_angle(xyz(side+'_SHOULDER'),xyz(side+'_ELBOW'),xyz(side+'_WRIST')) for side in ['LEFT','RIGHT']],axis=0)
 knee=np.mean([joint_angle(xyz(side+'_HIP'),xyz(side+'_KNEE'),xyz(side+'_ANKLE')) for side in ['LEFT','RIGHT']],axis=0)
 height=wrists[:,2]-hips[:,2]
 mask=(relative>=-.6)&(relative<=.4);before=(relative>=-.6)&(relative<=0)
 kt=relative[np.where(mask,np.gradient(knee,time),-np.inf).argmax()];et=relative[np.where(mask,np.gradient(elbow,time),-np.inf).argmax()]
 features=dict(zip(FEATURES,[velocity[anchor],np.min(knee[before]),elbow[anchor],height[anchor],kt,et,et-kt]))
 curves=np.stack([np.interp(GRID,relative,elbow),np.interp(GRID,relative,knee),np.interp(GRID,relative,height)],axis=1)
 return features,curves,anchor,sm,time
