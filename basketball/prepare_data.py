"""Create a lossless float64 subset of the source markers used by this study."""
from pathlib import Path
import sys,json,hashlib,subprocess,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
JOINTS=['NOSE','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST','LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE']
def main(source):
 index=[];source=Path(source);groups={}
 for p in sorted((source/'basketball/freethrow/data').rglob('BB*.json')):
  raw=p.read_bytes();x=json.loads(raw);key=x['trial_date']+'_'+x['participant_id'];bundle=groups.setdefault(key,{})
  tr=x['tracking'];trial=x['trial_id'];bundle[trial+'_pose']=np.array([[f['data']['player'].get(j,[None]*3) for j in JOINTS] for f in tr],dtype=np.float64)
  bundle[trial+'_time']=np.array([f['time'] for f in tr],dtype=np.float64);bundle[trial+'_ball']=np.array([f['data']['ball'] for f in tr],dtype=np.float64)
  meta={k:v for k,v in x.items() if k!='tracking'};meta.update(bundle=key+'.npz',frames=len(tr),source_path=str(p.relative_to(source)),source_sha256=hashlib.sha256(raw).hexdigest());index.append(meta)
 for key,arrays in groups.items():np.savez_compressed(ROOT/'data'/f'{key}.npz',**arrays)
 (ROOT/'data/index.json').write_text(json.dumps({'joints':JOINTS,'trials':index},indent=2))
 (ROOT/'data/provenance.json').write_text(json.dumps({'source':'https://github.com/Sport-Performance-Lab/SPL-Open-Data','commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'transformation':'Selected 13 core body markers, ball coordinates, time stamps and trial metadata; float64, source units preserved; no imputation or downsampling. Other hand/face/foot markers omitted. Source JSON files hashed in index.json.','license':'CC BY-NC-SA 4.0 plus source usage terms; see SOURCE_LICENSE.txt'},indent=2))
 shutil.copy2(source/'LICENSE',ROOT/'data/SOURCE_LICENSE.txt');shutil.copy2(source/'README.md',ROOT/'data/SOURCE_README.md');shutil.copy2(source/'basketball/freethrow/README.md',ROOT/'data/SOURCE_FREE_THROW_README.md')
 print('Trials:',len(index),'bundles:',len(groups),flush=True)
if __name__=='__main__':main(sys.argv[1])
