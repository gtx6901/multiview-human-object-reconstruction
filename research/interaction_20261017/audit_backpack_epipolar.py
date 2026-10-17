"""Compare extrinsic-convention hypotheses, preserving original cameras."""
import os
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',default='backpack_epipolar_audit_v1');p.add_argument('--calibration');o=p.parse_args()
m=json.loads((ROOT/'backpack_native000_098_v2/manifest.json').read_text());d=np.load(ROOT/'backpack_wholebody_all99_v2/detections.npz');xy=d['xy'][:,:,:17];valid=d['in_image'][:,:,:17]&(d['score'][:,:,:17]>2);K=np.array([v['K'] for v in m['views']]);W=np.array([v['W'] for v in m['views']]);pairs=[]
if o.calibration:
 cameras={}
 for block in Path(o.calibration).read_text().strip().split('\n\n'):
  if not block.strip():continue
  lines=[s.strip() for s in block.splitlines() if s.strip()];assert len(lines)==8
  matrix=np.eye(4);matrix[:3]=np.array([[float(x) for x in s.split()] for s in lines[5:8]]);matrix[:3,3]/=1000;cameras[lines[0]]=matrix
 W=np.stack([cameras[v['serial']] for v in m['views']])
for i in range(8):
 for j in range(i):
  row={'cameras':[m['views'][j]['serial'],m['views'][i]['serial']]}
  transpose=W.copy();transpose[:,:3,:3]=W[:,:3,:3].transpose(0,2,1)
  centers=W.copy();centers[:,:3,3]=-np.einsum('bij,bj->bi',W[:,:3,:3],W[:,:3,3])
  for label,Ws in [('as_prepared',W),('inverse_convention',np.linalg.inv(W)),('rotation_transposed_only',transpose),('last_column_is_camera_center',centers)]:
   T=Ws[i]@np.linalg.inv(Ws[j]);t=T[:3,3];S=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);F=np.linalg.inv(K[i]).T@S@T[:3,:3]@np.linalg.inv(K[j])
   a=np.concatenate([xy[j],np.ones_like(xy[j,...,:1])],-1);b=np.concatenate([xy[i],np.ones_like(xy[i,...,:1])],-1);la=a@F.T;lb=b@F
   e=abs((b*la).sum(-1))*.5*(1/np.linalg.norm(la[...,:2],axis=-1).clip(1e-10)+1/np.linalg.norm(lb[...,:2],axis=-1).clip(1e-10));row[label]=float(np.median(e[valid[i]&valid[j]]))
  pairs.append(row)
out=ROOT/o.out;out.mkdir(exist_ok=False);report={'pairs':pairs,'extrinsic_source':o.calibration or m['calibration'],'scope':'Same-frame native body17 symmetric epipolar distance, raw-score>2. Direct versus inverse extrinsic convention diagnostic only; prepared image intrinsics remain fixed even for alternative extrinsics. Not synchronization or calibration certification, no original camera altered.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(pairs),flush=True)
