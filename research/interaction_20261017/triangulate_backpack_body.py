"""Screen all-view backpack body landmarks without filling unobserved joints."""
import os
from pathlib import Path
import sys,json,time,itertools
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np
from PIL import Image,ImageDraw
out=ROOT/'backpack_body_triangulation_all99_v1';out.mkdir(exist_ok=False);start=time.time()
data=ROOT/'backpack_native000_098_v2';obs=ROOT/'backpack_wholebody_all99_v2';meta=json.loads((data/'manifest.json').read_text());r=json.loads((obs/'report.json').read_text());assert r['success'] and [v['serial'] for v in r['views']]==[v['serial'] for v in meta['views']]
d=np.load(obs/'detections.npz');xy=d['xy'][:,:,:17].astype(float);score=d['score'][:,:,:17];eligible=d['in_image'][:,:,:17]&(score>1.)
K=np.array([v['K'] for v in meta['views']]);W=np.array([v['W'] for v in meta['views']]);P=K@W[:,:3];centers=-np.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);T=xy.shape[1];assert T==99
xyz=np.full((T,17,3),np.nan);inlier=np.zeros_like(eligible);errors=np.full_like(score,np.nan,dtype=float)
def solve(ids,t,j):
 rows=np.concatenate([np.stack([xy[i,t,j,0]*P[i,2]-P[i,0],xy[i,t,j,1]*P[i,2]-P[i,1]]) for i in ids]);_,_,V=np.linalg.svd(rows);h=V[-1];return h[:3]/h[3] if abs(h[3])>1e-10 else np.full(3,np.nan)
for t in range(T):
 for j in range(17):
  ids=np.flatnonzero(eligible[:,t,j]);best=None
  for pair in itertools.combinations(ids,2):
   point=solve(pair,t,j)
   if not np.isfinite(point).all():continue
   q=P@np.r_[point,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,t,j],axis=1);ok=eligible[:,t,j]&(q[:,2]>.01)&(error<24);rank=(int(ok.sum()),-float(error[ok].sum()))
   if best is None or rank>best[0]:best=(rank,ok)
  if best is None or best[0][0]<3:continue
  keep=np.flatnonzero(best[1]);point=solve(keep,t,j);rays=point-centers[keep];rays/=np.linalg.norm(rays,axis=1)[:,None]
  if np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max()<3:continue
  q=P@np.r_[point,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,t,j],axis=1);ok=eligible[:,t,j]&(q[:,2]>.01)&(error<24)
  if ok.sum()<3:continue
  xyz[t,j]=point;inlier[:,t,j]=ok;errors[:,t,j]=error
np.savez_compressed(out/'triangulation.npz',xyz=xyz,observed=xy,raw_score=score,inlier=inlier,reprojection_error=errors,frames=d['frames'])
bones=[]
for name,a,b in [('left_upper_arm',5,7),('right_upper_arm',6,8),('left_forearm',7,9),('right_forearm',8,10),('left_thigh',11,13),('right_thigh',12,14),('left_shin',13,15),('right_shin',14,16),('shoulder_width',5,6)]:
 length=np.linalg.norm(xyz[:,a]-xyz[:,b],axis=-1);length=length[np.isfinite(length)];bones.append({'name':name,'supported_times':len(length),'median_numerical_units':float(np.median(length)) if len(length) else None,'coefficient_of_variation':float(length.std()/length.mean()) if len(length) else None})
samples=[0,15,30,50,75,98];sheet=Image.new('RGB',(6*384,8*312),'white')
for vi,v in enumerate(meta['views']):
 for col,t in enumerate(samples):
  im=Image.open(data/v['serial']/f'{t:04d}.png').convert('RGB');draw=ImageDraw.Draw(im)
  for j,(x,y) in enumerate(xy[vi,t]):
   if inlier[vi,t,j]:draw.ellipse((x-4,y-4,x+4,y+4),fill='lime')
  sheet.paste(im.resize((384,288)),(col*384,vi*312+24));ImageDraw.Draw(sheet).text((col*384+3,vi*312+4),f"{v['serial']} f{t} | accepted body points",fill='black')
sheet.save(out/'all8_geometric_screen.jpg',quality=94)
report={'success':True,'frames':99,'views':8,'observations_screened':792,'supported_3d_body_points':int(np.isfinite(xyz).all(-1).sum()),'supported_2d_points':int(inlier.sum()),'mean_inlier_error_native_px':float(errors[inlier].mean()),'per_joint_supported_times':np.isfinite(xyz).all(-1).sum(0).tolist(),'bones':bones,'seconds':time.time()-start,'scope':'Native RTMW body17: raw-score>1 provisional gate, pair RANSAC, at least3positive-depth views within24native pixels and3degree ray span. Own camera order/calibration/PTS; no helmet actor data. Missing3D joints remainNaN. Screening residual is selected input agreement, not body accuracy or calibration/synchronization acceptance. Left/right labeling, motion blur and near-parallel camera ambiguity remain; bone variation is diagnostic. Numerical raw calibration units divided1000 are not certified metres. Original overlays/coordinates remain private.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['supported_3d_body_points','supported_2d_points','mean_inlier_error_native_px','per_joint_supported_times','seconds']}),flush=True)
