"""Screen six native foot observations and infer fixed contact intervals."""
import os
from pathlib import Path
import sys,json,argparse,itertools,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.signal import savgol_filter
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='body_head_faces_all300_v1');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());d=np.load(ROOT/'observations_all300_v1/wholebody.npz');xy=d['xy'][:,:,17:23].astype(float);score=d['score'][:,:,17:23];usable=d['in_image'][:,:,17:23]&(score>2);assert xy.shape==(8,300,6,2)
b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);poses=b['poses'];scale=float(poses['scale']);ankles=poses['joints55_world'][:,[7,8]].numpy()
K=np.array([v['K'] for v in meta['views']]);W=np.array([v['W'] for v in meta['views']]);P=K@W[:,:3];centers=-np.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);world=np.full((300,6,3),np.nan);inlier=np.zeros_like(usable);residual=np.full_like(score,np.nan,dtype=float)
def triangulate(ids,observations):
 rows=[]
 for v in ids:
  x,y=observations[v];rows.extend([x*P[v,2]-P[v,0],y*P[v,2]-P[v,1]])
 _,_,vh=np.linalg.svd(rows,full_matrices=False);h=vh[-1];return h[:3]/h[3] if abs(h[3])>1e-10 else None
def errors(x,obs):
 pc=np.einsum('vij,j->vi',W[:,:3,:3],x)+W[:,:3,3];q=np.einsum('vij,vj->vi',K,pc);e=np.linalg.norm(q[:,:2]/q[:,2:]-obs,axis=-1);e[pc[:,2]<=0]=np.inf;return e
for t in range(300):
 for j in range(6):
  ids=np.where(usable[:,t,j])[0];best=None
  for pair in itertools.combinations(ids,2):
   x=triangulate(pair,xy[:,t,j])
   if x is None or not np.isfinite(x).all() or np.linalg.norm(x-ankles[t,j//3])>.35*scale:continue
   e=errors(x,xy[:,t,j]);ok=usable[:,t,j]&(e<20);rank=(int(ok.sum()),-float(np.median(e[ok])) if ok.any() else -1e9)
   if best is None or rank>best[0]:best=(rank,ok)
  if best is None or best[0][0]<3:continue
  x=triangulate(np.where(best[1])[0],xy[:,t,j]);e=errors(x,xy[:,t,j]);ok=usable[:,t,j]&(e<20)
  if ok.sum()<3:continue
  rays=x-centers[ok];rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
  if np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max()<3:continue
  world[t,j]=x;inlier[:,t,j]=ok;residual[:,t,j]=e
contact=np.zeros((300,2),bool);anchors=np.zeros((300,6,3));segments=[];speed_record=[]
for side in range(2):
 a=side*3;raw=world[:,a:a+3];good=np.isfinite(raw).all((1,2));assert good.sum()>150,'Insufficient screened foot evidence'
 filled=raw.copy()
 for j in range(3):
  for c in range(3):
   known=np.isfinite(raw[:,j,c]);filled[:,j,c]=np.interp(np.arange(300),np.where(known)[0],raw[known,j,c])
 smooth=savgol_filter(filled,21,2,axis=0);velocity=np.linalg.norm(np.gradient(smooth,axis=0)*30/scale,axis=-1).max(-1);slow=good&(velocity<.08);speed_record.append(velocity)
 edges=np.diff(np.r_[False,slow,False].astype(int));
 for lo,hi in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
  if hi-lo<12:continue
  anchor=np.nanmedian(raw[lo:hi],axis=0);radius=np.linalg.norm(raw[lo:hi]-anchor,axis=-1)/scale
  if np.nanpercentile(radius,90)>.035:continue
  contact[lo:hi,side]=True;anchors[lo:hi,a:a+3]=anchor;segments.append({'side':side,'start':int(lo),'stop_exclusive':int(hi),'frames':int(hi-lo),'observed_radius_p90_model_units':float(np.nanpercentile(radius,90))})
np.savez_compressed(out/'constraints.npz',xy=xy,raw_score=score,inlier=inlier,world=world,reprojection_error=residual,contact=contact,contact_anchors=anchors,smoothed_observed_speed=np.stack(speed_record,1),frames=d['frames'])
report={'success':True,'avatar_source':o.avatar,'screened_points':int(inlier.sum()),'possible_points':int(inlier.size),'triangulated_points':int(np.isfinite(world).all(-1).sum()),'median_native_reprojection_px':float(np.median(residual[inlier])),'contact_frames_per_foot':contact.sum(0).tolist(),'contact_segments':segments,'seconds':time.time()-start,'rules':{'min_views':3,'reprojection_native_px':20,'max_ankle_distance_model_units':.35,'min_ray_angle_deg':3,'contact_speed_model_units_per_sec':.08,'speed_filter_window_frames':21,'min_contact_frames':12,'contact_radius_p90_model_units':.035},'scope':'All8x300 existing native RTMW six-foot detections, fixed calibration, RANSAC with anatomical proximity guard. Contact intervals from low-speed smoothed triangulated toes/heel, at least12 frames, robust observed fixed anchors; gaps remain unlabelled. Not interpolated motion observations, measured physical contact or calibrated metric ground truth. Raw arrays remain private.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
