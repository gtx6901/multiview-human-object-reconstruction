"""RANSAC image evidence screening; uncorroborated hand points remain unknown."""
import os
from pathlib import Path
import sys,json,time,itertools,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',default='hands_multiview_screen000_135_v2');p.add_argument('--body',default='body_surface_pickup000_135_v2');p.add_argument('--observations-file',default='wholebody_native000_135_v1/detections.npz');p.add_argument('--observations-report',default='wholebody_native000_135_v1/report.json');p.add_argument('--exclude-camera',default='');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/options.observations_report).read_text());det=np.load(ROOT/options.observations_file)
if options.exclude_camera:
    ids=[i for i,v in enumerate(meta['views']) if v['serial']!=options.exclude_camera];assert len(ids)==len(meta['views'])-1
    det={k:det[k] for k in det.files}
    for key in ['xy','score','in_image']:det[key]=det[key][ids]
    meta['views']=[meta['views'][i] for i in ids]
xy=det['xy'][:,:,91:133].astype(float);score=det['score'][:,:,91:133];usable=det['in_image'][:,:,91:133]&(score>2.)
K=np.array([v['K'] for v in meta['views']]);W=np.array([v['W'] for v in meta['views']]);P=K@W[:,:3];centers=-np.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3])
poses=torch.load(ROOT/options.body/'pose_sequence.pt',weights_only=True)
wrists=poses['joints55_world'][:,[20,21]].numpy();scale=float(poses['scale']);T=len(det['frames'])
# In each view a severely misplaced hand wrist invalidates that whole hand.
for side,base in enumerate([0,21]):
    pc=np.einsum('vij,tj->vti',W[:,:3,:3],wrists[:,side])+W[:,None,:3,3];q=np.einsum('vij,vtj->vti',K,pc);pred=q[...,:2]/q[...,2:]
    good=(np.linalg.norm(pred-xy[:,:,base],axis=-1)<90)&(pc[...,2]>0)
    usable[:,:,base:base+21]&=good[:,:,None]
# Two nearly identical detected wrists indicate ambiguous overlapping hands.
ambiguous=np.linalg.norm(xy[:,:,0]-xy[:,:,21],axis=-1)<12
usable[ambiguous]=False
def triangulate(ids,points,weights=None):
    rows=[]
    for k,v in enumerate(ids):
        wt=1 if weights is None else np.sqrt(weights[k]);u,y=points[k]
        rows.extend([wt*(u*P[v,2]-P[v,0]),wt*(y*P[v,2]-P[v,1])])
    _,_,vh=np.linalg.svd(rows,full_matrices=False);hom=vh[-1]
    if abs(hom[3])<1e-10:return None
    return hom[:3]/hom[3]
def errors(point,observations):
    pc=np.einsum('vij,j->vi',W[:,:3,:3],point)+W[:,:3,3];q=np.einsum('vij,vj->vi',K,pc)
    error=np.linalg.norm(q[:,:2]/q[:,2:]-observations,axis=1);error[pc[:,2]<=0]=np.inf;return error
inliers=np.zeros_like(usable);world=np.full((T,42,3),np.nan);residual=np.full((len(meta['views']),T,42),np.nan);counts=np.zeros((T,42),int);angles=np.zeros((T,42))
for t in range(T):
    for j in range(42):
        ids=np.where(usable[:,t,j])[0]
        if len(ids)<3:continue
        best=None
        for pair in itertools.combinations(ids,2):
            point=triangulate(pair,xy[list(pair),t,j])
            if point is None or not np.isfinite(point).all():continue
            if np.linalg.norm(point-wrists[t,j//21])>.25*scale:continue
            error=errors(point,xy[:,t,j]);ok=usable[:,t,j]&(error<30)
            rank=(int(ok.sum()),-float(np.median(error[ok])) if ok.any() else -np.inf)
            if best is None or rank>best[0]:best=(rank,ok,point)
        if best is None or best[0][0]<3:continue
        ids=np.where(best[1])[0];point=triangulate(ids,xy[ids,t,j],np.minimum(score[ids,t,j],5))
        error=errors(point,xy[:,t,j]);ok=usable[:,t,j]&(error<30)
        if ok.sum()<3:continue
        rays=point-centers[ok];rays/=np.linalg.norm(rays,axis=1)[:,None]
        angle=float(np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max())
        if angle<3:continue
        world[t,j]=point;inliers[:,t,j]=ok;counts[t,j]=ok.sum();angles[t,j]=angle;residual[:,t,j]=error
    if t%30==0:print('screened frame',t,flush=True)
np.savez_compressed(out/'screened.npz',xy=xy,raw_score=score,eligible=usable,inlier=inliers,world=world,view_count=counts,max_ray_angle_deg=angles,reprojection_error=residual,frames=det['frames'])
report={'success':True,'frames':T,'raw_points':int(score.size),'eligible_points':int(usable.sum()),'inlier_2d_points':int(inliers.sum()),'corroborated_3d_points':int((counts>=3).sum()),'possible_3d_points':T*42,'frames_with_left_hand_constraints':int((inliers[:,:,:21].sum((0,2))>0).sum()),'frames_with_right_hand_constraints':int((inliers[:,:,21:].sum((0,2))>0).sum()),'median_inlier_error_native_px':float(np.median(residual[inliers])),'seconds':time.time()-start,'rules':{'raw_simcc_min':2,'min_views':3,'reprojection_threshold_native_px':30,'min_max_ray_angle_deg':3,'max_wrist_to_point_model_units':.25,'max_wrist_reprojection_native_px':90,'ambiguous_wrist_separation_native_px':12},'scope':'Geometric consistency screening of eight-view detector predictions using fixed calibrated cameras. These are pseudo-observations, not ground truth or visibility certification. Thresholds fixed before fitting; no held-out quality claim. Underdetermined points remain unknown, not fabricated.'}
report['sources']=vars(options)
report['training_cameras']=[v['serial'] for v in meta['views']]
if options.exclude_camera:report['scope']=report['scope'].replace('eight-view','seven-view')+' Held-out view removed before any triangulation or screening.'
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
