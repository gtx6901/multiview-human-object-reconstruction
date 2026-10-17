"""Seven-camera-only pose/scale initialization; no all8 fitted body loaded."""
import os
from pathlib import Path
import sys,json,itertools,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np,torch
from scipy.spatial.transform import Rotation,Slerp
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
torch.set_num_threads(4);out=ROOT/'heldout717_body_initial_v1';out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());views=[v for v in meta['views'] if v['serial']!='21305717'];ids=[i for i,v in enumerate(meta['views']) if v['serial']!='21305717'];assert len(views)==7
det=np.load(ROOT/'observations_all300_v1/body.npz');xy=det['xy'][ids];scores=det['score'][ids]
A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
P=np.stack([A@np.array(v['K'])@np.array(v['W'])[:3] for v in views]);xyz=np.full((300,17,3),np.nan);support=np.zeros((300,17),int)
def dlt(cameras,points):
    rows=np.concatenate([np.stack([point[0]*P[i,2]-P[i,0],point[1]*P[i,2]-P[i,1]]) for i,point in zip(cameras,points)])
    _,_,V=np.linalg.svd(rows);h=V[-1];return h[:3]/h[3]
for t in range(300):
    for j in range(5,17):
        eligible=np.flatnonzero((scores[:,t,j]>3)&(xy[:,t,j,0]>2)&(xy[:,t,j,0]<1022)&(xy[:,t,j,1]>2)&(xy[:,t,j,1]<766))
        if len(eligible)<3:continue
        best=None
        for pair in itertools.combinations(eligible,2):
            x=dlt(pair,xy[list(pair),t,j]);q=P@np.r_[x,1];err=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,t,j],axis=1);keep=np.isin(np.arange(7),eligible)&(err<15)&(q[:,2]>0)
            rank=(int(keep.sum()),-float(err[keep].sum()))
            if best is None or rank>best[0]:best=(rank,keep)
        if best[0][0]>=3:
            keep=np.flatnonzero(best[1]);xyz[t,j]=dlt(keep,xy[keep,t,j]);support[t,j]=len(keep)
    if t%75==0:print('seven-view triangulation',t,flush=True)
# NLF inference/fitting is per camera; fuse only seven permitted pose priors.
available=list(range(136))+list(range(180,212));initial=[]
for t in available:
    folder=ROOT/('nlf_smplh_pickup000_135_v1' if t<=135 else 'nlf_smplh_contiguous180_211_v1');rot=[]
    for v in views:
        pose=torch.load(folder/f"{v['serial']}_{t:04d}.pt",weights_only=True)['pose_rotvecs'][0].numpy()[:66].reshape(22,3);r=Rotation.from_rotvec(pose).as_matrix();r[0]=np.array(v['W'])[:3,:3].T@r[0];rot.append(r)
    rot=np.stack(rot);frame=[]
    for j in range(22):
        rs=Rotation.from_matrix(rot[:,j]);dist=np.array([np.linalg.norm((Rotation.from_matrix(r).inv()*rs).as_rotvec(),axis=1) for r in rot[:,j]]);medoid=dist.sum(1).argmin();frame.append(rs[dist[medoid]<np.deg2rad(70)].mean().as_rotvec())
    initial.append(frame)
initial=np.array(initial);times=np.arange(300);pose=np.stack([Slerp(available,Rotation.from_rotvec(initial[:,j]))(np.clip(times,available[0],available[-1])).as_rotvec() for j in range(22)],1)
aa=torch.cat([torch.tensor(pose,dtype=torch.float32),torch.zeros(300,33,3)],1);model=SMPLXSurfaceLandmarks();pred=model(aa)['coco17'].detach().numpy()
ratios=[]
for a,b in [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12)]:
    numerator=np.linalg.norm(xyz[:,a]-xyz[:,b],axis=1);denominator=np.linalg.norm(pred[:,a]-pred[:,b],axis=1);valid=np.isfinite(numerator)&(denominator>.02);ratios.extend((numerator[valid]/denominator[valid]).tolist())
scale=float(np.median(ratios));assert .05<scale<5
translation=np.nanmedian(xyz[:,5:17]-pred[:,5:17]*scale,axis=1);assert np.isfinite(translation).all()
torch.save({'frames':torch.arange(300),'root_pose':aa[:,0],'body_pose':aa[:,1:22],'translation_world':torch.tensor(translation,dtype=torch.float32),'scale':torch.tensor(scale),'betas':torch.zeros(1,10)},out/'pose_sequence.pt')
np.savez_compressed(out/'triangulation.npz',xyz=xyz,support=support)
(out/'report.json').write_text(json.dumps({'success':True,'excluded_camera':'21305717','training_cameras':[v['serial'] for v in views],'scale_from_training_view_triangulated_bones':scale,'frames':300,'per_camera_nlf_prior_frames':available,'seconds':time.time()-start,'scope':'Only seven independent body detections and per-camera NLF priors used for root/body/translation/scale initialization. No all8 fitted pose/shape/scale initialization loaded. Missing NLF times interpolated or held as INITIALIZATION only; subsequent fitting uses all300 training-view observations. Original calibrated shared world gauge retained. This is not held-out reconstruction quality evidence.'},indent=2));print('seven-view body initialization saved',flush=True)
