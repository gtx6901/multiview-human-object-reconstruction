"""Observed hand PCA, temporal wrist/finger acceleration, sparse interpolation."""
import os
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',required=True);p.add_argument('--steps',type=int,default=600);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);base=b['poses'];aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(300)]).cuda();model=SMPLXSurfaceLandmarks().cuda();scale=base['scale'].cuda();trans=base['translation_world'].cuda();beta=base['betas'].cuda()
d=np.load(ROOT/'hands_multiview_screen_all300_v1/screened.npz');xy=torch.tensor(d['xy'],device='cuda',dtype=torch.float32);valid=torch.tensor(d['inlier'],device='cuda');weight=torch.tensor(np.minimum(d['raw_score']/5,1)*d['inlier'],device='cuda',dtype=torch.float32)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
bases=torch.stack([model.left_pca[:12],model.right_pca[:12]]);oldhands=aa0[:,25:].reshape(300,2,45);coeff=torch.einsum('tsi,sic->tsc',oldhands,torch.linalg.pinv(bases)).detach().cpu().numpy()
# Reliable frames anchor a temporally interpolated pose, never the flat hand.
counts=d['inlier'].reshape(8,300,2,21).any(0).sum(-1);reliable=counts>=8;interp=np.empty_like(coeff)
for side in range(2):
 ids=np.flatnonzero(reliable[:,side]);assert len(ids)>5
 for k in range(12):interp[:,side,k]=np.interp(np.arange(300),ids,coeff[ids,side,k])
interp=savgol_filter(interp,9,2,axis=0);initial=torch.tensor(interp,device='cuda',dtype=torch.float32)
params=torch.nn.Parameter(initial.clone());raw=torch.nn.Parameter(torch.zeros(300,2,3,device='cuda'))
R=axis_angle_to_matrix(aa0[:,[20,21]]).cpu().numpy();sm=savgol_filter(R,9,2,axis=0);u,_,vh=np.linalg.svd(sm);fix=np.broadcast_to(np.eye(3),sm.shape).copy();fix[...,-1,-1]=np.linalg.det(u@vh);wrist0=matrix_to_axis_angle(torch.tensor(u@fix@vh,dtype=torch.float32)).cuda()
rot=axis_angle_to_matrix(aa0);J,_=batch_rigid_transform(rot,b['rig']['rest_joints'].cuda().expand(300,-1,-1),b['rig']['parents'].cuda());H=rot[:,0]
for j in [3,6,9,12,15]:H=H@rot[:,j]
headt=J[:,15]*scale+trans
# Approximate actual corrected LHM head envelope, not an uncorrected bare mesh.
ids=torch.where(b['rig']['weights'][:,15]>.8)[0];pts=(b['rig']['zero_coords'][ids]-b['rig']['rest_joints'][0,15]).numpy();pts=pts[::max(1,len(pts)//1500)];hull=ConvexHull(pts);planes=torch.tensor(hull.equations,device='cuda',dtype=torch.float32)
# Data-derived near-face intervals after donning: share one grip per contiguous segment.
with torch.no_grad():r0=model(aa0,scale,trans,betas=beta);near=((r0['hands21'][:,:,0]-headt[:,None]).norm(dim=-1)/scale<.20).cpu().numpy();near[:180]=False
segments=[];gid=np.full((300,2),-1,int)
for side in range(2):
 edges=np.flatnonzero(np.diff(np.r_[False,near[:,side],False]));
 for a,z in zip(edges[::2],edges[1::2]):
  if z-a>=8:gid[a:z,side]=len(segments);segments.append({'start':int(a),'stop_exclusive':int(z),'side':side})
grip=torch.nn.Parameter(torch.stack([initial[x['start']:x['stop_exclusive'],x['side']].median(0).values for x in segments]) if segments else torch.empty(0,12,device='cuda'))
gids=torch.tensor(gid,device='cuda');has=gids>=0;reliable_t=torch.tensor(reliable,device='cuda');robust=lambda x:2*(torch.sqrt(1+x*x)-1)
def project(q):
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],q)+W[:,None,None,:3,3];v=torch.einsum('vij,vtkj->vtki',K,pc);return v[...,:2]/v[...,2:].clamp_min(.01)
def state():
 c=params
 if segments:c=torch.where(has[...,None],grip[gids.clamp_min(0)],c)
 hands=torch.einsum('tsc,sci->tsi',c,bases).reshape(300,30,3);aa=aa0.clone();aa[:,25:]=hands;delta=raw.tanh()*(.5/3**.5);aa[:,[20,21]]=wrist0+delta
 r=model(aa,scale,trans,betas=beta);hp=r['hands21'];e=(project(hp.reshape(300,42,3))-xy).norm(dim=-1)
 data=(robust(e/12)*weight).sum()/weight.sum();R=axis_angle_to_matrix(aa[:,[20,21]]);racc=(R[2:]-2*R[1:-1]+R[:-2])*900;cacc=(c[2:]-2*c[1:-1]+c[:-2])*900
 temporal=.30*robust(racc/8).mean()+.25*robust(cacc/30).mean();sparse=.20*robust((c-initial)/.5)[~reliable_t].mean() if (~reliable_t).any() else c.sum()*0
 local=torch.einsum('tji,tskj->tski',H,(hp-headt[:,None,None])/scale);sdf=(torch.einsum('tski,pi->tskp',local,planes[:,:3])+planes[:,3]).amax(-1);penetration=torch.relu(-sdf-.002)
 collision=.15*robust(penetration/.008).mean();loss=data+temporal+sparse+collision+.05*(delta/.5).square().mean()
 return loss,e,aa,r,c,penetration
def stats(aa):
 r=model(aa,scale,trans,betas=beta);hp=r['hands21'];e=(project(hp.reshape(300,42,3))-xy).norm(dim=-1);R=axis_angle_to_matrix(aa[:,[20,21]]);acc=(R[2:]-2*R[1:-1]+R[:-2])*900
 return {'hand_native_px':float(e[valid].mean()),'wrist_rotation_acceleration_p95':float(torch.quantile(acc.norm(dim=(-2,-1)),.95)),'fingers_d2_p95':float(torch.quantile((aa[2:,25:]-2*aa[1:-1,25:]+aa[:-2,25:]).norm(dim=-1),.95))}
with torch.no_grad():before=stats(aa0)
opt=torch.optim.Adam([params,raw,grip],lr=.012);history=[];best=None
for i in range(o.steps+1):
 opt.zero_grad();loss,e,aa,r,c,pen=state();val=float(loss.detach())
 if best is None or val<best[0]:best=(val,[x.detach().clone() for x in [params,raw,grip]])
 if i%100==0:history.append({'step':i,'loss':val,'hand_native_px':float(e[valid].mean().detach()),'penetration_mean_model':float(pen.mean().detach())});print(json.dumps(history[-1]),flush=True)
 if i<o.steps:loss.backward();opt.step()
with torch.no_grad():
 for x,y in zip([params,raw,grip],best[1]):x.copy_(y)
 _,e,aa,r,c,pen=state();poses={**base,'body_pose':aa[:,1:22].cpu(),'left_hand_pose':aa[:,25:40].cpu(),'right_hand_pose':aa[:,40:55].cpu(),'lhand_pose':aa[:,25:40].cpu(),'rhand_pose':aa[:,40:55].cpu(),'hand_pca':c.cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu(),'projected':((project(r['coco17'][:,5:17])+.5)/2-.5).cpu()}
 keep=[j for j in range(55) if j not in [20,21] and j<25];assert torch.equal(aa[:,keep],aa0[:,keep]);torch.save(poses,out/'pose_sequence.pt');torch.save({**b,'poses':poses},out/'avatar.pt')
 report={'success':True,'avatar':o.avatar,'frames':300,'before':before,'after':stats(aa),'near_face_shared_grips':segments,'reliable_hand_frames':reliable.sum(0).tolist(),'history':history,'seconds':time.time()-start,'scope':'Existing geometric-screened eight-view hand observations; interpolated reliable PCA anchor for sparse frames, robust seconds-based wrist rotation/PCA acceleration, shared grip in data-derived contiguous near-face worn intervals. Weak nonpenetration prior against convex envelope of corrected LHM head Gaussian centers, not exact skin or physical contact. Fixed head/neck/body/feet/shape/features/decoder; only wrist and decoded fingers change. Requires continuous rendered motion review.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
