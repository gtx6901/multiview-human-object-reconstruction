"""Refit neck/head on the corrected actual LHM semantic anchors, retaining feet."""
import os
from pathlib import Path
import sys,argparse,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.signal import savgol_filter
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from canonical_identity import semantic_identity
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',required=True);p.add_argument('--anchors',required=True);p.add_argument('--reference-head',default='body_hands_all300_v1');p.add_argument('--steps',type=int,default=500);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);base=b['poses'];T=len(base['frames']);assert T==300
ar=json.loads((ROOT/o.anchors/'report.json').read_text());assert ar['semantic_identity_sha256']==semantic_identity(b)
local=torch.tensor(np.load(ROOT/o.anchors/'anchors.npz')['canonical_face_local'],device='cuda',dtype=torch.float32)
aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(T)]).cuda()
reference=torch.load(ROOT/o.reference_head/'pose_sequence.pt',weights_only=True)
assert torch.equal(reference['frames'],base['frames'])
ref=torch.cat([pose_axis_angles(reference,t,'cpu') for t in range(T)])[:,[12,15]]
# Zero-phase matrix smoothing avoids rotvec branch artifacts.
rot=axis_angle_to_matrix(ref).numpy();sm=savgol_filter(rot,11,2,axis=0);u,_,vh=np.linalg.svd(sm);fix=np.broadcast_to(np.eye(3),sm.shape).copy();fix[...,-1,-1]=np.linalg.det(u@vh)
ref=matrix_to_axis_angle(torch.tensor(u@fix@vh,dtype=torch.float32)).cuda()
raw=torch.nn.Parameter(torch.zeros(T,2,3,device='cuda'));rest=b['rig']['rest_joints'].cuda().expand(T,-1,-1);parents=b['rig']['parents'].cuda();scale=base['scale'].cuda();trans=base['translation_world'].cuda()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
d=np.load(ROOT/'body_head_faces_all300_v1/face_screen.npz');valid=d['inlier'].copy()
for vi,v in enumerate(meta['views']):
 hand=np.load(ROOT/'lhm_joint_head_occlusion_all300_v2'/f"{v['serial']}.npz")['hand_unknown'];q=np.floor((d['xy'][vi]+.5)*.125).astype(int);valid[vi]&=hand[np.arange(T)[:,None],q[...,1].clip(0,191),q[...,0].clip(0,255)]==0
xy=torch.tensor(d['xy'],device='cuda',dtype=torch.float32);valid=torch.tensor(valid,device='cuda');conf=torch.tensor(np.minimum(d['scores']/5,1),device='cuda',dtype=torch.float32)
def project_face(aa):
 R=axis_angle_to_matrix(aa);joints,_=batch_rigid_transform(R,rest,parents);head=R[:,0]
 for j in [3,6,9,12,15]:head=head@R[:,j]
 points=(joints[:,15,None]+torch.einsum('tij,kj->tki',head,local))*scale+trans[:,None]
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],points)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,pc)
 return q[...,:2]/q[...,2:].clamp_min(.01),head,joints[:,15]*scale+trans
with torch.no_grad():
 uv,head,ht=project_face(aa0);centers=-torch.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);front=local[0]-local[1:3].mean(0);front[1]=0;front=torch.nn.functional.normalize(front,dim=0);direction=torch.einsum('tij,j->ti',head,front);to=torch.nn.functional.normalize(centers[:,None]-ht[None],dim=-1);valid[:,:,:3]&=((to*direction[None]).sum(-1)>.15)[...,None]
 before=(uv-xy).norm(dim=-1);weight=conf*valid
robust=lambda x:2*(torch.sqrt(1+x*x)-1)
def evaluate():
 delta=raw.tanh()*torch.tensor([.35,.5],device='cuda')[None,:,None]/3**.5
 aa=aa0.clone();aa[:,[12,15]]=ref+delta
 uv,head,ht=project_face(aa);e=(uv-xy).norm(dim=-1);data=(robust(e/8)*weight).sum()/weight.sum()
 R=axis_angle_to_matrix(aa[:,[12,15]]);acc=(R[2:]-2*R[1:-1]+R[:-2])*900
 loss=data+.20*robust(acc/8).mean()+.25*(delta/torch.tensor([.35,.5],device='cuda')[None,:,None]).square().mean()
 return loss,e,aa,delta
opt=torch.optim.Adam([raw],lr=.025);hist=[];best=None
for i in range(o.steps+1):
 opt.zero_grad();loss,e,aa,delta=evaluate();v=float(loss.detach())
 if best is None or v<best[0]:best=(v,raw.detach().clone())
 if i%100==0:hist.append({'step':i,'loss':v,'face_native_px':float(e[valid].mean().detach())});print(json.dumps(hist[-1]),flush=True)
 if i<o.steps:loss.backward();opt.step()
with torch.no_grad():
 raw.copy_(best[1]);_,e,aa,delta=evaluate();model=SMPLXSurfaceLandmarks().cuda();r=model(aa,scale,trans,betas=base['betas'].cuda());pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],r['coco17'][:,5:17])+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,pc)
 poses={**base,'body_pose':aa[:,1:22].cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu(),'projected':((q[...,:2]/q[...,2:].clamp_min(.01)+.5)/2-.5).cpu()}
 ids=[j for j in range(55) if j not in [12,15]];assert torch.equal(aa[:,ids],aa0[:,ids]);torch.save({**b,'poses':poses},out/'avatar.pt');torch.save(poses,out/'pose_sequence.pt')
 report={'success':True,'avatar':o.avatar,'anchors':o.anchors,'reference_head':o.reference_head,'frames':T,'screened_points':int(valid.sum()),'before_face_native_px':float(before[valid].mean()),'after_face_native_px':float(e[valid].mean()),'head_delta_max_norm_rad':float(delta[:,1].norm(dim=-1).max()),'neck_delta_max_norm_rad':float(delta[:,0].norm(dim=-1).max()),'head_bounds_rad':.5,'neck_bounds_rad':.35,'features_decoder_rig_body_legs_fingers_fixed':True,'history':hist,'seconds':time.time()-start,'scope':'Corrected LHM-render semantic face5, hand-occlusion screen and directional nose/eye visibility. Strong bounded neck/head correction around smoothed pre-face-fitting head motion; rotation acceleration in seconds. Only neck/head local motion changes; no manual target angles. Candidate needs actual same-camera review.'}
 (out/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'face_screen.npz',xy=d['xy'],scores=d['scores'],inlier=valid.cpu().numpy(),xyz=d['xyz'],frames=d['frames']);print(json.dumps(report),flush=True)
