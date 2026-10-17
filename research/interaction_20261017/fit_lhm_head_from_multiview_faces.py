"""Fit ONLY neck/head from native multiview face observations, preserve LHM."""
import os
from pathlib import Path
import sys,json,time,itertools,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from avatar_rig import pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
parser=argparse.ArgumentParser();parser.add_argument('--out',default='body_head_faces_all300_v1');parser.add_argument('--face-anchor-source');options=parser.parse_args()
torch.set_num_threads(4);out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());d=np.load(ROOT/'observations_all300_v1/wholebody.npz');xy=d['xy'][:,:,:5].astype(float);scores=d['score'][:,:,:5];eligible=d['in_image'][:,:,:5]&(scores>2)
K=np.array([v['K'] for v in meta['views']]);W=np.array([v['W'] for v in meta['views']]);P=K@W[:,:3];centers=-np.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);inlier=np.zeros_like(eligible);xyz=np.full((300,5,3),np.nan)
def triangulate(ids,t,j):
    rows=np.concatenate([np.stack([xy[i,t,j,0]*P[i,2]-P[i,0],xy[i,t,j,1]*P[i,2]-P[i,1]]) for i in ids]);_,_,V=np.linalg.svd(rows);h=V[-1];return h[:3]/h[3]
for t in range(300):
    for j in range(5):
        ids=np.flatnonzero(eligible[:,t,j]);best=None
        for pair in itertools.combinations(ids,2):
            point=triangulate(pair,t,j);q=P@np.r_[point,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,t,j],axis=1);ok=eligible[:,t,j]&(error<24)&(q[:,2]>0);rank=(int(ok.sum()),-float(error[ok].sum()))
            if best is None or rank>best[0]:best=(rank,ok)
        if best is None or best[0][0]<3:continue
        ids=np.flatnonzero(best[1]);point=triangulate(ids,t,j);rays=point-centers[ids];rays/=np.linalg.norm(rays,axis=1)[:,None];angle=np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max()
        if angle<3:continue
        q=P@np.r_[point,1];err=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,t,j],axis=1);ok=eligible[:,t,j]&(err<24)&(q[:,2]>0)
        if ok.sum()<3:continue
        xyz[t,j]=point;inlier[:,t,j]=ok
np.savez_compressed(out/'face_screen.npz',xy=xy,scores=scores,inlier=inlier,xyz=xyz,frames=d['frames'])
base=torch.load(ROOT/'body_hands_all300_v1/pose_sequence.pt',weights_only=True);aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(300)]).cuda();scale=base['scale'].cuda();translation=base['translation_world'].cuda();model=SMPLXSurfaceLandmarks().cuda()
local_face=None
if options.face_anchor_source:
    anchor_report=json.loads((ROOT/options.face_anchor_source/'report.json').read_text());assert anchor_report['success'] and all(p['usable'] and p['support']>=3 for p in anchor_report['points'])
    local_face=torch.tensor(np.load(ROOT/options.face_anchor_source/'anchors.npz')['canonical_face_local'],device='cuda',dtype=torch.float32);assert torch.isfinite(local_face).all()
observed=torch.tensor(xy,device='cuda',dtype=torch.float32);valid=torch.tensor(inlier,device='cuda');weights=valid*torch.tensor(np.minimum(scores/5,1),device='cuda',dtype=torch.float32);assert weights.sum()>3000
Ks=torch.tensor(K,device='cuda',dtype=torch.float32);Ws=torch.tensor(W,device='cuda',dtype=torch.float32);raw=torch.nn.Parameter(torch.zeros(300,2,3,device='cuda'))
def project(points):
    pc=torch.einsum('vij,tkj->vtki',Ws[:,:3,:3],points)+Ws[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',Ks,pc);return q[...,:2]/q[...,2:].clamp_min(.01)
def evaluate():
    delta=torch.tanh(raw)*torch.tensor([.35,1.5],device='cuda')[None,:,None];aa=aa0.clone();aa[:,12]+=delta[:,0];aa[:,15]+=delta[:,1];result=model(aa,scale,translation);face=result['coco17'][:,:5]
    if local_face is not None:
        rotations=axis_angle_to_matrix(aa);global_head=torch.eye(3,device='cuda').expand(300,-1,-1)
        for j in [0,3,6,9,12,15]:global_head=global_head@rotations[:,j]
        face=result['joints55'][:,15,None]+torch.einsum('bij,kj->bki',global_head,local_face)*scale
    uv=project(face);error=(uv-observed).norm(dim=-1)
    loss=(2*(torch.sqrt(1+(error/8)**2)-1)*weights).sum()/weights.sum()+.04*delta[:,0].square().mean()+.005*delta[:,1].square().mean()
    loss+=.2*(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()+.015*(delta[1:]-delta[:-1]).square().mean()
    return loss,error,uv,aa,result,delta
with torch.no_grad():_,before,*_=evaluate();before=before.clone()
opt=torch.optim.Adam([raw],lr=.04);history=[];best=None
for step in range(401):
    opt.zero_grad();values=evaluate();loss,error=values[:2]
    if best is None or float(loss.detach())<best[0]:best=(float(loss.detach()),raw.detach().clone())
    if step<400:loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.04*(1-.9*step/400)
    if step%100==0:history.append({'step':step,'loss':float(loss.detach()),'face_input_error_native_px':float(error[valid].mean().detach())});print(json.dumps(history[-1]),flush=True)
with torch.no_grad():
    raw.copy_(best[1]);_,error,uv,aa,result,delta=evaluate();poses=dict(base);poses['body_pose']=aa[:,1:22].cpu();poses['joints55_world']=result['joints55'].cpu();poses['joints_world']=result['coco17'][:,5:17].cpu();poses['projected']=(project(result['coco17'][:,5:17]).cpu()+.5)/2-.5
    unchanged=[i for i in range(21) if i not in [11,14]];assert torch.equal(poses['body_pose'][:,unchanged],base['body_pose'][:,unchanged])
    for name in ['root_pose','translation_world','scale','betas','left_hand_pose','right_hand_pose']:assert torch.equal(poses[name],base[name])
    torch.save(poses,out/'pose_sequence.pt');np.savez_compressed(out/'reprojection.npz',observed=xy,valid=inlier,before_error=before.cpu(),after_error=error.cpu(),projected=uv.cpu())
    report={'success':True,'frames':300,'views':8,'seconds':time.time()-start,'history':history,'screened_2d_points':int(inlier.sum()),'frames_with_face_constraints':int(inlier.any((0,2)).sum()),'before_face_input_error_native_px':float(before[valid].mean()),'after_face_input_error_native_px':float(error[valid].mean()),'per_camera':[{'serial':v['serial'],'valid_points':int(valid[i].sum()),'before_native_px':float(before[i][valid[i]].mean()),'after_native_px':float(error[i][valid[i]].mean())} for i,v in enumerate(meta['views'])],'scope':'Only neck12/head15 optimized from three-or-more-view RANSAC-screened native RTMW COCO face5 pseudo-observations and detector-compatible surface-regressed landmarks. Fixed torso/arms/fingers/root/scale/shape/translation. Original LHM Gaussian appearance/decoder must remain unchanged. Input residuals are not true face geometry, visual improvement or held-out proof. 24px reprojection gate and3deg ray gate; neck±.35/head±1.5rad per-component bounds, temporal smoothness. Requires side-by-side render audit; helmet motion unchanged until separately refitted.'}
    report['face_anchor_source']=options.face_anchor_source or 'generic SMPL-X surface-regressed COCO5'
    if local_face is not None:report['scope']=report['scope'].replace('detector-compatible surface-regressed landmarks','instance-specific semantic anchors triangulated from actual LHM neural renders')+' Head-local anchor motion assumes an approximately rigid facial region with jaw/eye poses fixed. Before/after use this same anchor definition; do not directly compare to generic-anchor residuals as a quality gain.'
    (out/'report.json').write_text(json.dumps(report,indent=2));print('head fit candidate ready',flush=True)
bundle=torch.load(ROOT/'avatar_hands_all300_v1/avatar.pt',weights_only=True);bundle['poses']=poses;bundle['scope']='Original unchanged LHM Gaussian rig/features/decoder; fitted body/hands plus candidate multiview face-constrained head/neck. Not a mesh avatar; real quality acceptance pending.';torch.save(bundle,out/'avatar.pt')
