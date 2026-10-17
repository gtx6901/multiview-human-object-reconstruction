"""Refit coherent body/head motion with fixed canonical appearance and beta.

Unlike a torso-only shape fit, this includes actual-LHM semantic face anchors
in the SAME motion objective, so root/spine changes cannot silently lose face
alignment. No object pose is updated by this block.
"""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from joint_head_constraints import JointHeadConstraints
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='shared_shape_accepted_joint_motion_all300_v2');p.add_argument('--face-anchors',default='shared_shape_head_face_anchors_v2');p.add_argument('--face-weight',type=float,default=.25);p.add_argument('--steps',type=int,default=400);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);bundle=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);base=bundle['poses'];T=len(base['frames']);assert T==300
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());guard=JointHeadConstraints(ROOT,o.avatar,'shared_shape_head_occlusion_all300_v2',meta,torch.ones(300,device='cuda',dtype=torch.bool),o.face_anchors,True)
face_local=guard.local_face.detach();face_valid=guard.valid.detach();face_weight=guard.weights.detach();native_K=guard.K.detach();W=guard.W.detach();face_xy=guard.xy.detach();del guard
model=SMPLXSurfaceLandmarks().cuda();aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(T)]).cuda();root=torch.nn.Parameter(aa0[:,0].clone());body=torch.nn.Parameter(aa0[:,1:22].clone());trans=torch.nn.Parameter(base['translation_world'].cuda().clone());beta=base['betas'].cuda();scale=base['scale'].cuda()
A=torch.tensor([[.5,0,-.25],[0,.5,-.25],[0,0,1]],device='cuda');K=A@native_K
d=np.load(ROOT/'observations_all300_v1/body.npz');xy=torch.tensor(d['xy'][:,:,5:17],device='cuda',dtype=torch.float32);valid=torch.tensor((d['score'][:,:,5:17]>3)&(d['xy'][:,:,5:17,0]>2)&(d['xy'][:,:,5:17,0]<1022)&(d['xy'][:,:,5:17,1]>2)&(d['xy'][:,:,5:17,1]<766),device='cuda')
h=np.load(ROOT/'hands_multiview_screen_all300_v1/screened.npz');hxy=torch.tensor(h['xy']*.5-.25,device='cuda',dtype=torch.float32);hw=torch.tensor(np.minimum(h['raw_score']/5,1)*h['inlier'],device='cuda',dtype=torch.float32)
def project(points,k):
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],points)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',k,pc);return q[...,:2]/q[...,2:].clamp_min(.01)
def evaluate():
 aa=torch.cat([root[:,None],body,aa0[:,22:]],1);r=model(aa,scale,trans,betas=beta);rot=axis_angle_to_matrix(aa);head=rot[:,0]
 for j in [3,6,9,12,15]:head=head@rot[:,j]
 face=r['joints55'][:,15,None]+torch.einsum('tij,kj->tki',head,face_local)*scale;fe=(project(face,native_K)-face_xy).norm(dim=-1);be=(project(r['coco17'][:,5:17],K)-xy).norm(dim=-1);he=(project(r['hands21'].reshape(T,42,3),K)-hxy).norm(dim=-1)
 robust=lambda e,c:2*(torch.sqrt(1+(e/c)**2)-1)
 data=(robust(be,6)*valid).sum()/valid.sum()+.3*(robust(he,6)*hw).sum()/hw.sum()+o.face_weight*(robust(fe,8)*face_weight).sum()/face_weight.sum()
 acc=(r['joints55'][2:]-2*r['joints55'][1:-1]+r['joints55'][:-2])*30**2
 prior=.1*(body-aa0[:,1:22]).square().mean()+.1*(root-aa0[:,0]).square().mean()+.001*acc.square().mean()+.02*(body[2:]-2*body[1:-1]+body[:-2]).square().mean()
 return data+prior,be,he,fe,r
opt=torch.optim.Adam([{'params':[root,body],'lr':.002},{'params':[trans],'lr':.0005}]);best=None;history=[]
with torch.no_grad():_,bb,bh,bf,_=evaluate();bb=bb.clone();bh=bh.clone();bf=bf.clone()
for step in range(o.steps+1):
 opt.zero_grad();loss,be,he,fe,r=evaluate();value=float(loss.detach())
 if best is None or value<best[0]:best=(value,[x.detach().clone() for x in [root,body,trans]])
 if step%100==0 or step==o.steps:history.append({'step':step,'loss':value,'body_px1024':float(be[valid].mean().detach()),'hand_px1024':float(((he*hw).sum()/hw.sum()).detach()),'face_native_px':float(fe[face_valid].mean().detach())});print(json.dumps(history[-1]),flush=True)
 if step<o.steps:loss.backward();opt.step()
with torch.no_grad():
 for x,y in zip([root,body,trans],best[1]):x.copy_(y)
 _,be,he,fe,r=evaluate();poses={**base,'root_pose':root.cpu(),'body_pose':body.cpu(),'translation_world':trans.cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu(),'projected':project(r['coco17'][:,5:17],K).cpu()};assert torch.equal(poses['betas'],base['betas']) and torch.equal(poses['scale'],base['scale'])
 for k in ['left_hand_pose','right_hand_pose']:assert torch.equal(poses[k],base[k])
 torch.save(poses,out/'pose_sequence.pt');torch.save({**bundle,'poses':poses,'scope':'Canonical appearance, shaped rig, beta/scale and fingers unchanged; coherent body/root/head motion refit to body, hand and actual-LHM semantic face evidence. Object motion not yet refit; unadopted.'},out/'avatar.pt')
 phase={}
 for label,a,b in [('clean',0,136),('transition',136,180),('worn',180,300)]:
  fv=face_valid[:,a:b];phase[label]={'face_points':int(fv.sum()),'before_native_px':float(bf[:,a:b][fv].mean()),'after_native_px':float(fe[:,a:b][fv].mean())}
 result={'success':True,'source_avatar':o.avatar,'face_anchor_source':o.face_anchors,'face_weight':o.face_weight,'history':history,'body_before_px1024':float(bb[valid].mean()),'body_after_px1024':float(be[valid].mean()),'hand_before_px1024':float((bh*hw).sum()/hw.sum()),'hand_after_px1024':float((he*hw).sum()/hw.sum()),'face_by_phase':phase,'shared_beta_norm':float(beta.norm()),'seconds':time.time()-start,'scope':'All8x300 coherent body/head/root/translation fit with fixed actual canonical appearance, beta/scale, finger motions and cameras. Body surface COCO12, screened hands and identity-checked actual-LHM face anchors contribute jointly. Face front/hand screening freezes initial set; not true visibility, and fixed detector definitions can retain bias. Input error changes are not accepted image/contact quality; object motion and depth must refresh after changed body motion. Compare same-init face-weight0 control and actual rendered evidence before adoption.'};(out/'report.json').write_text(json.dumps(result,indent=2));print('coherent body/face motion candidate complete',flush=True)
