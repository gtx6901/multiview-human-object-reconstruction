"""Differentiable worn head anchors and visible face constraints for helmet fit."""
import os
from pathlib import Path
import torch,numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks

class JointHeadConstraints:
 def __init__(self,root,avatar,depth,meta,worn):
  self.root=Path(root);self.bundle=torch.load(self.root/avatar/'avatar.pt',weights_only=True);self.base=self.bundle['poses'];self.model=SMPLXSurfaceLandmarks().cuda();self.T=len(self.base['frames']);assert self.T==300
  self.aa0=torch.cat([pose_axis_angles(self.base,t,'cpu') for t in range(self.T)]).cuda();self.scale=self.base['scale'].cuda();self.translation=self.base['translation_world'].cuda();self.beta=self.base['betas'].cuda();self.worn=worn
  self.raw=torch.nn.Parameter(torch.zeros(self.T,2,3,device='cuda'))
  d=np.load(self.root/'body_head_faces_all300_v1/face_screen.npz');valid=d['inlier'].copy();valid[:,:,3:]=False
  # The existing conservative hand region marks face observations as unknown.
  for vi,v in enumerate(meta['views']):
   cache=np.load(self.root/depth/f"{v['serial']}.npz");hand=cache['hand_unknown'];coords=np.floor((d['xy'][vi]+.5)*.125).astype(int)
   x=coords[...,0].clip(0,255);y=coords[...,1].clip(0,191)
   valid[vi]&=hand[np.arange(self.T)[:,None],y,x]==0
  self.xy=torch.tensor(d['xy'],device='cuda',dtype=torch.float32);self.valid=torch.tensor(valid,device='cuda')&worn[None,:,None]
  self.weights=self.valid*torch.tensor(np.minimum(d['scores']/5,1),device='cuda',dtype=torch.float32)
  self.K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);self.W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
  assert int(self.valid.sum())>=100,'Too few visible worn face points for joint fitting'
  with torch.no_grad():self.initial_face_error=float(self.face_loss()[1][self.valid].mean())
 def state(self):
  delta=self.raw.tanh()*torch.tensor([.10,.20],device='cuda')[None,:,None]/3**.5*self.worn[:,None,None]
  aa=self.aa0.clone();aa[:,12]=aa[:,12]+delta[:,0];aa[:,15]=aa[:,15]+delta[:,1]
  result=self.model(aa,self.scale,self.translation,betas=self.beta);rot=axis_angle_to_matrix(aa);head=rot[:,0]
  for j in [3,6,9,12,15]:head=head@rot[:,j]
  return head,result['joints55'][:,15],aa,result,delta
 def face_loss(self):
  _,_,_,r,delta=self.state();pc=torch.einsum('vij,tkj->vtki',self.W[:,:3,:3],r['coco17'][:,:5])+self.W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',self.K,pc);uv=q[...,:2]/q[...,2:].clamp_min(.01);error=(uv-self.xy).norm(dim=-1)
  loss=(2*(torch.sqrt(1+(error/8)**2)-1)*self.weights).sum()/self.weights.sum().clamp_min(1)
  regularizer=.2*delta.square().mean()+.1*(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()
  return loss+regularizer,error
 def save(self,out):
  with torch.no_grad():
   _,_,aa,r,_=self.state();poses={**self.base,'body_pose':aa[:,1:22].cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu()}
   torch.save(poses,out/'body_pose_sequence.pt');torch.save({**self.bundle,'poses':poses,'scope':'Joint worn head/helmet candidate with original shared LHM appearance; fixed-cache visibility approximation, unadopted.'},out/'avatar.pt')
  with torch.no_grad():after=float(self.face_loss()[1][self.valid].mean());change=float((poses['body_pose']-self.base['body_pose']).abs().max())
  return {'usable_face_points':int(self.valid.sum()),'optimized_frames':int(self.worn.sum()),'before_face_error_native_px':self.initial_face_error,'after_face_error_native_px':after,'max_pose_component_change_rad':change,'face_scope':'Native COCO nose/eyes from existing3view RANSAC screen, ears excluded, hand-cache overlap unknown; generic surface anchors can retain bias. Fixed torso/shape/fingers, only worn neck/head change. Existing body depth is held fixed in this inner iteration and must be regenerated for outer visibility update.'}
