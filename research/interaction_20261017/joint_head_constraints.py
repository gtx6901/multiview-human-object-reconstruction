"""Differentiable worn head anchors and visible face constraints for helmet fit."""
import os
from pathlib import Path
import torch,numpy as np,json
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks

class JointHeadConstraints:
 def __init__(self,root,avatar,depth,meta,worn,face_anchors=None,front_visibility=False):
  self.root=Path(root);self.bundle=torch.load(self.root/avatar/'avatar.pt',weights_only=True);self.base=self.bundle['poses'];self.model=SMPLXSurfaceLandmarks().cuda();self.T=len(self.base['frames']);assert self.T==300
  self.aa0=torch.cat([pose_axis_angles(self.base,t,'cpu') for t in range(self.T)]).cuda();self.scale=self.base['scale'].cuda();self.translation=self.base['translation_world'].cuda();self.beta=self.base['betas'].cuda();self.worn=worn
  self.raw=torch.nn.Parameter(torch.zeros(self.T,2,3,device='cuda'))
  self.rest=self.bundle['rig']['rest_joints'].cuda().expand(self.T,-1,-1);self.parents=self.bundle['rig']['parents'].cuda();self.local_face=None;self.face_anchor_source=face_anchors
  expected=self.model.rest+torch.einsum('bl,jcl->bjc',self.beta,self.model.joint_shapedirs)[0];assert torch.allclose(self.rest[0],expected,atol=1e-6)
  if face_anchors:
   report=json.loads((self.root/face_anchors/'report.json').read_text());assert report['success'] and all(p['usable'] and p['support']>=3 for p in report['points'])
   reference=torch.load(self.root/'body_head_faces_all300_v1/avatar.pt',weights_only=True)
   for key in ['zero_coords','weights','rest_joints']:assert torch.equal(self.bundle['rig'][key],reference['rig'][key]),'Regenerate LHM face anchors for changed canonical geometry'
   assert torch.equal(self.bundle['features'],reference['features']) and self.beta.abs().max()==0,'Regenerate anchors for changed appearance or shape'
   self.local_face=torch.tensor(np.load(self.root/face_anchors/'anchors.npz')['canonical_face_local'],device='cuda',dtype=torch.float32)
  d=np.load(self.root/'body_head_faces_all300_v1/face_screen.npz');valid=d['inlier'].copy();valid[:,:,3:]=False
  # The existing conservative hand region marks face observations as unknown.
  for vi,v in enumerate(meta['views']):
   cache=np.load(self.root/depth/f"{v['serial']}.npz");hand=cache['hand_unknown'];coords=np.floor((d['xy'][vi]+.5)*.125).astype(int)
   x=coords[...,0].clip(0,255);y=coords[...,1].clip(0,191)
   valid[vi]&=hand[np.arange(self.T)[:,None],y,x]==0
  self.before_visibility_points=int((torch.tensor(valid,device='cuda')&worn[None,:,None]).sum());self.front_visibility=front_visibility
  if front_visibility:
   with torch.no_grad():
    head,t,_,_,_=self.state(surface=False)
    local=(self.local_face[0]-self.local_face[1:3].mean(0)) if self.local_face is not None else (self.rest[0,[23,24]].mean(0)-self.rest[0,15]);local=local.clone();local[1]=0;local=torch.nn.functional.normalize(local,dim=0)
    direction=torch.einsum('tij,j->ti',head,local);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32);centers=-torch.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);to_camera=torch.nn.functional.normalize(centers[:,None]-t[None],dim=-1);front=(to_camera*direction[None]).sum(-1)>.15
    valid&=front.cpu().numpy()[...,None]
  self.xy=torch.tensor(d['xy'],device='cuda',dtype=torch.float32);self.valid=torch.tensor(valid,device='cuda')&worn[None,:,None]
  self.weights=self.valid*torch.tensor(np.minimum(d['scores']/5,1),device='cuda',dtype=torch.float32)
  self.K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);self.W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
  assert int(self.valid.sum())>=100,'Too few visible worn face points for joint fitting'
  with torch.no_grad():self.initial_face_error=float(self.face_loss()[1][self.valid].mean())
 def state(self,surface=True):
  delta=self.raw.tanh()*torch.tensor([.10,.20],device='cuda')[None,:,None]/3**.5*self.worn[:,None,None]
  aa=self.aa0.clone();aa[:,12]=aa[:,12]+delta[:,0];aa[:,15]=aa[:,15]+delta[:,1]
  rot=axis_angle_to_matrix(aa);head=rot[:,0]
  if surface:result=self.model(aa,self.scale,self.translation,betas=self.beta)
  else:
   joints,_=batch_rigid_transform(rot,self.rest,self.parents);result={'joints55':joints*self.scale+self.translation[:,None]}
  for j in [3,6,9,12,15]:head=head@rot[:,j]
  return head,result['joints55'][:,15],aa,result,delta
 def face_loss(self):
  head,t,_,r,delta=self.state(surface=self.local_face is None)
  points=r['coco17'][:,:5] if self.local_face is None else t[:,None]+torch.einsum('bij,kj->bki',head,self.local_face)*self.scale
  pc=torch.einsum('vij,tkj->vtki',self.W[:,:3,:3],points)+self.W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',self.K,pc);uv=q[...,:2]/q[...,2:].clamp_min(.01);error=(uv-self.xy).norm(dim=-1)
  loss=(2*(torch.sqrt(1+(error/8)**2)-1)*self.weights).sum()/self.weights.sum().clamp_min(1)
  regularizer=.2*delta.square().mean()+.1*(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()
  return loss+regularizer,error
 def save(self,out):
  with torch.no_grad():
   _,_,aa,r,_=self.state();poses={**self.base,'body_pose':aa[:,1:22].cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu()}
   torch.save(poses,out/'body_pose_sequence.pt');torch.save({**self.bundle,'poses':poses,'scope':'Joint worn head/helmet candidate with original shared LHM appearance; fixed-cache visibility approximation, unadopted.'},out/'avatar.pt')
  with torch.no_grad():after=float(self.face_loss()[1][self.valid].mean());change=float((poses['body_pose']-self.base['body_pose']).abs().max())
  return {'usable_face_points':int(self.valid.sum()),'points_before_directional_visibility':self.before_visibility_points,'front_visibility_enabled':self.front_visibility,'frames_without_face_support':int((self.worn&~self.valid.any((0,2))).sum()),'optimized_frames':int(self.worn.sum()),'before_face_error_native_px':self.initial_face_error,'after_face_error_native_px':after,'max_pose_component_change_rad':change,'face_anchor_source':self.face_anchor_source or 'generic SMPL-X surface','face_scope':'Native COCO nose/eyes from existing3view RANSAC screen, ears excluded, hand-cache overlap unknown. Optional directional visibility freezes the initial front-camera set during this inner fit, so optimization cannot hide residuals by turning away. Anchor/visibility definitions recorded separately; metrics across different definitions are not comparable quality evidence. Fixed torso/shape/fingers, only worn neck/head change. Existing body depth is held fixed in this inner iteration and must be regenerated for outer visibility update.'}
