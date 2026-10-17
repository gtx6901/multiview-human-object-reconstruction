"""Optional weak contact prior from corroborated, fitted palm/fingertip points.

This is deliberately separate from collision/nonpenetration: a point-cloud
surface distance does not certify that a hand is outside a thin helmet shell.
"""
import os
from pathlib import Path
import torch,numpy as np,json,trimesh
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
class HandHelmetContact:
 def __init__(self,root,avatar,vertices,faces,object_scale,R,t):
  root=Path(root);b=torch.load(root/avatar/'avatar.pt',weights_only=True);p=b['poses'];self.avatar=avatar;self.human_scale=p['scale'].cuda();self.object_scale=object_scale;assert p['frames'].tolist()==list(range(300))
  screen=np.load(root/'hands_multiview_screen_all300_v1/screened.npz');assert screen['frames'].tolist()==list(range(300))
  model=SMPLXSurfaceLandmarks().cuda();aa=torch.cat([pose_axis_angles(p,f,'cpu') for f in range(300)]).cuda()
  with torch.no_grad():hands=model(aa,self.human_scale,p['translation_world'].cuda(),betas=p['betas'].cuda())['hands21']
  meta=json.loads((root/'helmet_native000_299_v1/manifest.json').read_text());K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
  flat=hands.reshape(300,42,3);cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],flat)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,cam);xy=q[...,:2]/q[...,2:].clamp_min(.01);valid=torch.tensor(screen['inlier'],device='cuda');error=(xy-torch.tensor(screen['xy'],device='cuda')).norm(dim=-1)
  fit_error=(error*valid).sum(0)/valid.sum(0).clamp_min(1);known=((valid.sum(0)>=3)&(fit_error<30)).reshape(300,2,21)
  ids=[4,8,12,16,20];mcp=[0,5,9,13,17];points=torch.cat([hands[:,:,ids],hands[:,:,mcp].mean(2,keepdim=True)],2);good=torch.cat([known[:,:,ids],(known[:,:,mcp].sum(-1)>=3)[...,None]],2)
  self.points=points.reshape(300,12,3).detach();self.good=good.reshape(300,12);tm=trimesh.Trimesh(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy(),process=False);samples,_=trimesh.sample.sample_surface(tm,8192,seed=2054);self.surface=torch.tensor(samples,device='cuda',dtype=torch.float32)
  frames=torch.arange(300,device='cuda');self.frames=torch.where((frames>=30)&(frames<136)&self.good.any(-1))[0]
  with torch.no_grad():d=self.distances(R,t);keep=d<.08;self.frames=self.frames[keep];self.before=self.distances(R,t).detach()
  assert len(self.frames)>=10,'Too few reliable near-contact held frames'
 def distances(self,R,t):
  f=self.frames;local=torch.einsum('tnj,tjk->tnk',self.points[f]-t[f,None],R[f])/self.object_scale
  distance=torch.cdist(local,self.surface[None],compute_mode='use_mm_for_euclid_dist').amin(-1)*self.object_scale/self.human_scale
  return distance.masked_fill(~self.good[f],float('inf')).amin(-1)
 def loss(self,R,t):
  d=self.distances(R,t);return torch.nn.functional.smooth_l1_loss(torch.relu(d-.01)/.03,torch.zeros_like(d),beta=1.)
 def report(self,R,t):
  with torch.no_grad():after=self.distances(R,t)
  return {'frames':self.frames.cpu().tolist(),'reliable_near_contact_frames':len(self.frames),'before_mean_gap_model_units':float(self.before.mean()),'after_mean_gap_model_units':float(after.mean()),'before_fraction_within_015_model_units':float((self.before<=.015).float().mean()),'after_fraction_within_015_model_units':float((after<=.015).float().mean()),'surface_samples':len(self.surface),'source_avatar':self.avatar,'scope':'Weak at-least-one palm/fingertip contact prior for held30..135, only points corroborated in>=3 cameras and fitted reprojection mean<30native px. Fixed initial frame set further requires some hand point within.08 human-model units. Distance to8192 deterministic area samples of the full shell, tolerance.01 model units, normalized by actual avatar scale. Coarse correspondence, finite surface sampling and hand fitting errors remain; this is not measured physical contact, a metric-unit statement or a nonpenetration/collision constraint. Frames without corroborated nearby hands remain unknown.'}
