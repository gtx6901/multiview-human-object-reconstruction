"""Phase-specific visibility prior for puffy head splats outside a worn helmet.

Uses the CLOSED outer hull, never the thin shell's solid occupancy. Canonical
geometry/features remain unchanged. Eyes/nose have explicit protected regions.
"""
import os
from pathlib import Path
import json
import numpy as np,torch,trimesh
from scipy.ndimage import binary_dilation
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import pose_axis_angles
class WornHeadEnvelope:
 def __init__(self,root,rig,poses,motion,shell_source):
  root=Path(root);report=json.loads((root/shell_source/'report.json').read_text());closed_source=report.get('source_motion',report.get('source_shape'));assert closed_source;source=torch.load(root/closed_source/'sequence.pt',weights_only=True);assert torch.allclose(motion['logscale'].cpu(),source['logscale'])
  hull=trimesh.Trimesh(source['vertices_local'].numpy(),source['faces'].numpy(),process=False);assert hull.is_watertight
  if hull.volume<0:hull.invert()
  scale=float(source['logscale'].exp());before=hull.vertices.copy();trimesh.smoothing.filter_taubin(hull,lamb=.5,nu=.53,iterations=report.get('smoothing_iterations',report.get('smoothing',{}).get('iterations',30)));delta=hull.vertices-before;size=np.linalg.norm(delta,axis=1)*scale;hull.vertices=before+delta*np.minimum(1,.005/np.maximum(size,1e-12))[:,None]
  vox=hull.voxelized(float(np.ptp(hull.vertices,axis=0).max()/96)).fill();volume=binary_dilation(vox.matrix,iterations=1);device=rig['weights'].device;self.volume=torch.tensor(volume,device=device);self.transform=torch.tensor(np.linalg.inv(vox.transform),device=device,dtype=torch.float32);self.normal=torch.tensor(report['opening_plane_normal_local'],device=device,dtype=torch.float32);self.level=float(report['opening_level_local']);self.ids=torch.where(rig['weights'][:,15]>.6)[0];assert len(self.ids)>100
  self.motion=motion;self.scale=motion['logscale'].exp();self.human_scale=poses['scale'];self.shape=torch.tensor(volume.shape,device=device);self.local_face=torch.tensor(np.load(root/'lhm_face_anchor_correspondence_v1/anchors.npz')['canonical_face_local'][:3],device=device,dtype=torch.float32)
  selection=report.get('selected',{});self.side=float(selection.get('side_drop',0));self.back=float(selection.get('back_drop',0));self.hull_center=torch.tensor(hull.center_mass,device=device,dtype=torch.float32);self.up=torch.tensor(report['opening_up_local'],device=device,dtype=torch.float32);self.up=torch.nn.functional.normalize(self.up,dim=0);self.front=torch.tensor(report['opening_front_local'],device=device,dtype=torch.float32);self.front=torch.nn.functional.normalize(self.front-self.up*(self.front@self.up),dim=0)
  aa=torch.cat([pose_axis_angles(poses,t,device) for t in range(len(poses['frames']))]);R=axis_angle_to_matrix(aa);joints,_=batch_rigid_transform(R,rig['rest_joints'].expand(len(aa),-1,-1),rig['parents']);head=R[:,0]
  for j in [3,6,9,12,15]:head=head@R[:,j]
  self.face=(joints[:,15]*poses['scale']+poses['translation_world'])[:,None]+torch.einsum('tij,kj->tki',head,self.local_face)*poses['scale'];self.records=[]
  self.scope={'shell_source':shell_source,'closed_source':closed_source,'head_weight_min':.6,'grid_longest_axis_cells':96,'outer_margin_voxels':1,'face_protection_radius_model_units':.035,'wear_interval':[180,300],'scope':'Visibility-only anatomical/helmet prior: above recorded opening plane and outside dilated CLOSED outer hull, only head-dominant Gaussians. Protect nose/eye neighborhoods; no canonical/feature/pose edit. Applies180..299 only. Not recovered hair compression or head shape, and extended splat/decoder effects still require rendered review.'}
 def apply(self,frame,g):
  if frame<180 or frame>=300:return g
  points=g['xyz'][self.ids];local=(points-self.motion['translation_world'][frame])@self.motion['R'][frame]/self.scale;index=torch.round(local@self.transform[:3,:3].T+self.transform[:3,3]).long();valid=((index>=0)&(index<self.shape)).all(-1);safe=torch.minimum(index.clamp_min(0),self.shape-1);inside=valid&self.volume[safe[:,0],safe[:,1],safe[:,2]];field=local@self.normal-self.level
  if self.side or self.back:
   q=local-self.hull_center;q=q-(q@self.up)[:,None]*self.up;cosine=((q@self.front)/q.norm(dim=1).clamp_min(1e-10)).clamp(-1,1);field=field+self.side*(1-cosine.square())+self.back*torch.relu(-cosine).square()
  above=field>0;protected=(torch.cdist(points,self.face[frame])/self.human_scale).amin(-1)<.035;drop=above&~inside&~protected
  opacity=g['opacity'].clone();opacity[self.ids[drop]]=0;self.records.append({'frame':int(frame),'head_candidates':len(self.ids),'above_opening':int(above.sum()),'outside_outer_hull':int((~inside).sum()),'protected_face_points':int(protected.sum()),'culled_gaussians':int(drop.sum())});return {**g,'opacity':opacity}
