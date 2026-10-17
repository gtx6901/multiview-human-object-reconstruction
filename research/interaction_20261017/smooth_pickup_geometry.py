"""Surface discretization diagnostic; preserve motion and vertex correspondence."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
out=ROOT/'helmet_pose_pickup_smoothed_v1';out.mkdir(exist_ok=False);data=torch.load(ROOT/'helmet_pose_pickup000_135_v1/sequence.pt',weights_only=True)
original=data['vertices_local'].numpy();mesh=trimesh.Trimesh(original.copy(),data['faces'].numpy(),process=False);before=float(abs(mesh.volume));trimesh.smoothing.filter_taubin(mesh,lamb=.5,nu=.53,iterations=30);vertices=np.array(mesh.vertices);assert np.isfinite(vertices).all();after=float(abs(mesh.volume));assert .8<after/before<1.2
data['vertices_local']=torch.tensor(vertices,dtype=torch.float32);torch.save(data,out/'sequence.pt');mesh.export(out/'local_candidate.obj');displacement=np.linalg.norm(vertices-original,axis=1)*float(data['logscale'].exp())
(out/'report.json').write_text(json.dumps({'success':True,'source_motion':'helmet_pose_pickup000_135_v1','filter':'Trimesh Taubin lambda=.5 nu=.53 iterations=30','vertices':len(vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'volume_ratio_after_before':after/before,'mean_world_vertex_displacement':float(displacement.mean()),'max_world_vertex_displacement':float(displacement.max()),'scope':'Only smooth voxel/discretization ripples of the same silhouette hull; fixed motion/scale retained without pose refitting. No cavity, texture, contact or true shape recovery. Preview uses explicitly neutral diagnostic material.'},indent=2))
