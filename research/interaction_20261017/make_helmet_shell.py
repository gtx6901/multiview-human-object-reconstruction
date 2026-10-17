"""Offline opening + inward solidification prior, explicitly not cavity recovery."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_shell_all300_v5');p.add_argument('--smooth-iterations',type=int,default=30);p.add_argument('--pose-source',default='helmet_shell_pose_all300_v1');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
motion=torch.load(ROOT/'helmet_phased_all300_v3/sequence.pt',weights_only=True);body=torch.load(ROOT/'body_hands_all300_v1/pose_sequence.pt',weights_only=True)
mesh=trimesh.Trimesh(motion['vertices_local'].numpy(),motion['faces'].numpy(),process=False)
was_inward=mesh.volume<0
if was_inward:mesh.invert()
before_vertices=np.array(mesh.vertices).copy();before_volume=float(mesh.volume)
if options.smooth_iterations:trimesh.smoothing.filter_taubin(mesh,lamb=.5,nu=.53,iterations=options.smooth_iterations)
delta=np.array(mesh.vertices)-before_vertices;unclamped=np.linalg.norm(delta,axis=1)*float(motion['logscale'].exp())
delta*=np.minimum(1,.005/np.maximum(unclamped,1e-12))[:,None]
mesh.vertices=before_vertices+delta
displacement=np.linalg.norm(mesh.vertices-before_vertices,axis=1)*float(motion['logscale'].exp())
assert displacement.max()<.01,'Smoothing moved the measured outer shape too far'
up=body['joints55_world'][180:,15].numpy()-body['joints55_world'][180:,12].numpy();up/=np.linalg.norm(up,axis=1)[:,None]
localup=np.einsum('tji,tj->ti',motion['R'][180:].numpy(),up);localup=np.median(localup,axis=0);localup/=np.linalg.norm(localup)
eyes=body['joints55_world'][180:,[23,24]].mean(1).numpy();head=body['joints55_world'][180:,15].numpy()
front=eyes-head;front-=(front*up).sum(1)[:,None]*up;front/=np.linalg.norm(front,axis=1)[:,None]
localfront=np.median(np.einsum('tji,tj->ti',motion['R'][180:].numpy(),front),axis=0);localfront-=np.dot(localfront,localup)*localup;localfront/=np.linalg.norm(localfront)
normal=localup-.6*localfront;normal/=np.linalg.norm(normal)
# Explicit half-helmet prior: the front opening must clear the eye line.
# Tilt the opening upward at the forehead, retaining lower posterior coverage.
clearance=.012
opening_world=eyes+clearance*up
opening_local=np.einsum('tji,tj->ti',motion['R'][180:].numpy(),opening_world-motion['translation_world'][180:].numpy())/float(motion['logscale'].exp())
level=float(np.median(opening_local@normal));origin=normal*level
cut_vertices,cut_faces,_=trimesh.intersections.slice_faces_plane(vertices=mesh.vertices,faces=mesh.faces,plane_normal=normal,plane_origin=origin)
outer=trimesh.Trimesh(cut_vertices,cut_faces,process=False)
outer.merge_vertices(digits_vertex=10);outer.remove_unreferenced_vertices();assert len(outer.faces)>100 and not outer.is_watertight
v=np.array(outer.vertices);f=np.array(outer.faces);N=len(v);thickness_world=.001;thickness_local=thickness_world/float(motion['logscale'].exp())
normals=np.array(outer.vertex_normals);inner=v-thickness_local*normals
edges=outer.edges;sorted_edges=np.sort(edges,axis=1);_,inverse,counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True);boundary=edges[counts[inverse]==1];assert len(boundary)>2
rim=np.concatenate([np.stack([boundary[:,1],boundary[:,0],boundary[:,0]+N],1),np.stack([boundary[:,1],boundary[:,0]+N,boundary[:,1]+N],1)])
shellfaces=np.concatenate([f,f[:,::-1]+N,rim]);shell=trimesh.Trimesh(np.concatenate([v,inner]),shellfaces,process=False)
assert shell.is_watertight and shell.is_winding_consistent and shell.volume>0
shell.export(out/'helmet_shell.obj')
pose=torch.load(ROOT/options.pose_source/'sequence.pt',weights_only=True)
assert torch.equal(pose['frames'],motion['frames']) and torch.equal(pose['logscale'],motion['logscale'])
torch.save({**motion,'R':pose['R'],'translation_world':pose['translation_world'],'vertices_local':torch.tensor(np.array(shell.vertices),dtype=torch.float32),'faces':torch.tensor(np.array(shell.faces),dtype=torch.int32)},out/'sequence.pt')
face_classes=np.r_[np.zeros(len(f),np.int32),np.ones(len(f),np.int32),np.full(len(rim),2,np.int32)]
torch.save({'vertices_local':torch.tensor(np.array(shell.vertices),dtype=torch.float32),'faces':torch.tensor(np.array(shell.faces),dtype=torch.int32),'colors':torch.full((len(shell.vertices),3),.65),'face_classes':torch.tensor(face_classes),'outer_vertex_count':N},out/'mesh.pt')
report={'success':True,'source_motion':'helmet_phased_all300_v3','seconds':time.time()-start,'vertices':len(shell.vertices),'faces':len(shell.faces),'watertight':bool(shell.is_watertight),'winding_consistent':bool(shell.is_winding_consistent),'source_winding_inverted':bool(was_inward),'boundary_edges_joined':len(boundary),'opening_up_local':localup.tolist(),'opening_level_local':level,'cut_fraction_of_height':.18,'thickness_normalized_world_units':thickness_world,'thickness_local':thickness_local,'scope':'EXPLICIT GEOMETRIC PRIOR: infer a nominal downward opening from worn head/neck direction, remove lower18% of closed-hull axial extent without capping, offset inward along vertex normals and join rims. Analogous to offline clipping/Solidify, not measured inner cavity or calibrated metric thickness. Preserves outer retained surface. Closed shell connectivity and winding checked; self-intersections, anatomical fit and true thickness not proven. Requires render/real-mask audit before acceptance. No recovered color/UV yet; private motion tensors not public licensed assets.'}
report.pop('cut_fraction_of_height',None)
report.update(opening_plane_normal_local=normal.tolist(),opening_front_local=localfront.tolist(),eye_line_clearance_normalized_world=clearance)
report['smoothing']={'iterations':options.smooth_iterations,'method':'Taubin lambda.5 nu.53 on closed outer hull BEFORE opening/thickening','mean_displacement_normalized_world':float(displacement.mean()),'max_displacement_normalized_world':float(displacement.max()),'closed_volume_ratio':float(mesh.volume/before_volume)}
report['smoothing'].update(unclamped_max_displacement=float(unclamped.max()),per_vertex_displacement_cap=.005)
report['pose_source']=options.pose_source
report['scope']='EXPLICIT GEOMETRIC PRIOR for the observed open-face half-helmet: Taubin smooth voxel stair artifacts first, use fitted worn head/neck/eye directions, tilt a nominal opening plane upward at the forehead (normal=up-.6front), place it above the eye line by.012 normalized-world units, cut without capping, offset inward and join rims. V2 arbitrary lower18% cut left excessive facial coverage and is superseded as a candidate. This is an anatomical/design prior, not observed cavity geometry or calibrated metric thickness. Retained outer surface changes only by measured smoothing offsets; closed shell connectivity/winding checked, self-intersections/real-image fit/anatomical clearance still require audit. No recovered UV/color.'
(out/'report.json').write_text(json.dumps(report,indent=2));print('offline helmet shell generated; prior requires visual QA',flush=True)
