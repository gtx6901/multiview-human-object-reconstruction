"""Preflight the actual closed hull's nonplanar opening family."""
import os
from pathlib import Path
import sys,json,itertools
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
from helmet_opening_profile import cut_opening
out=ROOT/'helmet_profile_geometry_controls_v1';out.mkdir(exist_ok=False);r=json.loads((ROOT/'helmet_shell_all300_v5/report.json').read_text());d=torch.load(ROOT/r['source_motion']/'sequence.pt',weights_only=True);scale=float(d['logscale'].exp());closed=trimesh.Trimesh(d['vertices_local'].numpy(),d['faces'].numpy(),process=False)
if closed.volume<0:closed.invert()
before=closed.vertices.copy();trimesh.smoothing.filter_taubin(closed,lamb=.5,nu=.53,iterations=r['smoothing']['iterations']);delta=closed.vertices-before;size=np.linalg.norm(delta,axis=1)*scale;closed.vertices=before+delta*np.minimum(1,.005/np.maximum(size,1e-12))[:,None];n=np.array(r['opening_plane_normal_local']);origin=n*r['opening_level_local'];up=np.array(r['opening_up_local']);front=np.array(r['opening_front_local']);baseline=cut_opening(closed,n,origin,up,front);rows=[]
for side,back in itertools.product([0.,.1,.2],repeat=2):
 outer=cut_opening(closed,n,origin,up,front,side,back);v=outer.vertices.copy();f=outer.faces.copy();N=len(v);inner=v-.001/scale*outer.vertex_normals;edges=outer.edges;_,idx,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True);edge=edges[count[idx]==1];rim=np.concatenate([np.stack([edge[:,1],edge[:,0],edge[:,0]+N],1),np.stack([edge[:,1],edge[:,0]+N,edge[:,1]+N],1)]);shell=trimesh.Trimesh(np.concatenate([v,inner]),np.concatenate([f,f[:,::-1]+N,rim]),process=False);assert shell.is_watertight and shell.is_winding_consistent and shell.volume>0;assert outer.area>=baseline.area-1e-8
 rows.append({'side_drop':side,'back_drop':back,'outer_area_ratio_to_planar':float(outer.area/baseline.area),'watertight':True,'consistent_winding':True,'positive_shell_volume':True})
(out/'report.json').write_text(json.dumps({'success':True,'candidates':rows,'scope':'Actual smoothed closed source hull with9 shared side/back drop combinations; open surface connected, positive watertight thin shell, and no reduction of retained outer area. Zero parameters use original plane code. Topology/implementation controls only, not observed helmet geometry or quality acceptance.'},indent=2));print('nonplanar opening9 actual-hull geometry controls passed',flush=True)
