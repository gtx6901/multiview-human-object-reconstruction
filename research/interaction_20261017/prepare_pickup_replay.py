"""Reuse validated frozen avatar rig with new all8 motion and fixed camera framing."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import animate
out=ROOT/'avatar_motion_pickup000_135_v1';out.mkdir(exist_ok=False);bundle=torch.load(ROOT/'avatar_motion180_211_v1/avatar.pt',weights_only=True);poses=torch.load(ROOT/'body_fused_pickup000_135_v1/pose_sequence.pt',weights_only=True)
assert torch.equal(poses['betas'],bundle['poses']['betas']) and torch.equal(poses['scale'],bundle['poses']['scale']);bundle['poses']=poses;bundle['scope']='Same validated fixed-shape rig/appearance, new shared all8 pickup/inspection motion. Zero fingers/expression; neural Gaussian representation, not standard mesh.'
torch.save(bundle,out/'avatar.pt');T=len(poses['frames']);aa=torch.cat([poses['root_pose'][:,None],poses['body_pose'],torch.zeros(T,33,3)],1)
with torch.inference_mode():joints,_=batch_rigid_transform(axis_angle_to_matrix(aa),bundle['rig']['rest_joints'].expand(T,-1,-1),bundle['rig']['parents'])
joints=joints*poses['scale']+poses['translation_world'][:,None]
motion=torch.load(ROOT/'helmet_pose_pickup000_135_v1/sequence.pt',weights_only=True);vertices=motion['vertices_local'];lo=vertices.min(0).values;hi=vertices.max(0).values;corners=torch.tensor([[x,y,z] for x in [float(lo[0]),float(hi[0])] for y in [float(lo[1]),float(hi[1])] for z in [float(lo[2]),float(hi[2])]])
obj=torch.einsum('tij,nj->tni',motion['R'],corners)*motion['logscale'].exp()+motion['translation_world'][:,None]
allpoints=torch.cat([joints,obj],1).numpy();cameras=json.loads((ROOT/'hybrid_frame30_v1/cameras.json').read_text());width=cameras['width'];height=cameras['height'];novel=[]
for original in [cameras['views'][16],cameras['views'][40]]:
    c=dict(original);K=np.array(c['K']);W=np.array(c['W']);pc=allpoints@W[:3,:3].T+W[:3,3];assert (pc[...,2]>.01).all();uv=pc@K.T;uv=(uv[...,:2]/uv[...,2:]).reshape(-1,2);low=uv.min(0);high=uv.max(0);center=(low+high)/2;factor=min((width-70)/(high[0]-low[0]),(height-70)/(high[1]-low[1]));A=np.array([[factor,0,width/2-factor*center[0]],[0,factor,height/2-factor*center[1]],[0,0,1.]])
    c['K']=(A@K).tolist();c['framing']='Fixed across complete motion; all posed joints plus helmet bounding corners with margin.';novel.append(c)
result={'width':width,'height':height,'views':cameras['views'][:8]+novel};(out/'cameras.json').write_text(json.dumps(result,indent=2))
checks=[]
with torch.inference_mode():
    for t in [0,30,68,135]:
        g=animate(bundle['rig'],poses,t);assert all(torch.isfinite(v).all() for v in g.values());checks.append({'frame':t,'xyz_min':g['xyz'].min(0).values.tolist(),'xyz_max':g['xyz'].max(0).values.tolist()})
(out/'report.json').write_text(json.dumps({'success':True,'frames':poses['frames'].tolist(),'checks':checks,'source_rig':'avatar_motion180_211_v1','scope':'Same verified rig and exact shape/scale; new actual all8 fitted motion. Tensor animation finite, no second full LHM reconstruction. New virtual K frames complete clip; all8 real camera matrices unchanged. Fixed hands and head/appearance limits remain.'},indent=2));print('pickup replay prepared',flush=True)
