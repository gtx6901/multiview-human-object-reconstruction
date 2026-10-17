"""Prepare original licensed SMPL-X rig and an independent LBS reference."""
import os
from pathlib import Path
import sys,json,subprocess,time,argparse
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_quaternion
from avatar_rig import pose_axis_angles

parser=argparse.ArgumentParser();parser.add_argument('--out',default='blender_human_animation_v2');options=parser.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False)
model=ROOT.parent/'lhmpp_20261016/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
d=np.load(model,allow_pickle=True);mesh=torch.load(ROOT/'human_mesh_textured_v1/mesh.pt',weights_only=True)
p=torch.load(ROOT/'body_hands_all300_v1/pose_sequence.pt',weights_only=True)
aa=torch.cat([pose_axis_angles(p,t,'cpu') for t in range(300)])
R=axis_angle_to_matrix(aa)
# The initial frame is BENT OVER: its root rotation cannot define world up.
# Upright late-frame head/ankle directions define a constant export gauge.
jworld=p['joints55_world'];up=(jworld[180:,15]-jworld[180:,[7,8]].mean(1)).median(0).values;up=up/up.norm()
forward=(R[180:,0]@torch.tensor([0.,0.,1.])).mean(0);forward=forward-up*(forward@up);forward=forward/forward.norm()
right=torch.linalg.cross(up,forward);right=right/right.norm();forward=torch.linalg.cross(right,up)
F=torch.stack([right,up,forward]);assert torch.allclose(F@F.T,torch.eye(3),atol=1e-5)
R[:,0]=F@R[:,0]
s=p['scale'];center=jworld[180:,[10,11]].reshape(-1,3).median(0).values;J=mesh['rest_joints'];parent=mesh['parents'];weights=mesh['weights'];delta=mesh['offsets_model_space']
translation=((p['translation_world']-center)/s+J[0])@F.T-J[0]
_,A=batch_rigid_transform(R,J[None].expand(300,-1,-1),parent)
samples=[0,42,150,180,299];reference=[]
posedirs=torch.tensor(d['posedirs'],dtype=torch.float32).reshape(-1,486)
coeff=(R[:,1:]-torch.eye(3)).reshape(300,486)
for t in samples:
    T=(weights@A[t].reshape(55,16)).reshape(-1,4,4)
    v=mesh['vertices_template']+delta+(posedirs@coeff[t]).reshape(-1,3)
    v=(T[:,:3,:3]@v[...,None])[...,0]+T[:,:3,3]+translation[t]
    reference.append(v.numpy())
np.savez(out/'animation_data.npz',template=d['v_template'].astype('float32'),faces=d['f'],vt=d['vt'],ft=d['ft'],
    weights=weights.numpy(),joints=J.numpy(),parents=parent.numpy(),offset=delta.numpy(),posedirs=d['posedirs'].astype('float32'),
    quaternion=matrix_to_quaternion(R).numpy(),translation=translation.numpy(),coeff=coeff.numpy(),reference=np.stack(reference),sample_frames=samples,world_gauge_F=F.numpy(),world_center=center.numpy(),body_scale=s.numpy())
job={'out':str(out),'data':str(out/'animation_data.npz'),'texture':str(ROOT/'human_mesh_textured_v1/human_basecolor.png'),
     'official_addon_source':str(ROOT/'vendor/smplx_blender_addon'),'scope':'Custom conversion of the locally licensed original neutral SMPL-X NPZ, not stock official Blender model library. Original weights/rest joints and all486 pose correctives; fitted cloth offsets and fitted300-frame55-joint motion. All8-trained neural texture; no held-out quality claim. Local derived model only.'}
(out/'job.json').write_text(json.dumps(job,indent=2))
runtime=json.loads((ROOT/'blender_runtime_v1/report.json').read_text())
with (out/'blender.log').open('w',encoding='utf-8') as log:
    result=subprocess.run([runtime['executable'],'--background','--factory-startup','--python',str(ROOT/'blender_human_animation.py'),'--',str(out/'job.json')],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
assert result.returncode==0 and (out/'report.json').exists(),'Inspect Blender log'
assert json.loads((out/'report.json').read_text())['success']
print('Animated human asset exported and independently checked',flush=True)
