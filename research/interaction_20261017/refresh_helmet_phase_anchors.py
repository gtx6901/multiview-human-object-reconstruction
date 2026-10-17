"""Recompute phase anchors from the actual current shared avatar motion."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from avatar_rig import pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.lbs import batch_rigid_transform
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',required=True);p.add_argument('--phases',default='helmet_phase_head_faces_all300_v1');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);poses=b['poses'];aa=torch.cat([pose_axis_angles(poses,t,'cpu') for t in range(len(poses['frames']))]);local=axis_angle_to_matrix(aa);parents=b['rig']['parents'].cpu();rest=b['rig']['rest_joints'].cpu().expand(len(aa),-1,-1)
joints,_=batch_rigid_transform(local,rest,parents);world=joints*poses['scale']+poses['translation_world'][:,None];globalR=[]
for j,parent in enumerate(parents.tolist()):globalR.append(local[:,j] if parent<0 else globalR[parent]@local[:,j])
torch.save({'global_joint_R':torch.stack(globalR,1),'joints_world':world,'frames':poses['frames']},out/'anchors.pt');old=json.loads((ROOT/o.phases/'report.json').read_text());report={**old,'sources':{'avatar':o.avatar,'phase_intervals':o.phases},'scope':'Original phase intervals retained; all joint rotations/positions recomputed from actual saved LHM rig rest/beta-consistent motion. No new phase/contact truth inferred.'};(out/'report.json').write_text(json.dumps(report,indent=2));print('phase anchors refreshed from current avatar')
