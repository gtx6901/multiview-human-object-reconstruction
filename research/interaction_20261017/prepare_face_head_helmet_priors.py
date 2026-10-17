"""Rebuild attachment anchors from newly face-constrained LHM-compatible motion."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation
from avatar_rig import pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
out=ROOT/'helmet_phase_head_faces_all300_v1';out.mkdir(exist_ok=False)
p=torch.load(ROOT/'body_head_faces_all300_v1/pose_sequence.pt',weights_only=True);obj=torch.load(ROOT/'helmet_material_pose_all300_v1/sequence.pt',weights_only=True);parents=torch.load(ROOT/'body_pose_v1/skeleton.pt',weights_only=True)['parents']
R=axis_angle_to_matrix(torch.cat([pose_axis_angles(p,t,'cpu') for t in range(300)]));globalR=[]
for j,parent in enumerate(parents.tolist()):globalR.append(R[:,j] if parent<0 else globalR[parent]@R[:,j])
globalR=torch.stack(globalR,1).numpy();world=p['joints55_world'].numpy();previous=json.loads((ROOT/'helmet_material_pose_all300_v1/report.json').read_text());phases=[]
for old in previous['phases']:
    a,b,j=old['start'],old['stop_exclusive'],old['joint'];relative=np.einsum('tji,tjk->tik',globalR[a:b,j],obj['R'][a:b].numpy());translation=np.einsum('tji,tj->ti',globalR[a:b,j],obj['translation_world'][a:b].numpy()-world[a:b,j]);r=Rotation.from_matrix(relative).mean();t=np.median(translation,axis=0);angles=np.rad2deg((r.inv()*Rotation.from_matrix(relative)).magnitude());phases.append({**old,'relative_R':r.as_matrix().tolist(),'relative_translation':t.tolist(),'median_relative_angle_deg':float(np.median(angles)),'max_relative_angle_deg':float(angles.max())})
torch.save({'global_joint_R':torch.tensor(globalR),'joints_world':p['joints55_world']},out/'anchors.pt')
(out/'report.json').write_text(json.dumps({'success':True,'phases':phases,'shared_relative_R':[x['relative_R'] for x in phases],'shared_relative_translation':[x['relative_translation'] for x in phases],'scope':'Same proposed grasp36–47 and worn180–299 intervals retained; anchor FK rebuilt from real face-constrained head/neck candidate, current helmet world poses only define warm-start relative means. Not adopted motion or measured contact; refit against actual all8 masks and inspect LHM contact renders.'},indent=2));print('new head attachment anchors prepared',flush=True)
