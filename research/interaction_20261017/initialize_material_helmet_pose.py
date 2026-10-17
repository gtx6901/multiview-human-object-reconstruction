"""Use observed material evidence to initialize ambiguous rigid orientation modes."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation,Slerp
out=ROOT/'helmet_material_initial_all300_v1';out.mkdir(exist_ok=False)
source='helmet_shell_pose_all300_v3';motion=torch.load(ROOT/source/'sequence.pt',weights_only=True);previous=json.loads((ROOT/source/'report.json').read_text());audit=json.loads((ROOT/'helmet_material_orientation_audit_v1/report.json').read_text());seeds=torch.load(ROOT/'helmet_material_orientation_audit_v1/seed_candidates.pt',weights_only=True)
body=torch.load(ROOT/'body_hands_all300_v1/pose_sequence.pt',weights_only=True);anchor=torch.load(ROOT/'helmet_phase_priors_all300_v1/anchors.pt',weights_only=True)
normal=np.array(json.loads((ROOT/'helmet_shell_all300_v5/report.json').read_text())['opening_plane_normal_local']);pivot=(motion['vertices_local'].min(0).values+motion['vertices_local'].max(0).values).numpy()/2;scale=float(motion['logscale'].exp())
group=np.concatenate([np.eye(3)[None],Rotation.create_group('O').as_matrix()]);headrows=[x for x in audit['records'] if x['frame']>=180];choices=[]
for qi,q in enumerate(group):
    dots=[]
    for row in headrows:
        f=row['frame'];up=(body['joints55_world'][f,15]-body['joints55_world'][f,12]).numpy();up/=np.linalg.norm(up);dots.append(float(np.dot(motion['R'][f].numpy()@q@normal,up)))
    if min(dots)<.45 or np.mean(dots)<.65:continue
    choices.append((np.mean([row['candidates'][qi]['joint_cost'] for row in headrows]),qi,dots))
assert choices;score,headq,dots=min(choices);Q=group[headq];oldR=motion['R'].numpy();oldT=motion['translation_world'].numpy();R=oldR.copy();t=oldT.copy()
R[180:]=oldR[180:]@Q;t[180:]=oldT[180:]+np.einsum('tij,j->ti',oldR[180:]-R[180:],pivot)*scale
knots={}
for row,index in zip(audit['records'],range(len(audit['records']))):
    f=row['frame']
    if f<180:knots[f]=seeds['R'][index].numpy()
for f in [36,42,47]:knots[f]=oldR[f]
knots[180]=R[180];times=sorted(knots);R[:180]=Slerp(times,Rotation.from_matrix(np.stack([knots[f] for f in times])))(np.arange(180)).as_matrix();R[36:48]=oldR[36:48]
t[:180]=oldT[:180]+np.einsum('tij,j->ti',oldR[:180]-R[:180],pivot)*scale
sharedR=np.array(previous['shared_relative_R']);sharedt=np.array(previous['shared_relative_translation'])
for i,phase in enumerate(previous['phases']):
    if phase['joint']==15:
        sharedR[i]=sharedR[i]@Q;a,b=phase['start'],phase['stop_exclusive'];j=phase['joint'];Ar=anchor['global_joint_R'][a:b,j].numpy();At=anchor['joints_world'][a:b,j].numpy();sharedt[i]=np.median(np.einsum('tji,tj->ti',Ar,t[a:b]-At),axis=0)
result={**motion,'R':torch.tensor(R,dtype=torch.float32),'translation_world':torch.tensor(t,dtype=torch.float32)};torch.save(result,out/'sequence.pt')
report={'success':True,'source':source,'phases':previous['phases'],'shared_relative_R':sharedR.tolist(),'shared_relative_translation':sharedt.tolist(),'head_common_candidate':int(headq),'head_common_delta_deg':float(np.rad2deg(Rotation.from_matrix(Q).magnitude())),'head_opening_up_dots':dots,'head_coarse_joint_cost':float(score),'scope':'INITIALIZATION ONLY. Discrete observed white-shell/dark-lining material audit proposes free-segment knots; rotations SO(3)-interpolated before current-frame refitting, original world bbox centers preserved. Stable-wrist36–47 retained; worn180–299 uses ONE common phase rotation mode with an explicit opening-up prior. Does not establish true pose from the proxy, no final animation adoption. All300 actual-image semantic+silhouette refinement and temporal/render evaluation required.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('material-aware initialization prepared',flush=True)
