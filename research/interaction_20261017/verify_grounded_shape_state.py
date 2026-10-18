"""Independent shaped-state replay and foot stability check on actual files."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
torch.set_num_threads(4);out=ROOT/'grounded_shape_state_verification_v2';out.mkdir(exist_ok=False)
a=torch.load(ROOT/'avatar_overhead_body_appearance_grounded_v2/avatar.pt',weights_only=True);b=torch.load(ROOT/'avatar_grounded_shared_shape_v2/avatar.pt',weights_only=True)
assert torch.equal(a['features'],b['features']);assert a['decoder_config']==b['decoder_config'];assert all(torch.equal(a['decoder'][k],b['decoder'][k]) for k in a['decoder'])
model=SMPLXSurfaceLandmarks();d=np.load(ROOT/'feet_constraints_all300_v1/constraints.npz');contact=torch.tensor(np.repeat(d['contact'],3,axis=1));both=contact[1:]&contact[:-1];stats=[];angles=[]
for bundle in [a,b]:
 p=bundle['poses'];aa=torch.cat([pose_axis_angles(p,t,'cpu') for t in range(300)]);angles.append(aa)
 with torch.no_grad():r=model(aa,p['scale'],p['translation_world'],p['betas'])
 ft=r['feet6'];v=(ft[1:]-ft[:-1])/p['scale']*30;d2=(ft[47:]-2*ft[46:-1]+ft[45:-2])/p['scale']
 stats.append({'foot_d2_model_median':float(d2.norm(dim=-1).median()),'foot_d2_model_p95':float(torch.quantile(d2.norm(dim=-1),.95)),'contact_speed_model_median':float(v.norm(dim=-1)[both].median()),'contact_speed_model_p95':float(torch.quantile(v.norm(dim=-1)[both],.95))})
assert torch.equal(angles[0][:,[12,15]],angles[1][:,[12,15]]);assert torch.equal(angles[0][:,22:],angles[1][:,22:]);assert torch.equal(a['poses']['frames'],b['poses']['frames'])
r={'success':True,'frames':300,'source':'avatar_overhead_body_appearance_grounded_v2','candidate':'avatar_grounded_shared_shape_v2','same_LHM_features_and_decoder':True,'head_neck_local_rotations_exact':True,'jaw_eyes_finger_rotations_exact':True,'before_feet':stats[0],'after_feet':stats[1],'scope':'Independent full300 saved-state replay through the official surface model. Shared beta/scale/rest geometry and root/body motion may change; original LHM features/decoder, head/neck local and finger rotations fixed. Foot stability measured on45..299 and observed stance intervals, not a conventional mesh asset or visual acceptance.'}
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
