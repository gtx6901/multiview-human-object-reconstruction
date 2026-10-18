"""Assemble verified neck/hands and local sleeve weights, check saved invariants."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
torch.set_num_threads(4);out=ROOT/'temporal_body_integrated_v3';out.mkdir(exist_ok=True);assert not (out/'avatar.pt').exists()
base=torch.load(ROOT/'avatar_grounded_shared_shape_v2/avatar.pt',weights_only=True);hand=torch.load(ROOT/'axial_temporal_hands_v3/avatar.pt',weights_only=True);skin=torch.load(ROOT/'underarm_decoded_chain_candidate_v4/avatar.pt',weights_only=True)
weights=skin['rig']['weights'];assert weights.shape==hand['rig']['weights'].shape
headmask=hand['rig']['weights'][:,[12,15,22,23,24]].sum(-1)>.5;assert torch.equal(weights[headmask],hand['rig']['weights'][headmask])
b={**hand,'rig':{**hand['rig'],'weights':weights}}
assert torch.equal(b['features'],base['features']);assert all(torch.equal(b['decoder'][k],base['decoder'][k]) for k in base['decoder'])
aa=[torch.cat([pose_axis_angles(x['poses'],t,'cpu') for t in range(300)]) for x in [base,b]]
keep=[j for j in range(55) if j not in [12,15,20,21] and j<25];assert torch.equal(aa[0][:,keep],aa[1][:,keep]);assert torch.equal(b['poses']['betas'],base['poses']['betas']);assert torch.equal(b['poses']['scale'],base['poses']['scale']);assert torch.equal(b['poses']['translation_world'],base['poses']['translation_world'])
with torch.no_grad():
 model=SMPLXSurfaceLandmarks();feet=[model(a,x['poses']['scale'],x['poses']['translation_world'],x['poses']['betas'])['feet6'] for a,x in zip(aa,[base,b])];footgap=float((feet[1]-feet[0]).abs().max());print('foot surface pose-corrective displacement',footgap,flush=True)
 contact=torch.tensor(np.repeat(np.load(ROOT/'feet_constraints_all300_v1/constraints.npz')['contact'],3,axis=1));both=contact[1:]&contact[:-1];footstats=[]
 for ft in feet:
  s=base['poses']['scale'];d2=(ft[47:]-2*ft[46:-1]+ft[45:-2])/s;vel=(ft[1:]-ft[:-1])/s*30;footstats.append({'d2_model_p95':float(torch.quantile(d2.norm(dim=-1),.95)),'stance_speed_model_p95':float(torch.quantile(vel.norm(dim=-1)[both],.95))})
 assert footstats[1]['d2_model_p95']<=footstats[0]['d2_model_p95']*1.1 and footstats[1]['stance_speed_model_p95']<=footstats[0]['stance_speed_model_p95']*1.1,footstats
torch.save(b,out/'avatar.pt');torch.save(b['poses'],out/'pose_sequence.pt')
r={'success':True,'frames':300,'same_features_decoder':True,'feet6_max_surface_displacement_world':footgap,'before_feet':footstats[0],'after_feet':footstats[1],'root_body_legs_beta_scale_translation_exact':True,'head_hands_changed':True,'weight_edited_gaussians':int((weights!=base['rig']['weights']).any(-1).sum()),'scope':'Integrated single candidate: shared canonical axial/pitch head correction, bounded data-fitted neck/head, temporal wrists/fingers and source-supported local dark sleeve weight correction. Leg motion exact; globally coupled SMPL-X pose-corrective foot displacement and stability recorded explicitly, stability regression limited to10percent. Features/decoder exact. No automatic master adoption.'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
