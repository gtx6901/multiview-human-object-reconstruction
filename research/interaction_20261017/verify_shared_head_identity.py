"""Check real deformation, semantic-anchor provenance and fitting gradients."""
import os
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
from joint_head_constraints import JointHeadConstraints
from canonical_identity import semantic_identity
from pytorch3d.transforms import quaternion_to_matrix,axis_angle_to_matrix
out=ROOT/'shared_head_identity_verification_v3';out.mkdir(exist_ok=False);b=torch.load(ROOT/'shared_head_alignment_clean_v3/avatar.pt',weights_only=True);old=torch.load(ROOT/'helmet_joint_lhm_face_visible_all300_v1/avatar.pt',weights_only=True);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());w=torch.arange(300,device='cuda')>=180
m=JointHeadConstraints(ROOT,'shared_head_alignment_clean_v3','lhm_joint_head_occlusion_all300_v2',meta,w,'shared_head_face_correspondence_v4',True);loss,_=m.face_loss();loss.backward();assert torch.isfinite(m.raw.grad).all() and m.raw.grad.abs().max()>0
try:JointHeadConstraints(ROOT,'shared_head_alignment_clean_v3','lhm_joint_head_occlusion_all300_v2',meta,w,'lhm_face_anchor_correspondence_v1',True)
except AssertionError:rejected=True
else:raise AssertionError('Stale canonical anchor was accepted')
assert semantic_identity(b)!=semantic_identity(old)
rig=old['rig'];q=b['rig']['neutral_rotation'];h=((rig['weights'][:,[15,22,23,24]].sum(-1)-.02)/.96).clamp(0,1);h=h.square()*(3-2*h);a=torch.tensor(json.loads((ROOT/'shared_head_alignment_clean_v3/report.json').read_text())['shared_rotation_axis_angle_rad']);C=axis_angle_to_matrix(h[:,None]*a);N=rig['neutral_to_zero'][:,:3,:3];expected=torch.linalg.inv(N)@C@N@quaternion_to_matrix(rig['neutral_rotation']);err=float((quaternion_to_matrix(q)-expected).abs().max());assert err<1e-4
assert torch.equal(b['rig']['zero_coords'][h==0],rig['zero_coords'][h==0]);assert torch.equal(q[h==0],rig['neutral_rotation'][h==0]);assert all(torch.equal(b['poses'][k],old['poses'][k]) for k in old['poses'])
report={'success':True,'new_anchor_gradient_max':float(m.raw.grad.abs().max()),'stale_anchor_rejected':rejected,'canonical_identity_changed':True,'max_canonical_axis_matrix_error':err,'zero_head_weight_centers_and_axes_exact':True,'motion_tensors_exact':True,'scope':'Actual changed-rig semantic-anchor identity/gradient check. Canonical axis conjugation projected to unit quaternions is approximate for weakly nonorthogonal LBS neutral transforms; measured maximum matrix difference recorded. No new quality or ground-truth claim.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
