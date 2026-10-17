"""Verify canonical identity and same-initialization face-term control."""
import os
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
from canonical_identity import semantic_identity
out=ROOT/'body_face_stage_verification_v1';out.mkdir(exist_ok=False)
source=torch.load(ROOT/'shared_shape_accepted_joint_motion_all300_v2/avatar.pt',weights_only=True);candidate=torch.load(ROOT/'shared_shape_body_face_motion_v1/avatar.pt',weights_only=True);control=torch.load(ROOT/'shared_shape_body_face_zero_control_v1/avatar.pt',weights_only=True);accepted=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True)
assert semantic_identity(source)==semantic_identity(candidate)==semantic_identity(control);assert torch.equal(candidate['features'],accepted['features'])
for key in ['betas','scale','left_hand_pose','right_hand_pose']:assert torch.equal(source['poses'][key],candidate['poses'][key])
a=json.loads((ROOT/'shared_shape_body_face_motion_v1/report.json').read_text());b=json.loads((ROOT/'shared_shape_body_face_zero_control_v1/report.json').read_text());assert a['source_avatar']==b['source_avatar'] and a['shared_beta_norm']>0
for key in ['body_before_px1024','hand_before_px1024']:assert abs(a[key]-b[key])<1e-6
for phase in a['face_by_phase']:assert abs(a['face_by_phase'][phase]['before_native_px']-b['face_by_phase'][phase]['before_native_px'])<1e-6
repair=json.loads((ROOT/'shared_shape_body_face_motion_v1/derived_projection_repair.json').read_text());assert repair['all_other_motion_tensors_exact'] and repair['canonical_tensors_exact']
report={'success':True,'canonical_tensors_fixed_between_motion_trials':True,'accepted_original_features_preserved':True,'shared_beta_scale_and_fingers_exact':True,'face_zero_control_same_initialization':True,'derived_projection_repair_render_inputs_unchanged':True,'scope':'Component identity and control-initialization checks only. Nonzero beta is data-estimated but not certified physical body shape; facial residual/selected rendering evidence is separate. Object integration not validated by this check.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
