"""Verify intentional canonical updates and restricted per-frame changes."""
import os
from pathlib import Path
import sys,json,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
from shared_scene import load_shared_scene
from canonical_identity import semantic_identity
scene=load_shared_scene(ROOT/'shared_scene_head_alignment_all300_v3');b=scene['avatar'];base=torch.load(ROOT/'shared_head_alignment_clean_v3/avatar.pt',weights_only=True);motion=torch.load(ROOT/'shared_head_joint_motion_all300_v3/avatar.pt',weights_only=True);accepted=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True)
assert semantic_identity(b)==semantic_identity(base)==semantic_identity(motion)
assert semantic_identity(b)!=semantic_identity(accepted),'Intentional canonical head update missing'
assert torch.equal(b['features'],accepted['features'])
for key in ['root_pose','translation_world','betas','scale','left_hand_pose','right_hand_pose']:assert torch.equal(b['poses'][key],base['poses'][key])
other=[i for i in range(21) if i not in [11,14]];assert torch.equal(b['poses']['body_pose'][:,other],base['poses']['body_pose'][:,other]);assert torch.equal(b['poses']['body_pose'][:180],base['poses']['body_pose'][:180])
report=json.loads((ROOT/'shared_head_alignment_portrait_all300_v3/report.json').read_text());assert report['success'] and report['frames']==list(range(300)) and len(report['videos'])==4 and all(v['decoded_frames']==300 for v in report['videos'])
result={'success':True,'canonical_avatar_matches_updated_source':True,'canonical_head_differs_from_original_baseline':True,'original_appearance_features_preserved':True,'only_worn_neck_head_motion_changed':True,'all_four_videos_independently_decoded_300':True,'scope':'Shared container hash/load and canonical/motion tensor checks plus video decode counts. Explicit head geometry change is intentional; does not assert image improvement, true helmet shape, novel-view accuracy or correct contact.'};(ROOT/'shared_scene_head_alignment_all300_v3/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
