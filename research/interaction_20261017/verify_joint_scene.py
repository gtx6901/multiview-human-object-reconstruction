"""Check that coupled motion still uses the accepted canonical LHM appearance."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from shared_scene import load_shared_scene
old=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True);new=load_shared_scene(ROOT/'shared_scene_joint_head_outer2_v1')['avatar']
def same(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
for k in ['rig','features','decoder','decoder_config','background_embedding']:assert same(old[k],new[k]),k
for k in ['root_pose','translation_world','betas','scale','left_hand_pose','right_hand_pose']:assert same(old['poses'][k],new['poses'][k]),k
keep=[i for i in range(21) if i not in [11,14]];assert torch.equal(old['poses']['body_pose'][:,keep],new['poses']['body_pose'][:,keep]);assert torch.equal(old['poses']['body_pose'][:180],new['poses']['body_pose'][:180])
video=json.loads((ROOT/'shared_scene_joint_head_portrait_all300_v1/report.json').read_text());assert video['success'] and len(video['frames'])==300 and all(r['decoded_frames']==300 for r in video['videos'])
report={'success':True,'canonical_rig_features_decoder_exactly_preserved':True,'non_head_motion_exactly_preserved':True,'before_worn_motion_exactly_preserved':True,'frames':300,'decoded_video_frames':[r['decoded_frames'] for r in video['videos']],'scope':'Actual shared-scene independent load and saved tensor comparison against previously accepted LHM canonical appearance. Only neck/head motion during180..299 differs in human motion; helmet motion also refitted. All four render videos decode300frames. Does not certify helmet shape, contact, recovered appearance or novel-view quality.'}
(ROOT/'shared_scene_joint_head_outer2_v1/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
