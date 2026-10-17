"""Reuse only provably unchanged frames and splice freshly rendered worn depth."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--updated-avatar',default='helmet_joint_head_all300_v1');p.add_argument('--fresh-cache',default='lhm_joint_head_occlusion_worn_v1');p.add_argument('--out',default='lhm_joint_head_occlusion_all300_v1');p.add_argument('--base-avatar',default='body_head_faces_all300_v1');p.add_argument('--base-cache',default='lhm_occlusion_all300_v1');o=p.parse_args()
old=ROOT/o.base_cache;fresh=ROOT/o.fresh_cache;out=ROOT/o.out
r=json.loads((fresh/'report.json').read_text());assert r['success'];out.mkdir(exist_ok=False)
a=torch.load(ROOT/o.base_avatar/'avatar.pt',weights_only=True);b=torch.load(ROOT/o.updated_avatar/'avatar.pt',weights_only=True)
base_report=json.loads((old/'report.json').read_text());assert base_report['success'] and base_report['avatar_source']==o.base_avatar
assert r['avatar_source']==o.updated_avatar
def same(x,y):
 if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
 if isinstance(x,torch.Tensor):return torch.equal(x,y)
 return x==y
for key in ['rig','features','decoder','background_embedding']:assert same(a[key],b[key]),key
ids=np.array(r['frames']);assert np.array_equal(ids,np.arange(180,300));unchanged=torch.arange(180)
for key in ['root_pose','translation_world','left_hand_pose','right_hand_pose','betas','scale']:assert torch.equal(a['poses'][key],b['poses'][key]),key
assert torch.equal(a['poses']['body_pose'][unchanged],b['poses']['body_pose'][unchanged])
views=[]
for p in sorted(old.glob('*.npz')):
 with np.load(p) as d,np.load(fresh/p.name) as n:
  assert np.array_equal(d['frames'],np.arange(300)) and np.array_equal(n['frames'],ids)
  merged={k:d[k].copy() for k in d.files}
  for key in ['K','W']:assert np.array_equal(d[key],n[key])
  for key in ['depth','alpha','hand_unknown','depth_std','decoded_alpha']:
   assert len(d[key])==300 and len(n[key])==120;merged[key][ids]=n[key];assert np.array_equal(merged[key][:180],d[key][:180])
  np.savez_compressed(out/p.name,**merged);views.append(p.stem)
report={'success':True,'frames':list(range(300)),'avatar_source':o.updated_avatar,'recomputed_frames':ids.tolist(),'views':views,'source_caches':[old.name,fresh.name],'unchanged_frame_asset_and_pose_checks':True,'scope':r['scope']+' Merged180..299 freshly rendered current worn-head depth;0..179 reused only after exact original/current rig, features, decoder and relevant pose comparisons. This is a visibility outer update, not new geometry truth or accepted helmet contact.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('full current-head depth cache verified and merged')
