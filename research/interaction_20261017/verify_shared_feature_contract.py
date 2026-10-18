"""Independently load a feature-only result and verify every fixed state."""
import os
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from shared_scene import load_shared_scene
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',required=True);o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False)
a=load_shared_scene(ROOT/o.source);b=load_shared_scene(ROOT/o.candidate)
checks=[]
def same(x,y,path):
 if isinstance(x,torch.Tensor):assert torch.equal(x,y),path;checks.append(path)
 elif isinstance(x,dict):
  assert x.keys()==y.keys(),path
  for k in x:same(x[k],y[k],path+'.'+k)
 else:assert x==y,path
for k in a['avatar']:
 if k!='features':same(a['avatar'][k],b['avatar'][k],'avatar.'+k)
same(a['mesh'],b['mesh'],'helmet');same(a['object_motion'],b['object_motion'],'object_motion');same(a['cameras'],b['cameras'],'cameras')
old=a['avatar']['features'].float();new=b['avatar']['features'].float();weights=a['avatar']['rig']['weights'];protected=(weights[:,12]+weights[:,15]+weights[:,20:].sum(-1))>.5
assert torch.equal(old[protected],new[protected]);assert torch.isfinite(new).all()
bound=json.loads((ROOT/o.candidate).parent.joinpath('report.json').read_text())['feature_bound'];maximum=float((new-old).abs().max());assert maximum<=bound+1e-6
r={'success':True,'source':o.source,'candidate':o.candidate,'fixed_tensor_fields_verified':len(checks),'motion_frames':len(a['avatar']['poses']['frames']),'protected_feature_rows':int(protected.sum()),'protected_features_exact':True,'all_rig_geometry_decoder_appearance_except_features_exact':True,'all_motion_and_helmet_exact':True,'feature_bound':bound,'feature_max_change':maximum,'scope':'Independent saved-scene loading and fixed-tensor comparison. Confirms the actual candidate changes only allowed body feature rows, all300 motion/rig/decoder/helmet/cameras unchanged. Does not claim visual improvement.'}
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
