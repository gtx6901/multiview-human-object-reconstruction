"""Independent check: final local fit changes only1219 allowed weight rows."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from avatar_rig import animate
torch.set_num_threads(4);out=ROOT/'temporal_weight_contract_v4';out.mkdir(exist_ok=False)
a=torch.load(ROOT/'temporal_body_integrated_v3/avatar.pt',weights_only=True);b=torch.load(ROOT/'underarm_weight_amount_fit_v3/avatar.pt',weights_only=True);orig=torch.load(ROOT/'axial_temporal_hands_v3/avatar.pt',weights_only=True)
def same(x,y):
 if isinstance(x,torch.Tensor):return torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
 return x==y
for key in ['features','decoder','decoder_config','background_embedding','poses']:assert same(a[key],b[key]),key
for key in a['rig']:
 if key!='weights':assert same(a['rig'][key],b['rig'][key]),key
before=orig['rig']['weights'];proposal=a['rig']['weights'];after=b['rig']['weights'];ids=(before!=proposal).any(-1);assert int(ids.sum())==1219;assert torch.equal(after[~ids],before[~ids]);delta=proposal[ids]-before[ids];amount=((after[ids]-before[ids])*delta).sum(-1)/delta.square().sum(-1);assert (amount>=0).all() and (amount<=1).all();assert torch.allclose(after[ids],before[ids]+amount[:,None]*delta,atol=2e-7)
assert torch.allclose(after.sum(-1),torch.ones(len(after)),atol=1e-5) and (after>=0).all();checks=[]
with torch.no_grad():
 for f in [0,42,165,225,299]:
  g=animate(b['rig'],b['poses'],f);assert all(torch.isfinite(v).all() for v in g.values());checks.append({'frame':f,'finite_gaussians':len(g['xyz'])})
r={'success':True,'poses_all300_exact':True,'features_decoder_geometry_exact':True,'only_weight_rows':int(ids.sum()),'unchanged_weight_rows':int((~ids).sum()),'amount_min':float(amount.min()),'amount_max':float(amount.max()),'checks':checks,'scope':'Independent saved-file comparison. Learned1219 weights remain convex combinations of original and proposed arm-only sleeve weights. All300 poses, Gaussian canonical geometry/attributes/features and decoder exact versus the prior integrated candidate. No visual acceptance implied.'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
