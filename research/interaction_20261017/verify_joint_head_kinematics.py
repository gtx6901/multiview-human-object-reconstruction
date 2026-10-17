"""Verify fast head anchors against the full shaped surface model and gradients."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from joint_head_constraints import JointHeadConstraints
torch.set_num_threads(4);torch.manual_seed(96);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());worn=torch.arange(300,device='cuda')>=180
p=argparse.ArgumentParser();p.add_argument('--out',default='joint_head_kinematics_verification_v1');p.add_argument('--front-visibility',action='store_true');o=p.parse_args()
m=JointHeadConstraints(ROOT,'helmet_joint_head_outer2_all300_v1','lhm_joint_head_occlusion_all300_v2',meta,worn,'lhm_face_anchor_correspondence_v1',o.front_visibility)
with torch.no_grad():m.raw.copy_(torch.randn_like(m.raw)*.3)
dense=m.state(surface=True);fast=m.state(surface=False);headerror=float((dense[0]-fast[0]).abs().max());jointerror=float((dense[1]-fast[1]).abs().max());assert headerror<1e-6 and jointerror<1e-6
gd=torch.autograd.grad(dense[0][:,0,1].sum()+dense[1][:,2].sum(),m.raw)[0];gf=torch.autograd.grad(fast[0][:,0,1].sum()+fast[1][:,2].sum(),m.raw)[0];gradienterror=float((gd-gf).abs().max());assert gradienterror<1e-5
loss,_=m.face_loss();g=torch.autograd.grad(loss,m.raw)[0];assert torch.isfinite(g).all() and g.abs().max()>0
out=ROOT/o.out;out.mkdir(exist_ok=False);report={'success':True,'max_head_rotation_error':headerror,'max_head_position_error':jointerror,'max_gradient_error':gradienterror,'face_gradient_max':float(g.abs().max()),'before_directional_points':m.before_visibility_points,'usable_face_points':int(m.valid.sum()),'front_visibility':o.front_visibility,'frames_without_face_support':int((m.worn&~m.valid.any((0,2))).sum()),'scope':'Fast actual-rig FK anchors compared with full shaped SMPL-X surface model under random worn neck/head perturbations; value and gradient equivalence verified. Actual-LHM-derived semantic face anchors enabled only after canonical rig/appearance identity checks. This is an implementation/definition check, not image-quality improvement or face observation truth.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
