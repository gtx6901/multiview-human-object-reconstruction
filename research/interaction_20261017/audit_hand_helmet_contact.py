"""Actual held-action contact diagnostic and displaced-object negative case."""
import os
from pathlib import Path
import sys,torch,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
from hand_helmet_contact import HandHelmetContact
out=ROOT/'hand_helmet_contact_audit_v1';out.mkdir(exist_ok=False);m=torch.load(ROOT/'shared_head_joint_motion_all300_v3/sequence.pt',weights_only=True);R=m['R'].cuda().requires_grad_();t=m['translation_world'].cuda().requires_grad_();c=HandHelmetContact(ROOT,'shared_head_joint_motion_all300_v3',m['vertices_local'],m['faces'],m['logscale'].cuda().exp(),R,t)
loss=c.loss(R,t);loss.backward();assert torch.isfinite(R.grad).all() and torch.isfinite(t.grad).all() and t.grad.abs().max()>0
with torch.no_grad():negative=c.loss(R,t+torch.tensor([.1,0,0],device='cuda')*c.human_scale)
assert negative>loss+1e-3
report={'success':True,'baseline_loss':float(loss.detach()),'controlled_displacement_loss':float(negative),'rotation_gradient_max':float(R.grad.abs().max()),'translation_gradient_max':float(t.grad.abs().max()),'baseline_contact':c.report(R,t),'scope':'Actual106 held-frame diagnostic plus deliberate object displacement. Baseline already satisfies the at-least-one-point proximity threshold in all106 screened frames, so adding this weak prior alone is not evidence of a useful contact improvement. Signed shell collision and extended-hand/Gaussian nonpenetration remain unimplemented.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
