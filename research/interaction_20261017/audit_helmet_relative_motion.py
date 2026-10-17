"""Measure actual helmet/anchor relative motion; do not infer metric units."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--motion',default='helmet_native_roi_all300_v1');p.add_argument('--anchors',default='helmet_phase_head_faces_all300_v1');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False)
motion=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);anchors=torch.load(ROOT/o.anchors/'anchors.pt',weights_only=True);phases=json.loads((ROOT/o.anchors/'report.json').read_text())['phases'];result=[]
def summary(x):return dict(zip(['min','median','p95','max'],map(float,np.quantile(x,[0,.5,.95,1]))))
for phase in phases:
 a,b,j=phase['start'],phase['stop_exclusive'],phase['joint'];ar=anchors['global_joint_R'][a:b,j].numpy();at=anchors['joints_world'][a:b,j].numpy()
 rr=np.einsum('tji,tjk->tik',ar,motion['R'][a:b].numpy());rt=np.einsum('tji,tj->ti',ar,motion['translation_world'][a:b].numpy()-at)
 rotations=Rotation.from_matrix(rr);mean=rotations.mean();delta=np.rad2deg((mean.inv()*rotations).magnitude());step=np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude());td=np.linalg.norm(rt-np.median(rt,axis=0),axis=1)
 result.append({'start':a,'stop_exclusive':b,'joint':j,'relative_angle_to_phase_mean_deg':summary(delta),'relative_rotation_step_deg':summary(step),'relative_translation_to_phase_median_normalized_world':summary(td),'relative_translation_axis_range_normalized_world':np.ptp(rt,axis=0).tolist()})
(out/'report.json').write_text(json.dumps({'success':True,'motion':o.motion,'anchors':o.anchors,'phases':result,'scope':'Observed saved relative transforms only. Nonzero motion quantifies flexibility, does not prove erroneous slipping; real helmet settling and fitting errors can both contribute. Normalized-world units have not been calibrated as metres. No mask/headstrap or true head-direction verification.'},indent=2));print(json.dumps(result),flush=True)
