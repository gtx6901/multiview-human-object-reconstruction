"""Transport attached object states to a changed human without refitting helmet."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from avatar_rig import pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.lbs import batch_rigid_transform
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',required=True);p.add_argument('--source-avatar',default='body_head_faces_all300_v1');p.add_argument('--motion',default='helmet_lhm_depth_tight_all300_v1');p.add_argument('--phases',default='helmet_phase_head_faces_all300_v1');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False)
old=torch.load(ROOT/o.source_avatar/'avatar.pt',weights_only=True);new=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);m=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);assert torch.equal(m['frames'],new['poses']['frames'])
def anchors(b):
 ps=b['poses'];aa=torch.cat([pose_axis_angles(ps,t,'cpu') for t in range(len(ps['frames']))]);R=axis_angle_to_matrix(aa);J,_=batch_rigid_transform(R,b['rig']['rest_joints'].expand(len(aa),-1,-1),b['rig']['parents']);G=[]
 for j,pa in enumerate(b['rig']['parents'].tolist()):G.append(R[:,j] if pa<0 else G[pa]@R[:,j])
 return torch.stack(G,1),J*ps['scale']+ps['translation_world'][:,None]
ar,at=anchors(old);br,bt=anchors(new);ratio=new['poses']['scale']/old['poses']['scale'];rr=m['R'].clone();tt=m['translation_world'].clone();checks=[];attached=torch.zeros(len(rr),dtype=torch.bool)
for ph in json.loads((ROOT/o.phases/'report.json').read_text())['phases']:
 a,z,j=ph['start'],ph['stop_exclusive'],ph['joint'];sl=slice(a,z);localR=ar[sl,j].transpose(-1,-2)@m['R'][sl];localt=torch.einsum('tji,tj->ti',ar[sl,j],m['translation_world'][sl]-at[sl,j]);rr[sl]=br[sl,j]@localR;tt[sl]=bt[sl,j]+torch.einsum('tij,tj->ti',br[sl,j],localt*ratio);attached[sl]=True
 checkR=br[sl,j].transpose(-1,-2)@rr[sl];checkt=torch.einsum('tji,tj->ti',br[sl,j],tt[sl]-bt[sl,j])/ratio
 checks.append({'start':a,'stop_exclusive':z,'joint':j,'relative_rotation_max_error':float((checkR-localR).abs().max()),'relative_translation_max_error':float((checkt-localt).abs().max())});assert torch.allclose(checkR,localR,atol=2e-6) and torch.allclose(checkt,localt,atol=2e-6)
assert torch.equal(rr[~attached],m['R'][~attached]) and torch.equal(tt[~attached],m['translation_world'][~attached]);torch.save({**m,'R':rr,'translation_world':tt},out/'sequence.pt')
(out/'report.json').write_text(json.dumps({'success':True,'source_avatar':o.source_avatar,'avatar':o.avatar,'motion':o.motion,'scale_ratio':float(ratio),'attached_frames':int(attached.sum()),'checks':checks,'scope':'Old full relative transforms INCLUDING per-frame residuals transported exactly to new joint transforms; relative translation scaled by human scale ratio. Free object states/geometry/texture unchanged. No new helmet fitting or prescribed clearance/angle. Attachment transitions require actual review.'},indent=2));print('Helmet attachment transported without optimization.',flush=True)
