"""Estimate one canonical head correction, then deform actual saved Gaussians.

Motion, camera, features, decoder and torso are fixed. Semantic anchors are a
fitting proxy; real rendered evidence is required after exporting this candidate.
"""
import os
from pathlib import Path
import sys,argparse,json,time
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_quaternion,quaternion_multiply
from smplx.lbs import batch_rigid_transform
from avatar_rig import pose_axis_angles,animate
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='helmet_joint_lhm_face_visible_all300_v1');p.add_argument('--face-anchors',default='lhm_face_anchor_correspondence_v1');p.add_argument('--mode',choices=['translation','rigid','axial-pitch'],default='rigid');p.add_argument('--steps',type=int,default=400);p.add_argument('--clean-camera-count',type=int,default=5);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
bundle=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);anchor_report=json.loads((ROOT/o.face_anchors/'report.json').read_text());assert anchor_report['success'] and all(q['usable'] and q['support']>=3 for q in anchor_report['points'])
if anchor_report.get('semantic_identity_sha256'):
 from canonical_identity import semantic_identity
 assert semantic_identity(bundle)==anchor_report['semantic_identity_sha256'],'Regenerate semantic anchors for changed canonical avatar'
else:
 reference=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True)
 for key in ['zero_coords','weights','rest_joints','neutral_rotation','neutral_to_zero']:assert torch.equal(bundle['rig'][key],reference['rig'][key]),'Regenerate semantic anchors for different canonical human'
 assert torch.equal(bundle['features'],reference['features']) and bundle['poses']['betas'].abs().max()==0
poses=bundle['poses'];rig=bundle['rig'];T=len(poses['frames']);assert T==300
aa=torch.cat([pose_axis_angles(poses,t,'cpu') for t in range(T)]).cuda();rot=axis_angle_to_matrix(aa);rest=rig['rest_joints'].cuda().expand(T,-1,-1);joints,_=batch_rigid_transform(rot,rest,rig['parents'].cuda());headR=rot[:,0]
for j in [3,6,9,12,15]:headR=headR@rot[:,j]
headt=joints[:,15]*poses['scale'].cuda()+poses['translation_world'].cuda();scale=poses['scale'].cuda()
count=5 if o.mode=='axial-pitch' else 3
local=torch.tensor(np.load(ROOT/o.face_anchors/'anchors.npz')['canonical_face_local'][:count],device='cuda',dtype=torch.float32)
screen=np.load(ROOT/'body_head_faces_all300_v1/face_screen.npz');xy=torch.tensor(screen['xy'][:,:,:count],device='cuda',dtype=torch.float32);valid=screen['inlier'][:,:,:count].copy()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
for vi,v in enumerate(meta['views']):
 hand=np.load(ROOT/'lhm_joint_head_occlusion_all300_v2'/f"{v['serial']}.npz")['hand_unknown'];q=np.floor((screen['xy'][vi,:,:count]+.5)*.125).astype(int);valid[vi]&=hand[np.arange(T)[:,None],q[...,1].clip(0,191),q[...,0].clip(0,255)]==0
localfront=local[0]-local[1:3].mean(0);localfront[1]=0;localfront=torch.nn.functional.normalize(localfront,dim=0);front=torch.einsum('tij,j->ti',headR,localfront);centers=-torch.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3]);tocam=torch.nn.functional.normalize(centers[:,None]-headt[None],dim=-1)
valid=torch.tensor(valid,device='cuda');valid[:,:,:3]&=((tocam*front[None]).sum(-1)>.15)[...,None];train=valid.clone();train[:,136:]=False;assert train.sum()>500
confidence=torch.tensor(np.minimum(screen['scores'][:,:,:count]/5,1),device='cuda',dtype=torch.float32)
neck_axis=torch.nn.functional.normalize((rig['rest_joints'][0,15]-rig['rest_joints'][0,12]).cuda(),dim=0)
pitch_axis=torch.tensor([1.,0.,0.],device='cuda')
selected_cameras=list(range(len(meta['views'])))
if o.mode=='axial-pitch':
 initial_points=headt[:,None]+torch.einsum('tij,kj->tki',headR,local)*scale
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],initial_points)+W[:,None,None,:3,3];qq=torch.einsum('vij,vtkj->vtki',K,pc);ee=(qq[...,:2]/qq[...,2:].clamp_min(.01)-xy).norm(dim=-1)
 ranks=[float(ee[v,:136][train[v,:136]].median()) if int(train[v,:136].sum())>=50 else float('inf') for v in range(len(meta['views']))]
 selected_cameras=sorted(range(len(ranks)),key=lambda v:ranks[v])[:o.clean_camera_count];assert all(np.isfinite(ranks[v]) for v in selected_cameras)
 for v in range(len(meta['views'])):
  if v not in selected_cameras:train[v]=False
weight=confidence*train
raw=torch.nn.Parameter(torch.zeros(6,device='cuda'))
def evaluate():
 if o.mode=='axial-pitch':shift=neck_axis*(raw[0].tanh()*.04);angle=pitch_axis*(raw[1].tanh()*15*np.pi/180)
 else:
  shift=raw[:3].tanh()*.04;angle=raw[3:].tanh()*(15*np.pi/180)/3**.5 if o.mode=='rigid' else raw[3:]*0
 C=axis_angle_to_matrix(angle);transformed=local@C.T+shift;points=headt[:,None]+torch.einsum('tij,kj->tki',headR,transformed)*scale
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],points)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,pc);error=(q[...,:2]/q[...,2:].clamp_min(.01)-xy).norm(dim=-1)
 loss=(2*(torch.sqrt(1+(error/8)**2)-1)*weight).sum()/weight.sum()+.02*(shift/.04).square().mean()+.02*(angle/(15*np.pi/180)).square().mean()
 return loss,error,shift,angle
with torch.no_grad():before=evaluate()[1].clone()
opt=torch.optim.Adam([raw],lr=.025);history=[]
for i in range(o.steps+1):
 opt.zero_grad();loss,error,shift,angle=evaluate()
 if i%100==0:history.append({'step':i,'loss':float(loss.detach()),'train_native_px':float(error[train].mean().detach())});print(json.dumps(history[-1]),flush=True)
 if i<o.steps:loss.backward();opt.step()
with torch.no_grad():
 _,error,shift,angle=evaluate();shift=shift.cpu();angle=angle.cpu();headweight=((rig['weights'][:,[15,22,23,24]].sum(-1)-.02)/.96).clamp(0,1);headweight=headweight.square()*(3-2*headweight);Ci=axis_angle_to_matrix(headweight[:,None]*angle);pivot=rig['rest_joints'][0,15];old=rig['zero_coords'];new=torch.einsum('nij,nj->ni',Ci,old-pivot)+pivot+headweight[:,None]*shift
 # Neutral rotation lives before neutral_to_zero; conjugate the correction
 # into that coordinate frame so anisotropic Gaussian axes move consistently.
 N=rig['neutral_to_zero'][:,:3,:3];correction=torch.linalg.inv(N)@Ci@N;dq=matrix_to_quaternion(correction);neutral=torch.nn.functional.normalize(quaternion_multiply(dq,rig['neutral_rotation']),dim=-1)
 new=torch.where((headweight>0)[:,None],new,old);neutral=torch.where((headweight>0)[:,None],neutral,rig['neutral_rotation'])
 assert torch.isfinite(new).all() and torch.isfinite(neutral).all();assert torch.equal(new[headweight==0],old[headweight==0]);assert shift.norm()>0
 candidate={**bundle,'rig':{**rig,'zero_coords':new,'neutral_rotation':neutral},'scope':'Unadopted shared canonical head correction from clean0..135 face pseudo-observations; all300 motion and features fixed. Gaussian centers and axes actually changed; not a proxy-only correction. Requires rendered semantic remeasurement and same-camera image review.'};torch.save(candidate,out/'avatar.pt')
 assert all(torch.equal(candidate['poses'][k],poses[k]) for k in poses)
 phase={}
 for label,a,b in [('clean_train',0,136),('transition',136,180),('worn_transfer',180,300)]:
  mask=valid[:,a:b];phase[label]={'points':int(mask.sum()),'before_native_px':float(before[:,a:b][mask].mean()),'after_native_px':float(error[:,a:b][mask].mean())}
 # Verify actual exported centers against the same LBS linear displacement.
 checks=[]
 for t in [0,42,180,299]:
  g0=animate(rig,poses,t);g1=animate(candidate['rig'],poses,t);_,A=batch_rigid_transform(axis_angle_to_matrix(pose_axis_angles(poses,t,'cpu')),rig['rest_joints'],rig['parents']);L=(rig['weights']@A[0].reshape(-1,16)).reshape(-1,4,4)[:,:3,:3];expected=torch.einsum('nij,nj->ni',L,new-old)*poses['scale'];gap=float(((g1['xyz']-g0['xyz'])-expected).abs().max());assert gap<1e-6;checks.append({'frame':t,'max_actual_center_displacement_error':gap})
 report={'success':True,'source_avatar':o.avatar,'mode':o.mode,'history':history,'shared_translation_model_units':shift.tolist(),'shared_rotation_axis_angle_rad':angle.tolist(),'shared_rotation_norm_deg':float(angle.norm()*180/np.pi),'phase_face_proxy_errors':phase,'actual_gaussian_displacement_checks':checks,'affected_gaussians':int((headweight>0).sum()),'unchanged_gaussians':int((headweight==0).sum()),'seconds':time.time()-start,'adopted':False,'selected_cameras':[meta['views'][v]['serial'] for v in selected_cameras],'face_point_count':count,'degrees_of_freedom':2 if o.mode=='axial-pitch' else (3 if o.mode=='translation' else 6),'scope':'Shared canonical head correction; fixed all300 motion/camera/features/decoder/body shape/rest skeleton. Axial-pitch mode uses neck-axis translation and canonical X pitch only, nose/eyes/ears, clean0..135 and five lowest initial residual cameras. Bounds .04 model units/15deg; these are optimization bounds, not prescribed angles. Actual Gaussian centers and anisotropic axes transformed. Render and semantic remeasurement required.'}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
