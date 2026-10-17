"""Shared SMPL-X avatar motion from all8 observed joints and real NLF pose priors."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='body_fused180_211_v1');p.add_argument('--data',default='helmet_contiguous180_211_v1');p.add_argument('--detections',default='detections_contiguous180_211_v2');p.add_argument('--alignment',default='nlf_world_alignment_contiguous180_211_v1');p.add_argument('--nlf',default='nlf_smplh_contiguous180_211_v1');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/options.data/'manifest.json').read_text());det=np.load(ROOT/options.detections/'detections.npz');frames=det['frames'].tolist();T=len(frames)
alignment=np.load(ROOT/options.alignment/'aligned_joints.npz')
assert alignment['frames'].tolist()==frames
target3=np.nanmedian(np.where(alignment['valid'][...,None],alignment['per_view_world_joints'],np.nan),axis=0)
assert np.isfinite(target3).all()
prior=np.stack([np.stack([torch.load(ROOT/options.nlf/f"{v['serial']}_{f:04d}.pt",weights_only=True)['pose_rotvecs'][0].numpy()[:66].reshape(22,3) for f in frames]) for v in meta['views']])
R=Rotation.from_rotvec(prior.reshape(-1,3)).as_matrix().reshape(8,T,22,3,3)
for i,v in enumerate(meta['views']):R[i,:,0]=np.array(v['W'])[:3,:3].T@R[i,:,0]
def robust_mean(matrices):
    rot=Rotation.from_matrix(matrices)
    distances=np.array([np.linalg.norm((Rotation.from_matrix(x).inv()*rot).as_rotvec(),axis=1) for x in matrices]);medoid=np.argmin(distances.sum(1));keep=distances[medoid]<np.deg2rad(70)
    return rot[keep].mean().as_rotvec()
initial=np.array([[robust_mean(R[:,t,j]) for j in range(22)] for t in range(T)])
skeleton=torch.load(ROOT/'body_pose_v1/skeleton.pt',weights_only=True);rest=skeleton['rest_joints'].cuda();parents=skeleton['parents'].cuda()
avatar=torch.load(ROOT/'body_pose_v1/pose_sequence.pt',weights_only=True);scale=avatar['scale'].cuda();betas=avatar['betas'].cuda();indices=torch.tensor([16,17,18,19,20,21,1,2,4,5,7,8],device='cuda')
root=torch.nn.Parameter(torch.tensor(initial[:,0],device='cuda',dtype=torch.float32));body=torch.nn.Parameter(torch.tensor(initial[:,1:],device='cuda',dtype=torch.float32));prior_body=body.detach().clone()
with torch.no_grad():
    aa=torch.cat([root[:,None],body,torch.zeros(T,33,3,device='cuda')],1);joints,_=batch_rigid_transform(axis_angle_to_matrix(aa),rest.expand(T,-1,-1),parents)
    translation=torch.tensor(target3[:,6:8].mean(1),device='cuda',dtype=torch.float32)-joints[:,[1,2]].mean(1)*scale
trans=torch.nn.Parameter(translation)
K=torch.tensor([v['K'] for v in meta['views']],device='cuda');W=torch.tensor([v['W'] for v in meta['views']],device='cuda')
xy=torch.tensor(det['xy'][:,:,5:],device='cuda',dtype=torch.float32);score=torch.tensor(det['score'][:,:,5:],device='cuda',dtype=torch.float32);valid=(score>3)&(xy[...,0]>2)&(xy[...,0]<1022)&(xy[...,1]>2)&(xy[...,1]<766)
target3=torch.tensor(target3,device='cuda',dtype=torch.float32);hist=[];best=None
def evaluate():
    aa=torch.cat([root[:,None],body,torch.zeros(T,33,3,device='cuda')],1);joints,_=batch_rigid_transform(axis_angle_to_matrix(aa),rest.expand(T,-1,-1),parents)
    world=joints[:,indices]*scale+trans[:,None];cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],world)+W[:,None,None,:3,3];uv=torch.einsum('vij,vtkj->vtki',K,cam);uv=uv[...,:2]/uv[...,2:].clamp_min(.01)
    error=torch.linalg.vector_norm(uv-xy,dim=-1);visual=(2*(torch.sqrt(1+(error/6)**2)-1)*valid).sum()/valid.sum()
    consistency=.2*torch.nn.functional.smooth_l1_loss(world/.06,target3/.06)
    velocity=(world[1:]-world[:-1])*30;smooth=.08*(velocity[1:]-velocity[:-1]).square().mean()
    prior=.01*(body-prior_body).square().mean()+.05*(body[:,[11,14]]-prior_body[:,[11,14]]).square().mean()
    return visual+consistency+smooth+prior,error,world,uv
opt=torch.optim.Adam([root,body,trans],lr=.01)
for step in range(501):
    opt.zero_grad();loss,error,world,uv=evaluate()
    if best is None or float(loss)<best[0]:best=(float(loss),[x.detach().clone() for x in [root,body,trans]])
    if step<500:loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.01*(1-.9*step/500)
    if step%100==0:
        row={'step':step,'loss':float(loss),'mean_body_error_px1024':float(error[valid].mean())};hist.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    for x,value in zip([root,body,trans],best[1]):x.copy_(value)
    loss,error,world,uv=evaluate();assert torch.isfinite(world).all()
torch.save({'frames':torch.tensor(frames),'root_pose':root.detach().cpu(),'body_pose':body.detach().cpu(),'translation_world':trans.detach().cpu(),'scale':scale.cpu(),'betas':betas.cpu(),'joints_world':world.detach().cpu(),'projected':uv.detach().cpu(),'lhand_pose':torch.zeros(T,15,3),'rhand_pose':torch.zeros(T,15,3)},out/'pose_sequence.pt')
np.savez_compressed(out/'reprojection.npz',observed=xy.cpu(),projected=uv.detach().cpu(),valid=valid.cpu())
for t in [0,T//2,T-1]:
    sheet=Image.new('RGB',(2048,768),'white')
    for i,v in enumerate(meta['views']):
        im=Image.open(ROOT/options.data/v['serial']/f'{frames[t]:04d}.png').convert('RGB');draw=ImageDraw.Draw(im);points=uv[i,t].detach().cpu().numpy()
        for j,(x,y) in enumerate(points):draw.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
        for j,(x,y) in enumerate(xy[i,t].cpu().numpy()):
            if valid[i,t,j]:draw.ellipse((x-2,y-2,x+2,y+2),fill='lime')
        sheet.paste(im.resize((512,384)),(i%4*512,i//4*384))
    sheet.save(out/f'{frames[t]:04d}_all8.jpg')
(out/'report.json').write_text(json.dumps({'success':True,'frames':frames,'views':8,'history':hist,'mean_body_error_px1024':float(error[valid].mean()),'seconds':time.time()-start,'scope':'One shared fixed-shape SMPL-X avatar motion jointly fitted to all8 real body detections, with robust SO(3) NLF priors transformed through original cameras and weak all8 3D-consistency prior. Fixed shared avatar scale, original cameras. Head/neck retain actual NLF pose priors because facial visibility is unverified here. Fingers fixed zero: no fine hand tracking/contact. Not CoCoNet output, not SMPL-H model substitution, not novel-view accuracy proof. Same-index sync provisional.'},indent=2))
