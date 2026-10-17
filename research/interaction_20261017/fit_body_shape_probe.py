"""Conditional shared shape/scale diagnostic, no accepted body replacement."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from avatar_rig import pose_axis_angles
out=ROOT/'body_shape_conditional_probe_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
assert json.loads((ROOT/'shaped_surface_verification_v1/report.json').read_text())['success']
base=torch.load(ROOT/'body_head_faces_all300_v1/pose_sequence.pt',weights_only=True);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());det=np.load(ROOT/'observations_all300_v1/body.npz');model=SMPLXSurfaceLandmarks().cuda()
aa=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(300)]).cuda();xy=torch.tensor(det['xy'][:,:,5:17],device='cuda',dtype=torch.float32);score=torch.tensor(det['score'][:,:,5:17],device='cuda');valid=(score>3)&(xy[...,0]>2)&(xy[...,0]<1022)&(xy[...,1]>2)&(xy[...,1]<766)
A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
K=torch.tensor(np.array([A@np.array(v['K']) for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32);s0=base['scale'].cuda();t0=base['translation_world'].cuda()
train=torch.arange(0,300,10,device='cuda');test=torch.tensor([t for t in range(300) if t%10],device='cuda');allids=torch.arange(300,device='cuda');rawbeta=torch.nn.Parameter(torch.zeros(1,10,device='cuda'));rawscale=torch.nn.Parameter(torch.zeros((),device='cuda'))
def parameters():return 3*torch.tanh(rawbeta),s0*torch.exp(.08*torch.tanh(rawscale))
def evaluate(ids):
 beta,scale=parameters();root=model.rest[0]+torch.einsum('bl,cl->bc',beta,model.joint_shapedirs[0])[0]
 translation=t0[ids]+model.rest[0]*s0-root*scale
 result=model(aa[ids],scale,translation,betas=beta);world=result['coco17'][:,5:17];cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],world)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,cam);uv=q[...,:2]/q[...,2:].clamp_min(.01);error=(uv-xy[:,ids]).norm(dim=-1);mask=valid[:,ids]&(cam[...,2]>.01)
 data=(2*(torch.sqrt(1+(error/6)**2)-1)*mask).sum()/mask.sum();regularizer=.02*beta.square().mean()+.1*torch.log(scale/s0).square();return data+regularizer,error,mask,translation,result
with torch.no_grad():_,before,mask,_,_=evaluate(allids);before=before.clone()
opt=torch.optim.Adam([rawbeta,rawscale],lr=.025);best=None;history=[]
for step in range(501):
 opt.zero_grad();loss,error,mask,_,_=evaluate(train);value=float(loss.detach())
 if best is None or value<best[0]:best=(value,rawbeta.detach().clone(),rawscale.detach().clone())
 if step<500:loss.backward();opt.step()
 if step%100==0:history.append({'step':step,'train_error_px1024':float(error[mask].mean().detach()),'loss':value});print(json.dumps(history[-1]),flush=True)
with torch.no_grad():
 rawbeta.copy_(best[1]);rawscale.copy_(best[2]);_,after,mask,translation,result=evaluate(allids);beta,scale=parameters();assert torch.isfinite(after).all()
 torch.save({'betas':beta.cpu(),'scale':scale.cpu(),'translation_world':translation.cpu(),'joints55_world':result['joints55'].cpu(),'source_poses':'body_head_faces_all300_v1'},out/'shape_candidate.pt')
 def metrics(ids):
  m=mask[:,ids];return {'before_px1024':float(before[:,ids][m].mean()),'after_px1024':float(after[:,ids][m].mean()),'observation_count':int(m.sum())}
 views=[]
 for i,v in enumerate(meta['views']):views.append({'serial':v['serial'],'before_px1024':float(before[i][mask[i]].mean()),'after_px1024':float(after[i][mask[i]].mean())})
 report={'success':True,'seconds':time.time()-start,'training_frames':train.cpu().tolist(),'test_time_frames_count':len(test),'train':metrics(train),'other_times':metrics(test),'all':metrics(allids),'per_camera':views,'beta_norm':float(beta.norm()),'maximum_beta_abs':float(beta.abs().max()),'scale_ratio_to_accepted':float(scale/s0),'history':history,'scope':'Conditional shared10-beta/scale fitting using all8 COCO body observations, train every10th time and diagnose other times. Accepted55-joint rotations FIXED, pelvis world path exactly preserved by translation compensation; shape±3 and scale±8percent bounded. Neither accepted avatar nor motion overwritten. Fixed poses were previously fitted on all frames, so other-times numbers are NOT independent generalization evidence. Shape may absorb detector, pose, clothing and calibration bias; not true anthropometry. Nonzero shape not yet propagated through original LHM neutral rig/query/appearance or rerendered. No global shape adoption from input residual alone.'}
 (out/'report.json').write_text(json.dumps(report,indent=2));print('conditional body shape probe complete',flush=True)
