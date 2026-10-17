"""Data-shared SMPL-X beta and time-varying body motion with observed hands."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from avatar_rig import pose_axis_angles
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--zero-shape',action='store_true');p.add_argument('--steps',type=int,default=300);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
base=torch.load(ROOT/'body_head_faces_all300_v1/pose_sequence.pt',weights_only=True);model=SMPLXSurfaceLandmarks().cuda();T=len(base['frames']);assert T==300
shape=torch.load(ROOT/'body_shape_conditional_probe_v1/shape_candidate.pt',weights_only=True)
aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(T)]).cuda();root=torch.nn.Parameter(aa0[:,0].clone());body=torch.nn.Parameter(aa0[:,1:22].clone())
beta0=torch.zeros(1,10) if o.zero_shape else shape['betas'];rawbeta=torch.nn.Parameter(torch.atanh((beta0.cuda()/3).clamp(-.999,.999)),requires_grad=not o.zero_shape)
s0=base['scale'].cuda();initial_scale=s0 if o.zero_shape else shape['scale'].cuda();rawscale=torch.nn.Parameter(torch.atanh((torch.log(initial_scale/s0)/.15).clamp(-.999,.999)))
trans=torch.nn.Parameter((base['translation_world'] if o.zero_shape else shape['translation_world']).cuda().clone())
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
K=torch.tensor(np.array([A@np.array(v['K']) for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
d=np.load(ROOT/'observations_all300_v1/body.npz');xy=torch.tensor(d['xy'][:,:,5:17],device='cuda',dtype=torch.float32);valid=torch.tensor((d['score'][:,:,5:17]>3)&(d['xy'][:,:,5:17,0]>2)&(d['xy'][:,:,5:17,0]<1022)&(d['xy'][:,:,5:17,1]>2)&(d['xy'][:,:,5:17,1]<766),device='cuda')
h=np.load(ROOT/'hands_multiview_screen_all300_v1/screened.npz');assert h['frames'].tolist()==base['frames'].tolist();hxy=torch.tensor(h['xy']*.5-.25,device='cuda',dtype=torch.float32);hw=torch.tensor(np.minimum(h['raw_score']/5,1)*h['inlier'],device='cuda',dtype=torch.float32)
def project(world):
 cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],world)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,cam);return q[...,:2]/q[...,2:].clamp_min(.01)
def evaluate():
 beta=3*rawbeta.tanh();scale=s0*torch.exp(.15*rawscale.tanh());aa=torch.cat([root[:,None],body,aa0[:,22:]],1);r=model(aa,scale,trans,betas=beta)
 error=(project(r['coco17'][:,5:17])-xy).norm(dim=-1);he=(project(r['hands21'].reshape(T,42,3))-hxy).norm(dim=-1)
 robust=lambda e:2*(torch.sqrt(1+(e/6)**2)-1)
 data=(robust(error)*valid).sum()/valid.sum()+.3*(robust(he)*hw).sum()/hw.sum().clamp_min(1)
 poseprior=.1*(body-aa0[:,1:22]).square().mean()+.1*(root-aa0[:,0]).square().mean()
 world=r['joints55'];acc=(world[2:]-2*world[1:-1]+world[:-2])*30**2;smooth=.001*acc.square().mean()
 loss=data+poseprior+smooth+.02*beta.square().mean()+.1*torch.log(scale/s0).square()
 return loss,error,he,r,beta,scale
groups=[{'params':[root,body],'lr':.002},{'params':[trans],'lr':.0005},{'params':[rawscale],'lr':.005}]
if not o.zero_shape:groups.append({'params':[rawbeta],'lr':.01})
opt=torch.optim.Adam(groups);history=[];best=None
with torch.no_grad():_,before,hbefore,_,_,_=evaluate();before=before.clone();hbefore=hbefore.clone()
for step in range(o.steps+1):
 opt.zero_grad();loss,error,he,r,beta,scale=evaluate();value=float(loss.detach())
 if best is None or value<best[0]:best=(value,[x.detach().clone() for x in [root,body,trans,rawbeta,rawscale]])
 if step%100==0 or step==o.steps:
  row={'step':step,'loss':value,'body_error_px1024':float(error[valid].mean().detach()),'hand_error_px1024':float((he*hw).sum().detach()/hw.sum())};history.append(row);print(json.dumps(row),flush=True)
 if step<o.steps:loss.backward();opt.step()
with torch.no_grad():
 for x,y in zip([root,body,trans,rawbeta,rawscale],best[1]):x.copy_(y)
 _,error,he,r,beta,scale=evaluate();result={**base,'root_pose':root.cpu(),'body_pose':body.cpu(),'translation_world':trans.cpu(),'betas':beta.cpu(),'scale':scale.cpu(),'joints55_world':r['joints55'].cpu()}
 torch.save(result,out/'pose_sequence.pt');torch.save({'betas':beta.cpu(),'scale':scale.cpu(),'translation_world':trans.cpu(),'source_poses':o.out},out/'shape_candidate.pt')
 np.savez_compressed(out/'reprojection.npz',observed=xy.cpu(),projected=project(r['coco17'][:,5:17]).cpu(),valid=valid.cpu())
 views=[{'serial':v['serial'],'before_px1024':float(before[i][valid[i]].mean()),'after_px1024':float(error[i][valid[i]].mean())} for i,v in enumerate(meta['views'])]
report={'success':True,'zero_shape_control':o.zero_shape,'frames':T,'views':views,'history':history,'beta_norm':float(beta.norm()),'scale_ratio':float(scale/s0),'seconds':time.time()-start,'scope':'Shared10-beta/scale and all300 body/root/translation jointly fitted to all8 surface COCO observations and previously screened21handpoints; fingers/head jaw/eyes inherited as priors. Shape±3 and scale±15percent bounds, weak pose and temporal acceleration priors. Joint fit can absorb detection/calibration errors; input residuals are not independent geometry truth. Zero-beta control uses same motion fitter. Must export through actual LHM dense shape blend and shaped rest rig and rerender before adoption; object motion not yet refitted and prior helmet attachment is not valid evidence of contact.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('shared shape/motion candidate complete',flush=True)
