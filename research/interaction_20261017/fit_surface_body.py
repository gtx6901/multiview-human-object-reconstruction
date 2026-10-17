"""Refit fixed-shape shared body motion using true COCO surface landmarks."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='body_surface_pickup000_135_v2');p.add_argument('--steps',type=int,default=500)
p.add_argument('--data',default='helmet_pickup_inspect000_135_v1');p.add_argument('--detections-file',default='detections_pickup000_135_v1/detections.npz');p.add_argument('--initial',default='body_fused_pickup000_135_v1')
p.add_argument('--exclude-camera',default='')
options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
assert json.loads((ROOT/'surface_landmark_verification_v1/report.json').read_text())['success']
meta=json.loads((ROOT/options.data/'manifest.json').read_text());det=np.load(ROOT/options.detections_file)
if options.exclude_camera:
    selected=[i for i,v in enumerate(meta['views']) if v['serial']!=options.exclude_camera]
    assert len(selected)==len(meta['views'])-1
    det={k:det[k] for k in det.files}
    for key in ['xy','score']:det[key]=det[key][selected]
    meta['views']=[meta['views'][i] for i in selected]
base=torch.load(ROOT/options.initial/'pose_sequence.pt',weights_only=True);frames=base['frames'].tolist();T=len(frames)
assert frames==det['frames'].tolist() and torch.count_nonzero(base['betas'])==0
model=SMPLXSurfaceLandmarks().cuda();scale=base['scale'].cuda()
root0=base['root_pose'].cuda().clone();body0=base['body_pose'].cuda().clone();trans0=base['translation_world'].cuda().clone()
root=torch.nn.Parameter(torch.zeros_like(root0));body=torch.nn.Parameter(torch.zeros_like(body0));translation=torch.nn.Parameter(trans0.clone())
pose_mask=torch.ones(1,21,1,device='cuda');pose_mask[:,[6,7,9,10,11,14,19,20]]=0
def current_pose():return root0+.15*torch.tanh(root),body0+.35*torch.tanh(body)*pose_mask
xy=torch.tensor(det['xy'][:,:,5:17],device='cuda',dtype=torch.float32);score=torch.tensor(det['score'][:,:,5:17],device='cuda',dtype=torch.float32)
valid=(score>3)&(xy[...,0]>2)&(xy[...,0]<1022)&(xy[...,1]>2)&(xy[...,1]<766)
Klist=[]
for v in meta['views']:
    sx=1024/v['width'];sy=768/v['height'];A=np.array([[sx,0,.5*sx-.5],[0,sy,.5*sy-.5],[0,0,1.]])
    Klist.append((A@np.array(v['K'])).tolist())
K=torch.tensor(Klist,device='cuda',dtype=torch.float32);W=torch.tensor([v['W'] for v in meta['views']],device='cuda',dtype=torch.float32)
def project(world):
    cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],world)+W[:,None,None,:3,3]
    q=torch.einsum('vij,vtkj->vtki',K,cam);return q[...,:2]/q[...,2:].clamp_min(.01)
def evaluate():
    current_root,current_body=current_pose()
    aa=torch.cat([current_root[:,None],current_body,torch.zeros(T,33,3,device='cuda')],1)
    result=model(aa,scale,translation);world=result['coco17'][:,5:17];uv=project(world);error=(uv-xy).norm(dim=-1)
    data=(2*(torch.sqrt(1+(error/6)**2)-1)*valid).sum()/valid.sum()
    acceleration=world[2:]-2*world[1:-1]+world[:-2]
    regularization=.3*(current_body-body0).square().mean()+1.*(current_root-root0).square().mean()+.1*((translation-trans0)/scale).square().mean()+72*acceleration.square().mean()
    return data+regularization,error,world,uv,result['joints55']
with torch.no_grad():
    _,before,_,_,_=evaluate();before=before.clone()
    rotation_centers=project(model(torch.cat([root0[:,None],body0,torch.zeros(T,33,3,device='cuda')],1),scale,translation)['joints55'][:,[16,17,18,19,20,21,1,2,4,5,7,8]])
    center_before=(rotation_centers-xy).norm(dim=-1)
opt=torch.optim.Adam([root,body,translation],lr=.03);best=None;history=[]
for step in range(options.steps+1):
    opt.zero_grad();loss,error,world,uv,joints=evaluate()
    if best is None or float(loss.detach())<best[0]:best=(float(loss.detach()),[x.detach().clone() for x in [root,body,translation]])
    if step<options.steps:loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.03*(1-.9*step/options.steps)
    if step%100==0:
        row={'step':step,'loss':float(loss.detach()),'surface_error_px1024':float(error[valid].mean().detach())};history.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    for x,b in zip([root,body,translation],best[1]):x.copy_(b)
    _,error,world,uv,joints=evaluate();assert torch.isfinite(world).all()
    current_root,current_body=current_pose()
    assert torch.equal(current_body[:,[6,7,9,10,11,14,19,20]],body0[:,[6,7,9,10,11,14,19,20]])
    poses=dict(base);poses.update(root_pose=current_root.cpu(),body_pose=current_body.cpu(),translation_world=translation.cpu(),joints_world=world.cpu(),projected=uv.cpu(),joints55_world=joints.cpu())
    torch.save({k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in poses.items()},out/'pose_sequence.pt')
    np.savez_compressed(out/'reprojection.npz',observed=xy.cpu(),before_surface_error=before.cpu(),after_surface_error=error.cpu(),projected=uv.cpu(),valid=valid.cpu())
    perview=[{'serial':v['serial'],'rotation_center_error_before':float(center_before[i][valid[i]].mean()),'surface_error_before':float(before[i][valid[i]].mean()),'surface_error_after':float(error[i][valid[i]].mean())} for i,v in enumerate(meta['views'])]
    perjoint=[{'coco_index':j+5,'before':float(before[:,:,j][valid[:,:,j]].mean()),'after':float(error[:,:,j][valid[:,:,j]].mean())} for j in range(12)]
    for t in [0,T//2,T-1]:
        sheet=Image.new('RGB',(2048,768),'white')
        for i,v in enumerate(meta['views']):
            im=Image.open(ROOT/options.data/v['serial']/f'{frames[t]:04d}.png').convert('RGB').resize((1024,768));draw=ImageDraw.Draw(im)
            for j,(x,y) in enumerate(xy[i,t].cpu().numpy()):
                if valid[i,t,j]:draw.ellipse((x-3,y-3,x+3,y+3),fill='lime')
            for x,y in uv[i,t].cpu().numpy():draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2)
            sheet.paste(im.resize((512,384)),(i%4*512,i//4*384))
        sheet.save(out/f'{t:04d}_all8.jpg')
    report={'success':True,'frames':frames,'views':8,'seconds':time.time()-start,'history':history,'mean_rotation_center_error_before':float(center_before[valid].mean()),'mean_surface_error_before':float(before[valid].mean()),'mean_surface_error_after':float(error[valid].mean()),'per_camera':perview,'per_joint':perjoint,'landmarks':'COCO17 regressed from posed SMPL-X surface, fixed neutral shape, upstream topology mapping','scope':'All8 input landmark fitting diagnostic; before/after surface residual uses identical observations/definition. Rotation-center residual is a separate old-definition metric, not directly comparable. Not held-out novel-view evidence. No new hand poses or camera calibration change.'}
    report['pose_constraints']={'max_root_component_correction_rad':.15,'max_body_component_correction_rad':.35,'unchanged_body_joint_ids':[7,8,10,11,12,15,20,21],'reason':'V1 visually regressed unconstrained head/ankle/wrist rotations despite smaller input landmark loss. V2 preserves those baseline joints and bounds observed-joint corrections.'}
    report['views']=len(meta['views']);report['excluded_camera']=options.exclude_camera
    if options.exclude_camera:report['scope']=report['scope'].replace('All8 input','Seven-view training input')+' Held-out observation entries removed before fitting. Initializer provenance must be audited separately.'
    report['sources']={'data':options.data,'detections_file':options.detections_file,'initial':options.initial,'observation_pixel_coordinates':'1024x768 undistorted; camera K explicitly mapped by pixel-center affine from dataset resolution'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['success','mean_surface_error_before','mean_surface_error_after','seconds']}),flush=True)
