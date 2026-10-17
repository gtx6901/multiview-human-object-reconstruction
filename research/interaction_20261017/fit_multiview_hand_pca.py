"""Shared SMPL-X hand PCA fitted to screened native eight-view observations."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='body_hands_pickup000_135_v2');p.add_argument('--body',default='body_surface_pickup000_135_v2');p.add_argument('--screen',default='hands_multiview_screen000_135_v2');p.add_argument('--observations-report',default='wholebody_native000_135_v1/report.json');p.add_argument('--body-detections-file',default='detections_pickup000_135_v1/detections.npz');p.add_argument('--data',default='helmet_pickup_inspect000_135_v1');options=p.parse_args()
torch.set_num_threads(4);out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
model=SMPLXSurfaceLandmarks().cuda();base=torch.load(ROOT/options.body/'pose_sequence.pt',weights_only=True)
screen=np.load(ROOT/options.screen/'screened.npz');meta=json.loads((ROOT/options.observations_report).read_text())
T=len(base['frames']);assert screen['frames'].tolist()==base['frames'].tolist()
root=base['root_pose'].cuda();body0=base['body_pose'].cuda();trans=base['translation_world'].cuda();scale=base['scale'].cuda()
K=torch.tensor([v['K'] for v in meta['views']],device='cuda',dtype=torch.float32);W=torch.tensor([v['W'] for v in meta['views']],device='cuda',dtype=torch.float32)
xy=torch.tensor(screen['xy'],device='cuda',dtype=torch.float32);valid=torch.tensor(screen['inlier'],device='cuda')
weights=torch.tensor(np.minimum(screen['raw_score']/5,1),device='cuda',dtype=torch.float32)*valid
bodydet=np.load(ROOT/options.body_detections_file)
bodyxy=(torch.tensor(bodydet['xy'][:,:,5:17],device='cuda',dtype=torch.float32)+.5)*2-.5
bx=bodydet['xy'][:,:,5:17];bodyvalid=torch.tensor((bodydet['score'][:,:,5:17]>3)&(bx[...,0]>2)&(bx[...,0]<1022)&(bx[...,1]>2)&(bx[...,1]<766),device='cuda')
rawpca=torch.nn.Parameter(torch.zeros(T,2,12,device='cuda'));rawarm=torch.nn.Parameter(torch.zeros(T,4,3,device='cuda'))
history=[];best=None
def project(points):
    pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],points)+W[:,None,None,:3,3];uv=torch.einsum('vij,vtkj->vtki',K,pc);return uv[...,:2]/uv[...,2:].clamp_min(.01)
def evaluate():
    pca=3*torch.tanh(rawpca/3)
    left=(pca[:,0]@model.left_pca[:12]).reshape(T,15,3);right=(pca[:,1]@model.right_pca[:12]).reshape(T,15,3)
    delta=torch.tanh(rawarm)*torch.tensor([.4,.4,1.,1.],device='cuda')[None,:,None]
    body=torch.cat([body0[:,:17],body0[:,17:21]+delta],1)
    aa=torch.cat([root[:,None],body,torch.zeros(T,3,3,device='cuda'),left,right],1)
    result=model(aa,scale,trans);uv=project(result['hands21'].reshape(T,42,3));error=(uv-xy).norm(dim=-1)
    loss=(2*(torch.sqrt(1+(error/12)**2)-1)*weights).sum()/weights.sum()
    berror=(project(result['coco17'][:,5:17])-bodyxy).norm(dim=-1)
    loss+=.4*(2*(torch.sqrt(1+(berror/12)**2)-1)*bodyvalid).sum()/bodyvalid.sum()
    loss+=.005*pca.square().mean()+.03*delta.square().mean()
    loss+=.025*(pca[2:]-2*pca[1:-1]+pca[:-2]).square().mean()+.01*(pca[1:]-pca[:-1]).square().mean()
    loss+=.1*(delta[2:]-2*delta[1:-1]+delta[:-2]).square().mean()
    return loss,error,berror,uv,body,left,right,pca,result
with torch.no_grad():
    _,before,beforebody,*_=evaluate();before=before.clone();beforebody=beforebody.clone()
opt=torch.optim.Adam([rawpca,rawarm],lr=.035)
for step in range(401):
    opt.zero_grad();values=evaluate();loss,error,berror=values[:3]
    if best is None or float(loss.detach())<best[0]:best=(float(loss.detach()),rawpca.detach().clone(),rawarm.detach().clone())
    if step<400:loss.backward();opt.step()
    for group in opt.param_groups:group['lr']=.035*(1-.9*step/400)
    if step%100==0:
        row={'step':step,'loss':float(loss.detach()),'hand_error_native_px':float(error[valid].mean().detach()),'body_error_native_px':float(berror[bodyvalid].mean().detach())};history.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    rawpca.copy_(best[1]);rawarm.copy_(best[2]);_,error,berror,uv,body,left,right,pca,result=evaluate()
    assert all(torch.isfinite(x).all() for x in [body,left,right,uv]) and left.abs().max()>0 and right.abs().max()>0
    poses=dict(base);poses.update(body_pose=body.cpu(),left_hand_pose=left.cpu(),right_hand_pose=right.cpu(),lhand_pose=left.cpu(),rhand_pose=right.cpu(),hand_pca=pca.cpu(),joints55_world=result['joints55'].cpu(),joints_world=result['coco17'][:,5:17].cpu())
    # Body projections must use the original 1024-pixel contract in this bundle.
    poses['projected']=(project(result['coco17'][:,5:17]).cpu()+.5)/2-.5
    torch.save(poses,out/'pose_sequence.pt')
    np.savez_compressed(out/'reprojection.npz',observed=xy.cpu(),projected=uv.cpu(),valid=valid.cpu(),before_error=before.cpu(),after_error=error.cpu())
    for t in sorted(set([0,T//4,T//2,3*T//4,T-1])):
        sheet=Image.new('RGB',(4*512,2*384),'white')
        for v,view in enumerate(meta['views']):
            im=Image.open(ROOT/options.data/view['serial']/f'{t:04d}.png').convert('RGB').resize((1024,768));draw=ImageDraw.Draw(im)
            for j,(x,y) in enumerate(((xy[v,t].cpu().numpy()+.5)/2-.5)):
                if valid[v,t,j]:draw.ellipse((x-2,y-2,x+2,y+2),fill='lime')
            for x,y in ((uv[v,t].cpu().numpy()+.5)/2-.5):draw.ellipse((x-2,y-2,x+2,y+2),outline='red')
            sheet.paste(im.resize((512,384)),(v%4*512,v//4*384))
        sheet.save(out/f'{t:04d}_all8_hands.jpg')
    perview=[{'serial':v['serial'],'points':int(valid[i].sum()),'before_native_px':float(before[i][valid[i]].mean()),'after_native_px':float(error[i][valid[i]].mean())} for i,v in enumerate(meta['views'])]
    report={'success':True,'frames':T,'views':8,'seconds':time.time()-start,'history':history,'hand_error_before_native_px':float(before[valid].mean()),'hand_error_after_native_px':float(error[valid].mean()),'body_error_before_native_px':float(beforebody[bodyvalid].mean()),'body_error_after_native_px':float(berror[bodyvalid].mean()),'per_camera':perview,'hand_pca_components':12,'flat_hand_mean':True,'max_abs_hand_axis_angle':float(torch.cat([left,right],1).abs().max()),'scope':'Shared twelve-component PCA per hand with small elbow/wrist corrections, using only geometrically screened real eight-view native RTMW pseudo-observations. Confidence weights are capped raw-score weights, not visibility probabilities. Frozen body root/translation/shape/scale and calibrated cameras; body landmarks regularize arm corrections. Temporal hand priors fill weak frames but are not measurements. Input residual improvement is not held-out or physical-contact accuracy. Avatar rendering must still be audited.'}
    report['sources']=vars(options)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['success','hand_error_before_native_px','hand_error_after_native_px','body_error_after_native_px','seconds']}),flush=True)
