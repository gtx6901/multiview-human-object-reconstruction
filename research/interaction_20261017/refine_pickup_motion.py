"""Joint time regularization of a shared rigid trajectory under real silhouettes."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from pytorch3d.transforms import axis_angle_to_matrix
from mesh_renderer import MeshRenderer
torch.set_num_threads(4);out=ROOT/'helmet_pose_pickup_temporal_v1';out.mkdir(exist_ok=False);start=time.time()
initial=torch.load(ROOT/'helmet_pose_pickup_smoothed_v1/sequence.pt',weights_only=True);base=initial['vertices_local'].cuda();faces=initial['faces'].cuda();seedR=initial['R'].cuda();seedt=initial['translation_world'].cuda();scale=initial['logscale'].cuda().exp();T=len(seedt)
meta=json.loads((ROOT/'helmet_pickup_inspect000_135_v1/manifest.json').read_text());objects_root=ROOT/'helmet_track_pickup000_135_v2';people=ROOT/'people_pickup000_135_v1';objects=json.loads((objects_root/'report.json').read_text())['records'];humans={(r['serial'],r['frame']):r for r in json.loads((people/'report.json').read_text())['records']};views={v['serial']:v for v in meta['views']}
width=256;height=192;s=width/1024;affine=np.array([[s,0,s*.5-.5],[0,s,s*.5-.5],[0,0,1.]])
times=[];cameraK=[];cameraW=[];targets=[];valids=[];distances=[]
for rec in objects:
    if rec['mask'] is None:continue
    v=views[rec['serial']];m=np.asarray(Image.open(objects_root/rec['mask']))>127;hrec=humans[(rec['serial'],rec['frame'])];h=np.asarray(Image.open(people/hrec['mask']))>127 if hrec['mask'] else np.zeros_like(m)
    good=np.asarray(Image.open(ROOT/'helmet_pickup_inspect000_135_v1'/rec['serial']/'valid.png'))>127;roi=np.zeros_like(m);x0,y0,x1,y1=rec['roi_xyxy'];roi[y0:y1,x0:x1]=True;good&=roi&(~h|m)
    times.append(rec['frame']);cameraK.append(affine@np.array(v['K']));cameraW.append(v['W']);targets.append(cv2.resize(m.astype('float32'),(width,height),interpolation=cv2.INTER_AREA));valids.append(cv2.resize(good.astype('float32'),(width,height),interpolation=cv2.INTER_NEAREST));distances.append(cv2.resize(distance_transform_edt(~m).astype('float32')/1024,(width,height)))
times=torch.tensor(times,device='cuda');K=torch.tensor(np.array(cameraK),device='cuda',dtype=torch.float32);W=torch.tensor(np.array(cameraW),device='cuda',dtype=torch.float32);target=torch.tensor(np.array(targets),device='cuda');valid=torch.tensor(np.array(valids),device='cuda');distance=torch.tensor(np.array(distances),device='cuda');N=len(times);renderer=MeshRenderer()
aa=torch.nn.Parameter(torch.zeros(T,3,device='cuda'));translation=torch.nn.Parameter(seedt.clone());opt=torch.optim.Adam([{'params':[aa],'lr':.015},{'params':[translation],'lr':.0015}]);history=[]
def silhouette(begin,end,Rall,tall):
    ids=times[begin:end];world=torch.einsum('bij,nj->bni',Rall[ids],base)*scale+tall[ids,None];alpha=renderer.alpha_batch(world,faces,K[begin:end],W[begin:end],width,height);pred=alpha*valid[begin:end];truth=target[begin:end];inter=(pred*truth).sum((1,2));union=(pred+truth-pred*truth).sum((1,2)).clamp_min(1);iou=inter/union
    cost=1-iou+2*(pred*distance[begin:end]).sum((1,2))/truth.sum((1,2)).clamp_min(1)
    return cost,iou
def stats(R,t):
    r=R.detach().cpu().numpy();xyz=t.detach().cpu().numpy();rel=np.einsum('tij,tkj->tik',r[1:],r[:-1]);angles=np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(),axis=1)*180/np.pi
    return {'max_rotation_step_degrees':float(angles.max()),'max_translation_step_body_height_units':float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).max())}
with torch.inference_mode():
    before_iou=torch.cat([silhouette(b,min(b+48,N),seedR,seedt)[1] for b in range(0,N,48)]).mean().item()
before=stats(seedR,seedt)
for step in range(101):
    opt.zero_grad();fit_cost=0.
    for begin in range(0,N,48):
        Rall=axis_angle_to_matrix(aa)@seedR;cost,_=silhouette(begin,min(begin+48,N),Rall,translation);loss=cost.sum()/N;fit_cost+=float(loss)
        if step<100:loss.backward()
    Rall=axis_angle_to_matrix(aa)@seedR
    rotation_acc=(Rall[2:]-2*Rall[1:-1]+Rall[:-2]).square().mean();rotation_velocity=(Rall[1:]-Rall[:-1]).square().mean();translation_acc=((translation[2:]-2*translation[1:-1]+translation[:-2])/scale).square().mean()
    regularizer=3.*rotation_acc+.02*rotation_velocity+.15*translation_acc+.001*((translation-seedt)/scale).square().mean()
    if step<100:regularizer.backward();opt.step()
    for g,lr in zip(opt.param_groups,[.015,.0015]):g['lr']=lr*(1-.7*step/100)
    if step%25==0:history.append({'step':step,'silhouette_cost':fit_cost,'regularizer':float(regularizer)});print(history[-1],flush=True)
with torch.inference_mode():
    Rall=axis_angle_to_matrix(aa)@seedR;after_iou=torch.cat([silhouette(b,min(b+48,N),Rall,translation)[1] for b in range(0,N,48)]).mean().item();assert torch.isfinite(Rall).all() and torch.isfinite(translation).all()
sequence={**initial,'R':Rall.detach().cpu(),'translation_world':translation.detach().cpu()};torch.save(sequence,out/'sequence.pt');after=stats(Rall,translation)
report={'success':True,'frames':initial['frames'].tolist(),'observed_silhouettes':N,'mean_input_soft_iou_before':before_iou,'mean_input_soft_iou_after':after_iou,'before':before,'after':after,'history':history,'seconds':time.time()-start,'scope':'All times jointly refined with actual multicamera masks and temporal matrix-rotation/translation second-difference penalties; fixed smoothed silhouette hull and scale. No invented object observations, no attachment to hands/head, no exact physical dynamics or orientation truth. Temporal priors regularize poorly observed frames, including frame7. Resolution256 objective is not directly comparable to prior384 metrics.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('joint temporal refinement complete',flush=True)
