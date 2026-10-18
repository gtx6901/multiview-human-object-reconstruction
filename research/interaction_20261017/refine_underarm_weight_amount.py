"""Fit only supported sleeve skinning correction amounts; freeze appearance."""
import os
from pathlib import Path
import sys,json,argparse,importlib.util,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate,pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=96);o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
b=torch.load(ROOT/'temporal_body_integrated_v3/avatar.pt',weights_only=True);source=torch.load(ROOT/'axial_temporal_hands_v3/avatar.pt',weights_only=True)
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 return x
rig=move(b['rig']);poses=move(b['poses']);features=b['features'].cuda().float();bg=b['background_embedding'][2].cuda().float();base=source['rig']['weights'].cuda();target=rig['weights'];ids=torch.where((base!=target).any(-1))[0];assert len(ids)==1219
raw=torch.nn.Parameter(torch.full((len(ids),1),2.,device='cuda'));opt=torch.optim.Adam([raw],lr=.05)
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**b['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(b['decoder'],strict=True,assign=True);decoder.eval().cuda().requires_grad_(False)
frames=[145,165,200,225];cache=ROOT/'overhead_body_garment_observations_all300_v1';records=[];w,h=json.loads((cache/'report.json').read_text())['resolution'];model=SMPLXSurfaceLandmarks().cuda()
with torch.no_grad():world=model(torch.cat([pose_axis_angles(poses,f,'cuda') for f in frames]),poses['scale'],poses['translation_world'][frames],poses['betas'])['coco17']
for path in sorted(cache.glob('*.npz')):
 with np.load(path) as d:
  K=torch.tensor(d['K'],device='cuda');W=torch.tensor(d['W'],device='cuda')
  for j,f in enumerate(frames):
   i=d['frames'].tolist().index(f);rgb=d['target_rgb_u8'][i].astype('float32')/255;support=d['fixed_initial_support'][i].copy();pc=world[j]@W[:3,:3].T+W[:3,3];q=pc@K.T;uv=(q[:,:2]/q[:,2:]).cpu().numpy();torso=np.linalg.norm((uv[5]+uv[6]-uv[11]-uv[12])/2);roi=np.zeros((h,w),np.uint8)
   for sh,hip in [(5,11),(6,12)]:cv2.circle(roi,tuple(np.round(.75*uv[sh]+.25*uv[hip]).astype(int)),max(2,int(.22*torso)),1,-1)
   support&=(roi>0)&(rgb.max(-1)<250/255)
   if support.sum()>=32:records.append({'serial':path.stem,'frame':f,'K':K,'W':W,'rgb':torch.tensor(rgb,device='cuda'),'mask':torch.tensor(support,device='cuda')})
assert len(records)>=24
def weights():
 amount=raw.sigmoid();value=base[ids]+amount*(target[ids]-base[ids]);return base.index_copy(0,ids,value)
def render(r):
 g=animate({**rig,'weights':weights()},poses,r['frame']);color,_,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=r['W'][None],Ks=r['K'][None],width=w,height=h,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,near_plane=.01,far_plane=100.,eps2d=.3)
 with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,h,w)
 return rgb[0,0].float()
gains={};before=[]
with torch.no_grad():
 for r in records:
  rgb=render(r);m=r['mask'];gain=(r['rgb'][m].median(0).values/rgb[m].median(0).values.clamp_min(.1)).clamp(.5,2);r['gain_probe']=gain
 for serial in sorted({r['serial'] for r in records}):gains[serial]=torch.stack([r['gain_probe'] for r in records if r['serial']==serial]).median(0).values
 for r in records:before.append(float((render(r)*gains[r['serial']]-r['rgb']).abs()[r['mask']].mean()))
history=[];order=np.random.default_rng(174).permutation(len(records))
for step in range(o.steps):
 r=records[int(order[step%len(order)])];opt.zero_grad();rgb=render(r);photo=(rgb*gains[r['serial']]-r['rgb']).abs()[r['mask']].mean();prior=.01*(1-raw.sigmoid()).square().mean();loss=photo+prior;loss.backward();assert raw.grad is not None and torch.isfinite(raw.grad).all();opt.step()
 if step%16==0 or step+1==o.steps:
  history.append({'step':step+1,'photo':float(photo.detach()),'amount_mean':float(raw.sigmoid().mean().detach())});torch.save({'raw':raw.detach().cpu(),'optimizer':opt.state_dict(),'step':step+1},out/'checkpoint.pt');print(json.dumps(history[-1]),flush=True)
with torch.no_grad():
 after=[float((render(r)*gains[r['serial']]-r['rgb']).abs()[r['mask']].mean()) for r in records];new=weights().cpu();assert torch.equal(new[~(base.cpu()!=target.cpu()).any(-1)],base.cpu()[~(base.cpu()!=target.cpu()).any(-1)]);torch.save({**b,'rig':{**b['rig'],'weights':new}},out/'avatar.pt')
report={'success':True,'frames':frames,'observations':len(records),'updates':o.steps,'edited_gaussians':len(ids),'before_photo_mae':float(np.mean(before)),'after_photo_mae':float(np.mean(after)),'amount_mean':float(raw.sigmoid().mean()),'amount_min':float(raw.sigmoid().min()),'amount_max':float(raw.sigmoid().max()),'camera_gains_srgb':{k:v.tolist() for k,v in gains.items()},'history':history,'seconds':time.time()-start,'scope':'Only1219 source-supported dark sleeve Gaussian skinning amounts learned between original and proposed arm-only smoothed weights. Positions/features/decoder/poses fixed; rendering geometry changes only through skinning.32 all8x4 native-crop underarm observations; source clipping/person/object reliability masks fixed; camera gains fixed before training. Photo training loss not visual acceptance; same20frame review required.'};(out/'report.json').write_text(json.dumps(report,indent=2));print('local sleeve weights fit complete',flush=True)
