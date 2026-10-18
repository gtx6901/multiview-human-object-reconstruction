"""Rasterize chain weights/dark marker, attribute real underarm mismatch pixels."""
import os
from pathlib import Path
import sys,json,argparse,importlib.util,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from avatar_rig import animate,pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='avatar_grounded_shared_shape_v2');p.add_argument('--render',default='grounded_shared_shape_review20_v2');p.add_argument('--canonical-render',default='grounded_shape_canonical_views_v3');p.add_argument('--correct-out');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True)
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 return x
rig=move(b['rig']);poses=move(b['poses']);weights=rig['weights'];arms=weights[:,[13,14,16,17,18,19]].sum(-1);spine=weights[:,[0,3,6,9]].sum(-1)
# Coarse SH RGB is an auxiliary output, not the final decoded garment color.
# Label actual neural A-pose renders with depth-screened multi-view votes.
cr=ROOT/o.canonical_render;cm=json.loads((cr/'cameras.json').read_text());cp=move(torch.load(cr/'pose.pt',weights_only=True));votes=torch.zeros(len(weights),device='cuda');seen=votes.clone()
with torch.no_grad():
 cg=animate(rig,cp,0)
 for c in cm['views']:
  ck=torch.tensor(c['K'],device='cuda',dtype=torch.float32);cw=torch.tensor(c['W'],device='cuda',dtype=torch.float32);ww,hh=cm['width'],cm['height'];pc=cg['xyz']@cw[:3,:3].T+cw[:3,3];q=pc@ck.T;ix=torch.round(q[:,:2]/q[:,2:]).long();good=(pc[:,2]>.01)&(ix[:,0]>=0)&(ix[:,0]<ww)&(ix[:,1]>=0)&(ix[:,1]<hh);ix[:,0].clamp_(0,ww-1);ix[:,1].clamp_(0,hh-1)
  rd,al,_=rasterization(means=cg['xyz'],quats=cg['rotation'],scales=cg['scaling'],opacities=cg['opacity'].flatten(),colors=torch.zeros(len(weights),1,device='cuda'),viewmats=cw[None],Ks=ck[None],width=ww,height=hh,backgrounds=torch.zeros(1,1,device='cuda'),packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
  depth=rd[0,...,-1][ix[:,1],ix[:,0]];good&=((depth-pc[:,2]).abs()<.008*poses['scale'])&(al[0,...,0][ix[:,1],ix[:,0]]>.8)
  rgb=torch.tensor(np.array(Image.open(cr/c['rgb']).convert('RGB')),device='cuda',dtype=torch.float32)/255;lum=rgb[ix[:,1],ix[:,0]].mean(-1);seen+=good;votes+=good*(lum<.45)
dark=((seen>=3)&(votes/seen.clamp_min(1)>.65)).float();mixed=(arms>.1)&(spine>.1);candidate=(dark>.5)&mixed
cams=json.loads((ROOT/o.render/'cameras.json').read_text());width,height=cams['width'],cams['height'];meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());native={v['serial']:v for v in meta['views']};frames=[145,165,200,225];model=SMPLXSurfaceLandmarks().cuda();rows=[];support=torch.zeros(len(weights),device='cuda')
with torch.no_grad():
 for f in frames:
  g=animate(rig,poses,f);land=model(pose_axis_angles(poses,f,'cuda'),poses['scale'],poses['translation_world'][f:f+1],betas=poses['betas'])['coco17'][0]
  for c in cams['views'][:8]:
   serial=c['label'];K=torch.tensor(c['K'],device='cuda',dtype=torch.float32);W=torch.tensor(c['W'],device='cuda',dtype=torch.float32)
   rgb,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.stack([arms,spine,dark],-1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.zeros(1,3,device='cuda'),packed=False,near_plane=.01,far_plane=100.,eps2d=.3)
   field=(rgb[0]/alpha[0].clamp_min(1e-5)).clamp(0,1);arr=field.cpu().numpy();a=alpha[0,...,0].cpu().numpy()
   n=native[serial];H=np.array(c['K'])@np.linalg.inv(np.array(n['K']));src=cv2.warpAffine(np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB')),H[:2],(width,height),borderValue=(255,255,255));pred=np.array(Image.open(ROOT/o.render/f'{f:04d}_{serial}.png').convert('RGB'))
   pc=land@W[:3,:3].T+W[:3,3];uv=pc@K.T;uv=(uv[:,:2]/uv[:,2:]).cpu().numpy();torso=np.linalg.norm((uv[5]+uv[6]-uv[11]-uv[12])/2);roi=np.zeros((height,width),np.uint8)
   for sh,hip in [(5,11),(6,12)]:cv2.circle(roi,tuple(np.round(.75*uv[sh]+.25*uv[hip]).astype(int)),max(2,int(.18*torso)),1,-1)
   mask=(roi>0)&(a>.8)&(src.mean(-1)>170)&(pred.mean(-1)<120)&(src.max(-1)<250)
   # Influence map gradient attributes a masked mixed-dark rendering to centers.
   means=g['xyz']@W[:3,:3].T+W[:3,3];q=means@K.T;ix=torch.round(q[:,:2]/q[:,2:]).long();inside=(means[:,2]>.01)&(ix[:,0]>=0)&(ix[:,0]<width)&(ix[:,1]>=0)&(ix[:,1]<height);idx=ix.clamp_min(0);idx[:,0].clamp_max_(width-1);idx[:,1].clamp_max_(height-1);hit=torch.tensor(mask,device='cuda')[idx[:,1],idx[:,0]]&inside;support+=hit*candidate
   vals=arr[mask];rows.append({'frame':f,'serial':serial,'mismatch_pixels':int(mask.sum()),'mean_arm_weight':float(vals[:,0].mean()) if len(vals) else None,'mean_spine_weight':float(vals[:,1].mean()) if len(vals) else None,'mean_dark_marker':float(vals[:,2].mean()) if len(vals) else None,'dark_mixed_pixel_fraction':float(((vals[:,0]>.1)&(vals[:,1]>.1)&(vals[:,2]>.5)).mean()) if len(vals) else None})
   sheet=Image.new('RGB',(768,408),'white')
   for k,(im,label) in enumerate([(pred,'generated RGB'),(np.uint8(arr*255),'R arm / G spine / B dark')]):sheet.paste(Image.fromarray(im).resize((384,384)),(k*384,24));ImageDraw.Draw(sheet).text((k*384+3,4),f'{serial} f{f} {label}',fill='black')
   sheet.save(out/f'{serial}_{f:04d}_generated.jpg',quality=94)
  print('skinning maps',f,flush=True)
 supported=candidate&(support>=2);change_count=0
 if o.correct_out:
  dest=ROOT/o.correct_out;dest.mkdir(exist_ok=False);new=weights.clone();sp=[0,3,6,9];new[supported]=new[supported].clone();new[supported[:,None].expand_as(new)&torch.tensor([i in sp for i in range(55)],device='cuda')[None]]=0
  new[supported]/=new[supported].sum(-1,keepdim=True).clamp_min(1e-8)
  # Smooth only the edited dark sleeve subset, using dark arm neighbors in canonical space.
  ids=torch.where(supported)[0].cpu().numpy();pool=torch.where((dark>.5)&(arms>.2))[0].cpu().numpy()
  if len(ids) and len(pool):
   tree=cKDTree(rig['zero_coords'][pool].cpu().numpy());dist,near=tree.query(rig['zero_coords'][ids].cpu().numpy(),k=min(8,len(pool)));near=np.atleast_2d(near);neighbor=new[torch.tensor(pool[near],device='cuda')].mean(1);neighbor[:,sp]=0;neighbor/=neighbor.sum(-1,keepdim=True).clamp_min(1e-8);new[torch.tensor(ids,device='cuda')]=.75*new[torch.tensor(ids,device='cuda')]+.25*neighbor
  new[supported]/=new[supported].sum(-1,keepdim=True);assert torch.isfinite(new).all() and torch.allclose(new.sum(-1),torch.ones(len(new),device='cuda'),atol=1e-5);change_count=int(supported.sum());assert torch.equal(new[~supported],weights[~supported]);torch.save({**b,'rig':{**b['rig'],'weights':new.cpu()}},dest/'avatar.pt')
 report={'success':True,'avatar':o.avatar,'frames':frames,'views':8,'dark_coarse_rgb_threshold':.35,'dark_mixed_gaussians':int(candidate.sum()),'source_mismatch_supported_gaussians':int(supported.sum()),'edited_gaussians':change_count,'records':rows,'seconds':time.time()-start,'scope':'Actual chain-weight and coarse-color-marker rasterization. Source bright/model dark underarm ROIs fixed per source pose. Two-view projected center support selects dark arm/spine mixed Gaussians; center support is a conservative attribution proxy, not exact per-splat visibility. Optional candidate removes spine weights only on supported dark sleeve points and smooths within dark arm neighbors; white/other points, features, decoder and motion unchanged. Generated maps public; sources remain private. Actual RGB rerender determines benefit.'};(out/'report.json').write_text(json.dumps(report,indent=2))
 if o.correct_out:(ROOT/o.correct_out/'report.json').write_text(json.dumps({'success':True,'source':o.avatar,'audit':o.out,'edited_gaussians':change_count,'scope':report['scope']},indent=2))
 print('skinning audit completed',change_count,flush=True)
