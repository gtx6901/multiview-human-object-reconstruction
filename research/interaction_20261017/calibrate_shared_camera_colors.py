"""Robust exposure graph from real colors at matched visible Gaussian centers.

Fixed reference camera defines canonical brightness. Approximate correspondences
still carry pose/geometry/specular bias; log ratios are not albedo truth.
"""
import os
from pathlib import Path
import sys,json,time,argparse,importlib.util
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
import torch.nn.functional as F
from shared_scene import load_shared_scene
from avatar_rig import animate
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scene',default='shared_scene_candidate_v1');p.add_argument('--observations',default='shared_human_observations_all300_v1');p.add_argument('--reference',default='21305717');p.add_argument('--stride',type=int,default=5);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 return x
b=load_shared_scene(ROOT/o.scene)['avatar'];rig=move(b['rig']);poses=move(b['poses']);frames=poses['frames'].tolist();selected=frames[::o.stride];caches={}
for path in sorted((ROOT/o.observations).glob('*.npz')):
 with np.load(path) as d:
  ids=[d['frames'].tolist().index(f) for f in selected];caches[path.stem]={'rgb':d['target_rgb_u8'][ids],'support':d['fixed_initial_support'][ids],'K':torch.tensor(d['K'],device='cuda'),'W':torch.tensor(d['W'],device='cuda')}
 print('loaded correspondence cache',path.stem,flush=True)
serials=list(caches);ref=serials.index(o.reference);assert len(serials)==8
ids=torch.where((rig['opacity'].flatten()>.5)&(rig['scaling'].amax(-1)<.03))[0];assert len(ids)>=512
gen=torch.Generator(device='cuda').manual_seed(42);ids=ids[torch.randperm(len(ids),generator=gen,device='cuda')[:4096]]
edges=[];width,height=504,378
def linear(x):return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
with torch.inference_mode():
 for fi,f in enumerate(selected):
  g=animate(rig,poses,frames.index(f));colors=[];good=[]
  for serial,c in caches.items():
   rgba,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.ones_like(g['xyz']),viewmats=c['W'][None],Ks=c['K'][None],width=width,height=height,packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
   pc=g['xyz'][ids]@c['W'][:3,:3].T+c['W'][:3,3];q=pc@c['K'].T;uv=q[:,:2]/q[:,2:].clamp_min(.01);grid=(2*(uv+.5)/torch.tensor([width,height],device='cuda')-1)[None,:,None]
   def sample(v,mode='bilinear'):return F.grid_sample(v,grid,mode=mode,align_corners=False,padding_mode='zeros')[0,:,:,0].T
   rgb=sample(torch.tensor(c['rgb'][fi],device='cuda').permute(2,0,1)[None].float()/255)
   dep=sample(rgba[0,...,-1][None,None])[:,0];a=sample(alpha.permute(0,3,1,2))[:,0];support=sample(torch.tensor(c['support'][fi],device='cuda')[None,None].float(),'nearest')[:,0]>.5
   valid=(pc[:,2]>.01)&(uv[:,0]>1)&(uv[:,0]<width-2)&(uv[:,1]>1)&(uv[:,1]<height-2)&((pc[:,2]-dep).abs()<.005)&(a>.95)&support&(rgb.amin(-1)>.05)&(rgb.amax(-1)<.95)
   colors.append(linear(rgb).clamp_min(1e-5).log().cpu().numpy());good.append(valid.cpu().numpy())
  for i in range(8):
   for j in range(i):
    valid=good[i]&good[j];n=int(valid.sum())
    if n<16:continue
    diff=colors[i][valid]-colors[j][valid];median=np.median(diff,axis=0);mad=np.median(np.abs(diff-median),axis=0)
    edges.append({'frame':f,'i':i,'j':j,'n':n,'log_ratio':median.tolist(),'mad':mad.tolist()})
  if fi%10==0:print('matched real colors',f,'edges',len(edges),flush=True)
assert len(edges)>=28,'Insufficient correspondence graph'
unknown=[i for i in range(8) if i!=ref];X=np.zeros((len(edges),7));Y=np.array([r['log_ratio'] for r in edges]);baseweight=np.array([min(r['n'],128)/(1+np.mean(r['mad'])) for r in edges])
for k,e in enumerate(edges):
 for i,sign in [(e['i'],1),(e['j'],-1)]:
  if i!=ref:X[k,unknown.index(i)]=sign
assert np.linalg.matrix_rank(X)==7,'Reference exposure graph disconnected'
solution=[]
for ch in range(3):
 weights=baseweight.copy();a=np.zeros(7)
 for step in range(30):
  a=np.linalg.lstsq(X*np.sqrt(weights[:,None]),Y[:,ch]*np.sqrt(weights),rcond=None)[0];res=X@a-Y[:,ch];weights=baseweight*np.minimum(1,.15/np.maximum(np.abs(res),1e-6))
 solution.append(a)
loggain=np.zeros((8,3));loggain[unknown]=np.stack(solution,axis=1);gain=np.exp(loggain);assert np.isfinite(gain).all()
residual=X@np.stack(solution,axis=1)-Y
quality={'usable_for_refinement':bool(np.median(np.abs(residual))<.15 and gain.min()>=.125 and gain.max()<=8),'median_absolute_log_residual_limit':.15,'reason':'Accept only a reference-connected, mutually consistent response graph; otherwise gains are diagnostic and MUST NOT enter feature fitting.'}
report={'success':True,'reference_camera':o.reference,'gain_direction':'linear measured_rgb = reference_canonical_rgb * camera_gain','fixed_camera_linear_rgb_gains':dict(zip(serials,gain.tolist())),'frames_compared':selected,'correspondence_edges':len(edges),'median_absolute_log_ratio_residual':np.median(np.abs(residual),axis=0).tolist(),'edges':edges,'seconds':time.time()-start,'scope':'Reference camera exactly1; real-image colors at4096 deterministic Gaussian-center candidates over sampled times, fixed support excludes candidate object. Accumulated alpha>.95 and center-to-expected-depth distance<.005 normalized units approximate visibility. Unsaturated sRGB converted to linear, per-frame pair median log ratios, connected graph robust IRLS. Geometry/pose bias and view-dependent lighting remain. No neural rendered colors used to fit gains, no photometric image-quality acceptance; no dynamic selection or heldout-camera validation yet.'}
report['quality_gate']=quality
(out/'report.json').write_text(json.dumps(report,indent=2));print('reference camera exposure graph complete',quality,flush=True)
