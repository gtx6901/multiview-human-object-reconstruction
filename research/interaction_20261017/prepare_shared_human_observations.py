"""All-time observation comparison for subsequent shared feature refinement.

Stores fixed initial support so a changing decoder mask cannot hide RGB errors.
These are candidate segmentation/render-derived observations, not ground truth.
"""
import os
from pathlib import Path
import sys,json,time,importlib.util,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from PIL import Image
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from shared_scene import load_shared_scene
from mesh_renderer import MeshRenderer
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"])
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scene',default='shared_scene_candidate_v1');p.add_argument('--frames',nargs='+',type=int);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 return x
scene=load_shared_scene(ROOT/o.scene);b=scene['avatar'];rig=move(b['rig']);poses=move(b['poses']);mesh=move(scene['mesh']);motion=move(scene['object_motion'])
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**b['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(b['decoder'],strict=True,assign=True);decoder.eval().cuda().requires_grad_(False)
features=b['features'].cuda().float();bg=b['background_embedding'][2].cuda().float();renderer=MeshRenderer()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());frames=poses['frames'].cpu().tolist();selected=frames if o.frames is None else o.frames
assert len(set(selected))==len(selected) and set(selected)<=set(frames)
width,height=504,378;A=np.array([[width/2048,0,(width/2048-1)/2],[0,height/1536,(height/1536-1)/2],[0,0,1.]])
records=[]
with torch.inference_mode():
 for v in meta['views']:
  serial=v['serial'];K=torch.tensor(A@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
  targets=[];supports=[];masks=[];viewrecords=[]
  for f in selected:
   t=frames.index(f);g=animate(rig,poses,t)
   color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,near_plane=.01,far_plane=100.,eps2d=.3)
   with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,height,width)
   prediction=rgb[0,0].float().cpu().numpy();mask=mask[0,0].float().squeeze().cpu().numpy();a=alpha[0,...,0].cpu().numpy()
   native=ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png';target=np.asarray(Image.open(native).convert('RGB'));assert target.shape==(1536,2048,3)
   target=cv2.resize(target,(width,height),interpolation=cv2.INTER_AREA)
   seg=ROOT/'sam2_all300_v1'/serial
   person=cv2.resize(np.asarray(Image.open(seg/'person'/f'{f:04d}.png')),(width,height),interpolation=cv2.INTER_NEAREST_EXACT)>0
   observed_object=cv2.resize(np.asarray(Image.open(seg/'helmet_proposals'/f'{f:04d}.png')),(width,height),interpolation=cv2.INTER_NEAREST_EXACT)>0
   world=(mesh['vertices_local']*motion['logscale'].exp())@motion['R'][t].T+motion['translation_world'][t]
   mr=renderer(world,mesh['faces'],K,W,width,height,mesh['colors'])
   rendered_object=mr['alpha'].cpu().numpy()>.01
   object_unknown=cv2.dilate((observed_object|rendered_object).astype('uint8'),np.ones((7,7),np.uint8))>0
   person_core=cv2.erode(person.astype('uint8'),np.ones((3,3),np.uint8))>0
   support=person_core&~object_unknown&(mask>.8)&(a>.8)
   n=int(support.sum());person_n=int(person_core.sum());mae=float(np.abs(prediction-target/255.)[support].mean()) if n else None
   row={'frame':f,'serial':serial,'person_core_pixels':person_n,'support_pixels':n,'coverage_fraction':n/max(person_n,1),'initial_rgb_mae_srgb':mae,'object_exclusion_pixels':int(object_unknown.sum()),'eligible':n>=128}
   viewrecords.append(row);records.append(row);targets.append(target);supports.append(support);masks.append(mask.astype('float16'))
  np.savez_compressed(out/f'{serial}.npz',frames=np.array(selected),target_rgb_u8=np.stack(targets),fixed_initial_support=np.stack(supports),initial_decoded_alpha=np.stack(masks),K=K.cpu().numpy(),W=W.cpu().numpy())
  (out/'partial.json').write_text(json.dumps({'completed_views':serial,'records':records},indent=2));print(json.dumps({'serial':serial,'frames':len(selected),'eligible':sum(r['eligible'] for r in viewrecords),'mean_coverage':float(np.mean([r['coverage_fraction'] for r in viewrecords]))}),flush=True)
assert len(records)==len(meta['views'])*len(selected)
report={'success':True,'scene':o.scene,'scene_manifest_sha256':hashlib.sha256((ROOT/o.scene/'scene.json').read_bytes()).hexdigest(),'resolution':[width,height],'frames':selected,'observations_compared':len(records),'eligible_observations':sum(r['eligible'] for r in records),'records':records,'seconds':time.time()-start,'scope':'All selected frames and all8 cameras compared with shared animated LHM scene. Native RGB area-downsampled with exact pixel-center K. Eroded candidate person mask, union of observed and rendered helmet dilated3pixels excluded, initial decoded/accumulated alpha>.8. Fixed initial support and decoder alpha are cached for subsequent fitting and support accounting. Per-observation128pixel eligibility is minimum support only; no claim of segmentation accuracy, cross-camera consistency, dynamic selection, all-data feature fitting or novel-view quality. Appearance, shape and motion are unchanged. No camera gain fitted here; sRGB residual includes exposure, geometry, pose and segmentation errors. Private RGB/camera cache must not be published.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('shared human observation comparison complete',flush=True)
