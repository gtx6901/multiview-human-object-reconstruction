"""Fit one shared LHM feature tensor to time-varying real observations."""
import os
from pathlib import Path
import sys,json,time,importlib.util,argparse,hashlib,shutil
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from shared_scene import load_shared_scene
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scene',default='shared_scene_candidate_v1');p.add_argument('--observations',default='shared_human_observations_all300_v1');p.add_argument('--epochs',type=int,default=1);p.add_argument('--frames',nargs='+',type=int);p.add_argument('--holdout',help='Optional camera excluded from feature gradients and gain calibration; evaluated with gain1')
p.add_argument('--response',choices=['srgb-model-fit','linear-reference'],default='srgb-model-fit');p.add_argument('--reference-camera',default='21305717');p.add_argument('--feature-bound',type=float,default=.2)
o=p.parse_args();assert 0<o.feature_bound<=.2;out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(712)
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 return x
scene=load_shared_scene(ROOT/o.scene);b=scene['avatar'];rig=move(b['rig']);poses=move(b['poses']);base=b['features'].cuda().float();bg=b['background_embedding'][2].cuda().float()
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**b['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(b['decoder'],strict=True,assign=True);decoder.eval().cuda().requires_grad_(False)
report=json.loads((ROOT/o.observations/'report.json').read_text());assert report['success']
assert report['scene_manifest_sha256']==hashlib.sha256((ROOT/o.scene/'scene.json').read_bytes()).hexdigest(),'Observation support belongs to a different initializer'
frames=poses['frames'].cpu().tolist();selected=frames if o.frames is None else o.frames;assert len(set(selected))==len(selected) and set(selected)<=set(frames)
width,height=report['resolution'];caches={}
for path in sorted((ROOT/o.observations).glob('*.npz')):
 with np.load(path) as d:
  ids=[d['frames'].tolist().index(f) for f in selected]
  caches[path.stem]={'rgb':d['target_rgb_u8'][ids],'support':d['fixed_initial_support'][ids],'mask':d['initial_decoded_alpha'][ids],'K':torch.tensor(d['K'],device='cuda'),'W':torch.tensor(d['W'],device='cuda')}
assert len(caches)==8 and (o.holdout is None or o.holdout in caches)
delta=torch.nn.Parameter(torch.zeros_like(base));opt=torch.optim.Adam([delta],lr=.003)
def response(x):
 if o.response=='srgb-model-fit':return x
 x=x.clamp(0,1);return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def render(c,f):
 with torch.no_grad():g=animate(rig,poses,frames.index(f))
 color,_,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),base+o.feature_bound*delta.tanh()],1),viewmats=c['W'][None],Ks=c['K'][None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,near_plane=.01,far_plane=100.,eps2d=.3)
 with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,height,width)
 return rgb[0,0].float(),mask[0,0].float().squeeze()
def tensors(c,i):
 raw=torch.tensor(c['rgb'][i],device='cuda').float()/255;support=torch.tensor(c['support'][i],device='cuda')
 if o.response=='linear-reference':support=support&~(raw>.985).all(-1)
 return response(raw),support,torch.tensor(c['mask'][i],device='cuda').float()
gains={};baseline=[]
with torch.no_grad():
 for serial,c in caches.items():
  numerator=torch.zeros(3,device='cuda');denominator=numerator.clone();predictions=[];ratios=[]
  for i,f in enumerate(selected):
   rgb,mask=render(c,f);rgb=response(rgb);target,support,anchor=tensors(c,i);numerator+=(target[support]*rgb[support]).sum(0);denominator+=rgb[support].square().sum(0)
   if o.response=='linear-reference':
    good=support&(rgb.amin(-1)>.01)&(rgb.amax(-1)<.9)&(target.amin(-1)>.01)&(target.amax(-1)<.9)
    if int(good.sum())>=128:ratios.append((target[good].log()-rgb[good].log()).median(0).values.cpu())
   predictions.append((rgb.cpu(),mask.cpu()))
  if o.response=='linear-reference':
   assert ratios or serial in [o.holdout,o.reference_camera],'No reliable initial response samples'
   loggain=torch.stack(ratios).median(0).values.cuda() if ratios else torch.zeros(3,device='cuda');scalar=loggain.mean();gain=torch.exp(scalar.clamp(-1.386294,1.386294)+(loggain-scalar).clamp(-.1,.1))
   if serial in [o.reference_camera,o.holdout]:gain=torch.ones(3,device='cuda')
  else:gain=torch.ones(3,device='cuda') if serial==o.holdout else (numerator/denominator.clamp_min(1e-6)).clamp(.5,2)
  gains[serial]=gain
  for i,(rgb,mask) in enumerate(predictions):
   target,support,_=tensors(c,i);target=target.cpu();support=support.cpu()
   error=(rgb*gain.cpu()-target).abs() if o.response=='srgb-model-fit' else (rgb-target/gain.cpu()).abs()
   baseline.append({'serial':serial,'frame':selected[i],'fixed_pixels':int(support.sum()),'rgb_mae':float(error[support].mean()) if support.any() else None})
  print('calibrated and compared',serial,flush=True)
history=[];train=[s for s in caches if s!=o.holdout];updates=0
for epoch in range(o.epochs):
 for i in np.random.default_rng(712+epoch).permutation(len(selected)):
  opt.zero_grad();total=0;count=0
  for serial in train:
   c=caches[serial];target,support,anchor=tensors(c,i)
   if int(support.sum())<128:continue
   rgb,mask=render(c,selected[i]);rgb=response(rgb);error=((rgb*gains[serial]-target).abs() if o.response=='srgb-model-fit' else (rgb-target/gains[serial]).abs()).mean(-1)
   loss=error[support].mean()+.2*(mask-anchor).square().mean();(loss/len(train)).backward();total+=float(loss.detach());count+=1
  if count:
   prior=.01*delta.tanh().square().mean();prior.backward();assert delta.grad is not None and torch.isfinite(delta.grad).all();opt.step();updates+=1
  if updates%25==0 or len(selected)<=3:
   row={'epoch':epoch,'frame':selected[i],'updates':updates,'online_loss':total/max(count,1)};history.append(row);print(json.dumps(row),flush=True)
 torch.save({'delta':delta.detach().cpu(),'optimizer':opt.state_dict(),'epoch_completed':epoch+1},out/'checkpoint.pt')
after=[]
with torch.no_grad():
 for serial,c in caches.items():
  for i,f in enumerate(selected):
   rgb,mask=render(c,f);rgb=response(rgb);target,support,anchor=tensors(c,i);error=(rgb*gains[serial]-target).abs() if o.response=='srgb-model-fit' else (rgb-target/gains[serial]).abs()
   after.append({'serial':serial,'frame':f,'fixed_pixels':int(support.sum()),'rgb_mae':float(error[support].mean()) if support.any() else None,'mask_anchor_mae':float((mask-anchor).abs().mean()),'remaining_support_fraction':float((mask[support]>.8).float().mean()) if support.any() else None})
  print('final comparison',serial,flush=True)
candidate=out/'scene';candidate.mkdir()
manifest=scene['manifest']
for row in manifest['assets']:shutil.copyfile(ROOT/o.scene/row['path'],candidate/row['path'])
human=torch.load(candidate/'human_canonical.pt',weights_only=True);human['features']=(base+o.feature_bound*delta.tanh()).detach().cpu();torch.save(human,candidate/'human_canonical.pt')
manifest['sources']={'initializer':o.scene,'refinement':o.out};manifest['scope']='Shared feature refinement candidate only; frozen rig, geometry, motion and neural decoder; not adopted quality result.'
for row in manifest['assets']:row['sha256']=hashlib.sha256((candidate/row['path']).read_bytes()).hexdigest()
(candidate/'scene.json').write_text(json.dumps(manifest,indent=2));loaded=load_shared_scene(candidate);assert torch.equal(loaded['avatar']['features'],human['features'])
r={'success':True,'frames':selected,'observations_compared':len(after),'training_cameras':train,'holdout_camera':o.holdout,'epochs':o.epochs,'updates':updates,'fixed_camera_rgb_gains':{s:g.cpu().tolist() for s,g in gains.items()},'baseline':baseline,'after':after,'history':history,'feature_max_change':float((human['features']-b['features']).abs().max()),'seconds':time.time()-start,'scope':'One canonical feature tensor receives gradients from all eligible selected times and training cameras. Fixed initial support prevents mask shrinking from removing RGB loss; initial decoder mask anchored. Fixed per-camera channel gains calibrated on initial model; heldout gain1. Bounded feature delta±.2, static geometry/poses/decoder, no geometry or beta refinement. Camera holdout is only conditional feature-overfit diagnostic because initializer and earlier motion used all8. Training error and support are not independent novel-view quality; must rerender before adoption. No dynamic cross-view reliability selection yet. Private tensors/images remain local.'}
r['response']=o.response;r['reference_camera']=o.reference_camera;r['feature_bound']=o.feature_bound;r['scope']=r['scope'].replace('delta±.2',f'delta±{o.feature_bound:g}')
if o.response=='linear-reference':r['scope']+=' NEW response: linear-light RGB; fixed reference camera gain1; other gains from temporal medians of same-view model-to-image log ratios, scalar exposure plus log-white-balance±.1; clipped-white target pixels excluded with static support. Loss compares corrected real RGB in reference units, so this metric is not comparable with old sRGB loss. This fixes brightness gauge but still inherits initial geometry/appearance bias; correspondence-graph calibration was rejected and is NOT used.'
(out/'report.json').write_text(json.dumps(r,indent=2));print('shared feature refinement candidate complete',flush=True)
