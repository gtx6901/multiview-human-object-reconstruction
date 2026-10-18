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
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scene',default='shared_scene_candidate_v1');p.add_argument('--frames',nargs='+',type=int);p.add_argument('--portrait-crops',action='store_true');p.add_argument('--crop-padding',type=float,default=.05);p.add_argument('--alignment-screen',action='store_true',help='Use only well-aligned anatomical regions, exclude clipping and wider silhouette boundary');o=p.parse_args()
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
width,height=504,(756 if o.portrait_crops else 378);A=np.array([[width/2048,0,(width/2048-1)/2],[0,height/1536,(height/1536-1)/2],[0,0,1.]])
records=[];crop_records=[]
wholebody=np.load(ROOT/'observations_all300_v1/wholebody.npz') if o.alignment_screen else None
surface_world=None
if o.alignment_screen:
 from smplx_surface_landmarks import SMPLXSurfaceLandmarks
 from avatar_rig import pose_axis_angles
 surface=SMPLXSurfaceLandmarks().cuda();parts=[]
 with torch.no_grad():
  for begin in range(0,len(frames),16):
   end=min(begin+16,len(frames));aa=torch.cat([pose_axis_angles(poses,t,'cuda') for t in range(begin,end)])
   parts.append(surface(aa,poses['scale'],poses['translation_world'][begin:end],poses['betas'])['coco17'].cpu().numpy())
 surface_world=np.concatenate(parts);del surface,parts
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  serial=v['serial'];crop_box=None;C=A
  if o.portrait_crops:
   bounds=[]
   for f in selected:
    mask=Image.open(ROOT/'sam2_all300_v1'/serial/'person'/f'{f:04d}.png');box=mask.getbbox()
    if box is not None:bounds.append(box)
   assert bounds,'No person mask support for crop'
   bounds=np.array(bounds);lo=bounds[:,:2].min(0);hi=bounds[:,2:].max(0);ratio=float(np.maximum((hi-lo)/[width,height],1e-5).max())*(1+2*o.crop_padding);size=np.ceil(np.array([width,height])*ratio).astype(int);origin=np.floor((lo+hi-size)/2).astype(int);crop_box=(*origin,*(origin+size));sx,sy=np.array([width,height])/size;C=np.array([[sx,0,sx*(.5-origin[0])-.5],[0,sy,sy*(.5-origin[1])-.5],[0,0,1.]])
   crop_records.append({'serial':serial,'native_crop_xyxy':list(map(int,crop_box)),'scope':'Fixed per-camera envelope of all selected person masks, padded and aspect fitted. Does not follow the current fitted model or discard motion extrema.'})
  K=torch.tensor(C@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
  targets=[];supports=[];masks=[];viewrecords=[]
  for f in selected:
   t=frames.index(f);g=animate(rig,poses,t)
   color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,near_plane=.01,far_plane=100.,eps2d=.3)
   with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,height,width)
   prediction=rgb[0,0].float().cpu().numpy();mask=mask[0,0].float().squeeze().cpu().numpy();a=alpha[0,...,0].cpu().numpy()
   native=ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png';target=np.asarray(Image.open(native).convert('RGB'));assert target.shape==(1536,2048,3)
   if crop_box is None:target=cv2.resize(target,(width,height),interpolation=cv2.INTER_AREA)
   else:target=np.asarray(Image.fromarray(target).crop(crop_box).resize((width,height),Image.Resampling.BOX))
   seg=ROOT/'sam2_all300_v1'/serial
   def resize_mask(path):
    im=Image.open(path)
    return (cv2.resize(np.asarray(im),(width,height),interpolation=cv2.INTER_NEAREST_EXACT) if crop_box is None else np.asarray(im.crop(crop_box).resize((width,height),Image.Resampling.NEAREST)))>0
   person=resize_mask(seg/'person'/f'{f:04d}.png')
   observed_object=resize_mask(seg/'helmet_proposals'/f'{f:04d}.png')
   world=(mesh['vertices_local']*motion['logscale'].exp())@motion['R'][t].T+motion['translation_world'][t]
   mr=renderer(world,mesh['faces'],K,W,width,height,mesh['colors'])
   rendered_object=mr['alpha'].cpu().numpy()>.01
   object_unknown=cv2.dilate((observed_object|rendered_object).astype('uint8'),np.ones((7,7),np.uint8))>0
   person_core=cv2.erode(person.astype('uint8'),np.ones((3,3),np.uint8))>0
   support=person_core&~object_unknown&(mask>.8)&(a>.8)
   screen_record={}
   if o.alignment_screen:
    initial_count=int(support.sum());obs=wholebody['xy'][vi,f,:17];conf=wholebody['score'][vi,f,:17];in_image=wholebody['in_image'][vi,f,:17]
    world_j=surface_world[t];nativeW=np.asarray(v['W']);nativeK=np.asarray(v['K']);pc=world_j@nativeW[:3,:3].T+nativeW[:3,3];q=pc@nativeK.T;pred=q[:,:2]/q[:,2:]
    torso=float(np.linalg.norm((obs[5]+obs[6]-obs[11]-obs[12])/2));threshold=max(4.,.04*torso)
    residual=np.linalg.norm(obs-pred,axis=-1);good=in_image&(conf>2)&np.isfinite(residual)&(residual<threshold)&(pc[:,2]>0)
    hobs=np.column_stack([obs,np.ones(17)])@C.T;points=hobs[:,:2]/hobs[:,2:];region=np.zeros((height,width),np.uint8)
    if good[[5,6,11,12]].all():cv2.fillConvexPoly(region,np.round(points[[5,6,12,11]]).astype('int32'),1)
    radius=max(2,int(torso*abs(C[0,0])*.12))
    for ja,jb in [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]:
     if good[ja] and good[jb]:cv2.line(region,tuple(np.round(points[ja]).astype(int)),tuple(np.round(points[jb]).astype(int)),1,2*radius)
    boundary_core=cv2.erode(person.astype('uint8'),np.ones((9,9),np.uint8))>0
    unclipped=(target.max(-1)<250)&(target.max(-1)>3)
    support&=(region>0)&boundary_core&unclipped
    screen_record={'pixels_before_alignment_screen':initial_count,'reliable_body_joints':int(good[5:].sum()),'joint_residual_threshold_native_px':threshold,'excluded_by_alignment_boundary_clipping':initial_count-int(support.sum())}
   n=int(support.sum());person_n=int(person_core.sum());mae=float(np.abs(prediction-target/255.)[support].mean()) if n else None
   row={'frame':f,'serial':serial,'person_core_pixels':person_n,'support_pixels':n,'coverage_fraction':n/max(person_n,1),'initial_rgb_mae_srgb':mae,'object_exclusion_pixels':int(object_unknown.sum()),'eligible':n>=128,**screen_record}
   viewrecords.append(row);records.append(row);targets.append(target);supports.append(support);masks.append(mask.astype('float16'))
  np.savez_compressed(out/f'{serial}.npz',frames=np.array(selected),target_rgb_u8=np.stack(targets),fixed_initial_support=np.stack(supports),initial_decoded_alpha=np.stack(masks),K=K.cpu().numpy(),W=W.cpu().numpy())
  (out/'partial.json').write_text(json.dumps({'completed_views':serial,'records':records},indent=2));print(json.dumps({'serial':serial,'frames':len(selected),'eligible':sum(r['eligible'] for r in viewrecords),'mean_coverage':float(np.mean([r['coverage_fraction'] for r in viewrecords]))}),flush=True)
assert len(records)==len(meta['views'])*len(selected)
report={'success':True,'scene':o.scene,'scene_manifest_sha256':hashlib.sha256((ROOT/o.scene/'scene.json').read_bytes()).hexdigest(),'resolution':[width,height],'frames':selected,'observations_compared':len(records),'eligible_observations':sum(r['eligible'] for r in records),'records':records,'seconds':time.time()-start,'scope':'All selected frames and all8 cameras compared with shared animated LHM scene. Native RGB area-downsampled with exact pixel-center K. Eroded candidate person mask, union of observed and rendered helmet dilated3pixels excluded, initial decoded/accumulated alpha>.8. Fixed initial support and decoder alpha are cached for subsequent fitting and support accounting. Per-observation128pixel eligibility is minimum support only; no claim of segmentation accuracy, cross-camera consistency, dynamic selection, all-data feature fitting or novel-view quality. Appearance, shape and motion are unchanged. No camera gain fitted here; sRGB residual includes exposure, geometry, pose and segmentation errors. Private RGB/camera cache must not be published.'}
report['portrait_crops']=o.portrait_crops;report['camera_crop_records']=crop_records
report['alignment_screen']=o.alignment_screen
if o.alignment_screen:report['scope']+=' Additional local selection: native COCO17 surface-joint residual below4% observed torso length (4px floor), RTMW score>2/in-image/positive depth; torso polygon and limb capsules only when corresponding observed joints agree.9pixel eroded silhouette and any-channel>=250 clipping excluded. Head/hands are not feature-training targets in this conservative body pass. Selection uses geometry/detector evidence, not low photometric error; it is not a cross-view surface-correspondence verification.'
if o.portrait_crops:report['scope']+=' High-resolution portrait-crop comparison:504x756, fixed per-camera all-time person-mask envelope plus explicit padding/aspect fit, exact pixel-center K. Source RGB uses area box resampling, masks nearest; source beyond native canvas is masked-out padding. No model-derived crop or per-frame crop motion.'
(out/'report.json').write_text(json.dumps(report,indent=2));print('shared human observation comparison complete',flush=True)
