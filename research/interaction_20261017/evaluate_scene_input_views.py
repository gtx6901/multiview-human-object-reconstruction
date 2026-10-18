"""Same-camera source-foreground metrics for explicit shared-scene candidates.

These are input-view diagnostics with candidate masks, not novel-view truth.
Reference support is fixed from the input images, never each candidate alpha.
"""
import os
from pathlib import Path
import sys,json,argparse,time,importlib.util
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from shared_scene import load_shared_scene
from avatar_rig import animate
from mesh_renderer import MeshRenderer
from hybrid_compositor import mesh_visible_opacities,composite_clipped_body
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scenes',nargs='+',required=True);p.add_argument('--frames',nargs='+',type=int,default=[42,180,299]);p.add_argument('--all-frames',action='store_true');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2);start=time.time()
def cuda(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 return x
scenes=[load_shared_scene(ROOT/f) for f in o.scenes];reference=scenes[0]['avatar'];states=[]
for scene in scenes:
 b=scene['avatar'];assert b['decoder_config']==reference['decoder_config'] and all(torch.equal(b['decoder'][k],reference['decoder'][k]) for k in reference['decoder']);states.append({'rig':cuda(b['rig']),'poses':cuda(b['poses']),'features':b['features'].cuda().float(),'background':b['background_embedding'][2].cuda().float(),'mesh':cuda(scene['mesh']),'motion':cuda(scene['object_motion'])})
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**reference['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(reference['decoder'],strict=True,assign=True);decoder.eval().cuda();renderer=MeshRenderer();width,height=1008,756;s=width/2048;C=np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1.]])
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());masks={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};frames=list(range(300)) if o.all_frames else o.frames;saveframes={42,180,299};rows=[]
def iou(a,b):return float((a&b).sum()/max((a|b).sum(),1))
def mse(a,b,m):return float(((a-b)**2)[m].mean()) if m.any() else None
def psnr(v):return float(-10*np.log10(max(v,1e-12))) if v is not None else None
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  K=torch.tensor(C@v['K'],device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
  for f in frames:
   rgb=cv2.resize(np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB')),(width,height),interpolation=cv2.INTER_AREA).astype('float32')/255
   person=cv2.resize(np.array(Image.open(ROOT/'sam2_all300_v1'/v['serial']/'person'/f'{f:04d}.png')),(width,height),interpolation=cv2.INTER_NEAREST_EXACT)>0;rec=masks[(v['serial'],f)];known_object=rec['mask'] is not None;obj=np.zeros_like(person) if not known_object else cv2.resize(np.array(Image.open(ROOT/'sam2_all300_v1'/rec['mask'])),(width,height),interpolation=cv2.INTER_NEAREST_EXACT)>127
   foreground=person|obj;body_truth=person&~obj;interior=cv2.erode(foreground.astype('uint8'),np.ones((11,11),np.uint8))>0;truth=np.where(foreground[...,None],rgb,1.);pictures=[]
   for name,state in zip(o.scenes,states):
    t=state['poses']['frames'].tolist().index(f);g=animate(state['rig'],state['poses'],t);motion=state['motion'];mesh=state['mesh'];world=mesh['vertices_local']@motion['R'][t].T*motion['logscale'].exp()+motion['translation_world'][t];mr=renderer(world,mesh['faces'],K,W,width,height,mesh['colors'],uv=mesh.get('uv'),texture=mesh.get('texture'));opacity,_=mesh_visible_opacities(g['xyz'],g['opacity'].flatten(),K,W,mr)
    color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=opacity,colors=torch.cat([g['shs'].squeeze(1),state['features']],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),state['background']])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,height,width)
    ba=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001);result=composite_clipped_body(pred[0,0].float()*ba[...,None],ba,mr);image=result['rgb'].cpu().numpy().clip(0,1);assert np.isfinite(image).all();pred_fg=result['alpha'].cpu().numpy()>.5;pred_person=ba.cpu().numpy()>.5;pred_obj=(mr['alpha']*(1-ba)).cpu().numpy()>.5
    _,ssim_map=structural_similarity(truth,image,channel_axis=-1,data_range=1.,full=True);ssim_map=ssim_map.mean(-1) if ssim_map.ndim==3 else ssim_map;fg_mse=mse(image,rgb,foreground);helmet_mse=mse(image,rgb,obj) if known_object else None
    rows.append({'scene':name,'camera':v['serial'],'frame':f,'foreground_pixels':int(foreground.sum()),'ssim_interior_pixels':int(interior.sum()),'foreground_mse_srgb':fg_mse,'foreground_psnr_db':psnr(fg_mse),'foreground_ssim_interior':float(ssim_map[interior].mean()) if interior.any() else None,'foreground_iou':iou(pred_fg,foreground),'visible_person_iou':iou(pred_person,body_truth),'visible_helmet_iou':iou(pred_obj,obj) if known_object else None,'helmet_psnr_db':psnr(helmet_mse),'helmet_annotation_known':known_object})
    if f in saveframes:pictures.append(Image.fromarray(np.uint8(image*255)))
   if f in saveframes:
    generated=Image.new('RGB',(width*len(pictures),height+24),'white');private=Image.new('RGB',(width*(len(pictures)+1),height+24),'white');private.paste(Image.fromarray(np.uint8(truth*255)),(0,24));ImageDraw.Draw(private).text((3,4),'PRIVATE SOURCE foreground on white',fill='black')
    for ci,(pic,name) in enumerate(zip(pictures,o.scenes)):generated.paste(pic,(ci*width,24));private.paste(pic,((ci+1)*width,24));ImageDraw.Draw(generated).text((ci*width+3,4),name,fill='black');ImageDraw.Draw(private).text(((ci+1)*width+3,4),name,fill='black')
    generated.save(out/f'{v["serial"]}_{f:04d}_generated.jpg',quality=95);private.save(out/f'{v["serial"]}_{f:04d}_private.jpg',quality=95)
  print('evaluated input camera',v['serial'],flush=True)
summary=[]
for name in o.scenes:
 r=[x for x in rows if x['scene']==name];weight=sum(x['foreground_pixels'] for x in r);value=sum(x['foreground_mse_srgb']*x['foreground_pixels'] for x in r if x['foreground_mse_srgb'] is not None)/max(weight,1);summary.append({'scene':name,'weighted_foreground_psnr_db':psnr(value),'mean_foreground_ssim_interior':float(np.mean([x['foreground_ssim_interior'] for x in r if x['foreground_ssim_interior'] is not None])),'mean_foreground_iou':float(np.mean([x['foreground_iou'] for x in r])),'mean_visible_helmet_iou':float(np.mean([x['visible_helmet_iou'] for x in r if x['visible_helmet_iou'] is not None]))})
(out/'report.json').write_text(json.dumps({'success':True,'frames':frames,'resolution':[width,height],'records':rows,'summary':summary,'seconds':time.time()-start,'scope':'Same calibrated8 input cameras and explicit pixel-center1008x756 rendering, Gaussian-center mesh culling and saved neural decoder. Candidate SAM person/accepted helmet masks define FIXED source-foreground RGB support and eroded11pixel SSIM interior; missing rendered foreground counts as RGB error. White target background used only to avoid raw background contamination in SSIM windows. Mask IoUs include predicted leaks, separate visible body/helmet masks; unknown helmet annotations skipped. No per-candidate exposure fitting or color gains applied, so camera-response/light differences remain. Sampled or all-frame scope explicit. Input-view diagnostics only, not independent novel-view truth or contact certification; original-derived private sheets must not be published.'},indent=2));print(json.dumps(summary),flush=True)
