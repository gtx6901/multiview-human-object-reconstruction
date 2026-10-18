"""One integrated candidate, same helmet/appearance, real and novel cameras."""
import os
from pathlib import Path
import sys,json,time,importlib.util
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'lhmpp_20261016'))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from mesh_renderer import MeshRenderer
from hybrid_compositor import mesh_visible_opacities,composite_clipped_body
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
out=ROOT/'grounded_legs_integrated_review_v1';out.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2);start=time.time()
def cuda(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 return x
before=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True);after=torch.load(ROOT/'grounded_legs_all300_v1/avatar.pt',weights_only=True)
for key in ['features','background_embedding']:assert torch.equal(before[key],after[key])
for key in before['rig']:assert torch.equal(before['rig'][key],after['rig'][key])
rig=cuda(before['rig']);poses=[cuda(before['poses']),cuda(after['poses'])];features=before['features'].cuda().float();bg=before['background_embedding'][2].cuda().float()
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**before['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(before['decoder'],strict=True,assign=True);decoder.eval().cuda();renderer=MeshRenderer();motion=cuda(torch.load(ROOT/'helmet_lhm_depth_tight_all300_v1/sequence.pt',weights_only=True));mesh=cuda(torch.load(ROOT/'helmet_texture_consistent_lhm_v1/mesh.pt',weights_only=True));assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],motion['vertices_local'][motion['faces'].long()],atol=1e-6)
cams=json.loads((ROOT/'lhm_portrait_framing_v1/cameras.json').read_text());width,height=567,756;C=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
selected=[0,30,42,60,90,120,135,145,155,165,175,180,190,200,210,225,240,260,280,299];continuous=list(range(45,96));novel=cams['views'][8:] if len(cams['views'])==10 else [cams['views'][16],cams['views'][40]];original=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());native={v['serial']:v for v in original['views']};rows=[]
def render(g,f,c):
 K=torch.tensor(C@np.array(c['K']),device='cuda',dtype=torch.float32);W=torch.tensor(c['W'],device='cuda',dtype=torch.float32);world=mesh['vertices_local']@motion['R'][f].T*motion['logscale'].exp()+motion['translation_world'][f];mr=renderer(world,mesh['faces'],K,W,width,height,mesh['colors'],uv=mesh.get('uv'),texture=mesh.get('texture'));opacity,_=mesh_visible_opacities(g['xyz'],g['opacity'].flatten(),K,W,mr)
 color,a,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=opacity,colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
 with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,height,width)
 ba=mask[0,0].float().squeeze()*(a[0,...,0]>.001);result=composite_clipped_body(rgb[0,0].float()*ba[...,None],ba,mr);arr=np.uint8(result['rgb'].cpu().numpy().clip(0,1)*255);assert np.isfinite(arr).all();return Image.fromarray(arr)
with torch.inference_mode():
 writers=[imageio.get_writer(str(out/f'novel_{i}_before_after_45_95.mp4'),format='FFMPEG',fps=30,codec='libx264',macro_block_size=2) for i in range(2)]
 try:
  for f in sorted(set(selected+continuous)):
   gs=[animate(rig,p,f) for p in poses]
   if f in continuous:
    for vi,c in enumerate(novel):
     pair=Image.new('RGB',(width*2,height+28),'white');draw=ImageDraw.Draw(pair)
     for k,g in enumerate(gs):pair.paste(render(g,f,c),(k*width,28));draw.text((k*width+5,5),f'{"BEFORE" if k==0 else "GROUNDED LEGS"} f{f}',fill='black')
     writers[vi].append_data(np.asarray(pair))
     if f in [45,60,75,95]:pair.save(out/f'novel{vi}_{f:04d}_generated.jpg',quality=95)
   if f in selected:
    private=Image.new('RGB',(width*3,8*(height+24)),'white');generated=Image.new('RGB',(width*2,8*(height+24)),'white')
    for vi,c in enumerate(cams['views'][:8]):
     serial=c['label'];v=native[serial];H=C@np.array(c['K'])@np.linalg.inv(np.array(v['K']));assert np.allclose(H[2],[0,0,1]);src=np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB'));src=cv2.warpAffine(src,H[:2],(width,height),borderValue=(255,255,255));private.paste(Image.fromarray(src),(0,vi*(height+24)+24))
     for k,g in enumerate(gs):
      pic=render(g,f,c);generated.paste(pic,(k*width,vi*(height+24)+24));private.paste(pic,((k+1)*width,vi*(height+24)+24))
     ImageDraw.Draw(private).text((3,vi*(height+24)+4),f'{serial} f{f} | PRIVATE SOURCE / BEFORE / AFTER',fill='black');ImageDraw.Draw(generated).text((3,vi*(height+24)+4),f'{serial} f{f} | BEFORE / AFTER',fill='black')
    private.resize((width*3//2,8*(height+24)//2)).save(out/f'all8_{f:04d}_private.jpg',quality=93);generated.resize((width,8*(height+24)//2)).save(out/f'all8_{f:04d}_generated.jpg',quality=93)
   print('integrated leg review frame',f,flush=True)
 finally:
  for writer in writers:writer.close()
for i in range(2):
 reader=imageio.get_reader(str(out/f'novel_{i}_before_after_45_95.mp4'));count=sum(1 for _ in reader);reader.close();assert count==len(continuous);rows.append({'novel_camera':i,'decoded_frames':count})
(out/'report.json').write_text(json.dumps({'success':True,'selected_real_camera_frames':selected,'real_cameras':8,'continuous_novel_frames':continuous,'videos':rows,'resolution_per_panel':[width,height],'seconds':time.time()-start,'scope':'Actual saved original accepted LHM canonical rig/features/decoder/upper body and SAME helmet mesh/texture/motion; only lower-body pose changes. 20phase-spanning frames at all8 true cameras, original photo sheets PRIVATE; generated sheets/videos contain only model renders. Two fixed novel cameras51continuous frames45..95 before/after. Actual image review required; selected sequence is not a full300-frame video.'},indent=2));print('integrated leg review complete',flush=True)
