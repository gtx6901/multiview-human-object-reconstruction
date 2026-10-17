"""Independent replay of saved animated avatar and all-view object poses."""
import os
from pathlib import Path
import sys,json,time,importlib.util,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016'
sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from PIL import Image,ImageDraw,ImageOps
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from mesh_renderer import MeshRenderer
from hybrid_compositor import composite,mesh_visible_opacities,composite_clipped_body
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"])
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('reference',nargs='?',default='NONE');p.add_argument('motion',nargs='?',default='helmet_pose180_211_v1');p.add_argument('mesh',nargs='?',default='helmet_colors180_211_v1');p.add_argument('--avatar',default='avatar_motion180_211_v1');p.add_argument('--data',default='helmet_contiguous180_211_v1');p.add_argument('--cameras',default='hybrid_frame30_v1/cameras.json');p.add_argument('--neutral-helmet',action='store_true')
p.add_argument('--occlusion',choices=['expected-depth','gaussian-center'],default='expected-depth')
p.add_argument('--frames',type=int,nargs='+',help='Optional selected original frame IDs for an explicit preview; default renders full saved sequence.')
p.add_argument('--scene',help='Shared-scene directory: canonical human/helmet assets and separate per-frame states')
options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
reference=ROOT/options.reference
def move(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    return x
scene_data=None
if options.scene:
    from shared_scene import load_shared_scene
    scene_data=load_shared_scene(ROOT/options.scene)
bundle=scene_data['avatar'] if scene_data else torch.load(ROOT/options.avatar/'avatar.pt',weights_only=True)
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**bundle['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(bundle['decoder'],strict=True,assign=True);decoder.eval().cuda()
rig=move(bundle['rig']);poses=move(bundle['poses']);features=bundle['features'].cuda().float();bg=bundle['background_embedding'][2].cuda().float()
motion_source=ROOT/options.motion;mesh_source=ROOT/options.mesh
if scene_data:
    motion=move(scene_data['object_motion']);mesh=move(scene_data['mesh']);motion_source=ROOT/options.scene/'motion.pt';mesh_source=ROOT/options.scene/'helmet_canonical.pt'
else:motion=move(torch.load(motion_source/'sequence.pt',weights_only=True));mesh=move(torch.load(mesh_source/'mesh.pt',weights_only=True))
assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],motion['vertices_local'][motion['faces'].long()],atol=1e-6),'Renderer geometry differs from fitted geometry'
if options.neutral_helmet:
    import trimesh
    tm=trimesh.Trimesh(mesh['vertices_local'].cpu().numpy(),mesh['faces'].cpu().numpy(),process=False)
    normal_sign=1 if tm.volume>=0 else -1
    local_normals=torch.tensor(np.array(tm.vertex_normals)*normal_sign,device='cuda',dtype=torch.float32)
assert torch.equal(poses['frames'],motion['frames'])
cameras=scene_data['cameras'] if scene_data else json.loads((ROOT/options.cameras).read_text());width=cameras['width'];height=cameras['height']
novel=cameras['views'][8:] if len(cameras['views'])==10 else [cameras['views'][16],cameras['views'][40]]
(out/'cameras.json').write_text(json.dumps({'width':width,'height':height,'views':cameras['views'][:8]+novel},indent=2))
renderer=MeshRenderer();records=[];preview=[];checks=[]
render_indices=list(range(len(poses['frames']))) if options.frames is None else [poses['frames'].tolist().index(f) for f in options.frames]
assert render_indices and len(set(render_indices))==len(render_indices)
qa_indices=sorted(set([render_indices[0],render_indices[len(render_indices)//2],render_indices[-1]]))
with torch.inference_mode():
 for t in render_indices:
    frame=int(poses['frames'][t])
    g=animate(rig,poses,t);world=mesh['vertices_local'].float()@motion['R'][t].float().T*motion['logscale'].float().exp()+motion['translation_world'][t].float()
    views=novel+(cameras['views'][:8] if t in qa_indices else [])
    pics=[]
    for c in views:
        K=torch.tensor(c['K'],device='cuda',dtype=torch.float32);W=torch.tensor(c['W'],device='cuda',dtype=torch.float32)
        colors=mesh['colors'].float()
        if options.neutral_helmet:
            normals=local_normals@motion['R'][t].float().T;center=-W[:3,:3].T@W[:3,3];light=torch.nn.functional.normalize(center-world,dim=1);diffuse=(normals*light).sum(1).clamp_min(0);colors=(.65*(.4+.6*diffuse))[:,None].expand(-1,3)
        mr=renderer(world,mesh['faces'],K,W,width,height,colors,
            uv=mesh.get('uv') if not options.neutral_helmet else None,
            texture=mesh.get('texture') if not options.neutral_helmet else None)
        opacities=g['opacity'].flatten()
        if options.occlusion=='gaussian-center':
            opacities,blocked=mesh_visible_opacities(g['xyz'],opacities,K,W,mr)
        color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=opacities,colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
        with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,height,width)
        ba=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001)
        if options.occlusion=='gaussian-center':result=composite_clipped_body(rgb[0,0].float()*ba[...,None],ba,mr)
        else:result=composite(rgb[0,0].float()*ba[...,None],ba,color[0,...,-1],mr)
        assert torch.isfinite(result['rgb']).all()
        arr=np.uint8(result['rgb'].cpu().numpy().clip(0,1)*255);pic=Image.fromarray(arr)
        filename=f'{frame:04d}_{c["label"]}.png';pic.save(out/filename)
        if reference and reference.name!='NONE':
            delta=np.abs(arr.astype(int)-np.asarray(Image.open(reference/filename)).astype(int));assert delta.max()==0
            checks.append({'file':filename,'max_u8_difference':int(delta.max())})
        if c in novel:pics.append(pic)
        records.append({'frame':frame,'view':c['label'],'mesh_visible_pixels':int(((mr['alpha']>.5)&~result['body_first']).sum())})
    strip=Image.new('RGB',(width*3,height+40),'white')
    source=Image.open(ROOT/options.data/'21305717'/f'{frame:04d}.png')
    strip.paste(ImageOps.pad(source,(width,height),color='white'),(0,40))
    for j,pic in enumerate(pics):strip.paste(pic,((j+1)*width,40))
    material='neutral helmet diagnostic' if options.neutral_helmet else 'candidate geometry/color'
    d=ImageDraw.Draw(strip);d.text((8,10),f'SOURCE camera 717 | frame {frame}',fill='black');d.text((width+8,10),'NOVEL A | '+material,fill='black');d.text((2*width+8,10),'NOVEL B | '+material,fill='black')
    preview.append(np.asarray(strip));print(f'frame {frame} rendered',flush=True)
 for name,fps in [('source_vs_novel_30fps.mp4',30),('source_vs_novel_slow10fps.mp4',10)]:
    with imageio.get_writer(str(out/name),format='FFMPEG',fps=fps,codec='libx264',macro_block_size=2) as writer:
        for arr in preview:writer.append_data(arr)
 for name,fps in [('novel_only_30fps.mp4',30),('novel_only_slow10fps.mp4',10)]:
    with imageio.get_writer(str(out/name),format='FFMPEG',fps=fps,codec='libx264',macro_block_size=2) as writer:
        for arr in preview:writer.append_data(arr[:,width:].copy())
sheet=Image.new('RGB',(width*3,(height+40)*3),'white')
for i,t in enumerate(qa_indices):sheet.paste(Image.fromarray(preview[render_indices.index(t)]),(0,i*(height+40)))
sheet.save(out/'motion_contact.jpg')
video_checks=[]
for name in ['source_vs_novel_30fps.mp4','source_vs_novel_slow10fps.mp4','novel_only_30fps.mp4','novel_only_slow10fps.mp4']:
    reader=imageio.get_reader(str(out/name),format='FFMPEG');count=sum(1 for _ in reader);reader.close();assert count==len(render_indices);video_checks.append({'file':name,'decoded_frames':count})
report={'success':True,'helmet_material':'neutral gray plus synthetic headlight; geometry diagnostic only' if options.neutral_helmet else 'observed vertex colors plus gray for unobserved','object_motion_source':str(motion_source),'object_mesh_source':str(mesh_source),'frames':poses['frames'].tolist(),'seconds':time.time()-start,'real_cameras_used_in_body_fitting':8,'novel_fixed_cameras':len(novel),'records':records,'replay_checks':checks,'videos':video_checks,'scope':'Continuous sampled original-time frames, shared all8-fitted body and candidate object poses; object observed-view coverage detailed in its source report. Fixed original Gaussian appearance and zero finger articulation. Approximate Gaussian expected-depth compositing. Candidate geometry remains unaccepted; see object source report for method and limits. No CoCoNet predictions used or full-action completion claimed.'}
report['occlusion_method']=options.occlusion
if options.scene:report['shared_scene_source']=options.scene
report['frames']=[int(poses['frames'][t]) for t in render_indices]
if options.frames is not None:report['scope']='Selected-frame preview only; not a continuous full-action video. '+report['scope']
if 'texture' in mesh and not options.neutral_helmet:
    report['helmet_material']='Baked native-frame multiview UV RGB appearance; see source texture report for measured/inferred coverage and limitations'
report['fitted_hand_fields_present']=all(k in poses for k in ['left_hand_pose','right_hand_pose'])
if report['fitted_hand_fields_present']:
    report['scope']=report['scope'].replace('zero finger articulation','saved fitted finger articulation')
if options.occlusion=='gaussian-center':
    report['scope']=report['scope'].replace('Approximate Gaussian expected-depth compositing.','Gaussian-center depth clipping before feature rasterization and neural decoding.')
    report['scope']+=' Gaussian-center clipping precedes feature rasterization/decoding; extended splats and decoder boundaries remain approximate. Not evidence of correct physical contact.'
(out/'report.json').write_text(json.dumps(report,indent=2));print('complete',flush=True)
