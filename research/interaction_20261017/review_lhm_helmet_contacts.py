"""Native-camera closeups with SAME LHM body, object and two compositors."""
import os
from pathlib import Path
import sys,json,importlib.util,time,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from PIL import Image,ImageDraw
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from mesh_renderer import MeshRenderer
from hybrid_compositor import composite,mesh_visible_opacities,composite_clipped_body
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
parser=argparse.ArgumentParser();parser.add_argument('--out',default='lhm_helmet_contact_review_v1');parser.add_argument('--avatar',default='avatar_hands_all300_v1');parser.add_argument('--motion',default='helmet_material_pose_all300_v1');parser.add_argument('--mesh',default='helmet_texture_native_all300_v4');parser.add_argument('--frames',type=int,nargs='+',default=[30,42,90,150,180,240,299]);options=parser.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.time()
def cuda(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
    return x
bundle=torch.load(ROOT/options.avatar/'avatar.pt',weights_only=True);rig=cuda(bundle['rig']);poses=cuda(bundle['poses']);features=bundle['features'].cuda().float();bg=bundle['background_embedding'][2].cuda().float()
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**bundle['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(bundle['decoder'],strict=True,assign=True);decoder.eval().cuda()
motion=cuda(torch.load(ROOT/options.motion/'sequence.pt',weights_only=True));mesh=cuda(torch.load(ROOT/options.mesh/'mesh.pt',weights_only=True));assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],motion['vertices_local'][motion['faces'].long()],atol=1e-6)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());renderer=MeshRenderer();size=504;records=[];samples=options.frames
with torch.inference_mode():
    for v in meta['views']:
        sheet=Image.new('RGB',(size*3,len(samples)*(size+24)),'white');W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);K0=np.array(v['K'])
        for row,t in enumerate(samples):
            world=mesh['vertices_local']@motion['R'][t].T*motion['logscale'].exp()+motion['translation_world'][t]
            pc=world.cpu().numpy()@np.array(v['W'])[:3,:3].T+np.array(v['W'])[:3,3];q=pc@K0.T;uv=q[:,:2]/q[:,2:];lo=uv.min(0);hi=uv.max(0);extent=max(240.,float(max(hi-lo)*2.2));origin=(lo+hi)/2-extent/2;factor=size/extent;A=np.array([[factor,0,-factor*origin[0]],[0,factor,-factor*origin[1]],[0,0,1.]])
            K=torch.tensor(A@K0,device='cuda',dtype=torch.float32);source=Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{t:04d}.png');photo=source.transform((size,size),Image.Transform.AFFINE,(1/factor,0,origin[0],0,1/factor,origin[1]),resample=Image.Resampling.BICUBIC);sheet.paste(photo,(0,row*(size+24)+24))
            mr=renderer(world,mesh['faces'],K,W,size,size,mesh['colors'],uv=mesh['uv'],texture=mesh['texture']);g=animate(rig,poses,t)
            clipped,blocked=mesh_visible_opacities(g['xyz'],g['opacity'].flatten(),K,W,mr)
            for col,method in enumerate(['expected-depth','gaussian-center'],1):
                opacities=g['opacity'].flatten() if method=='expected-depth' else clipped
                color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=opacities,colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=size,height=size,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
                with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,size,size)
                a=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001)
                result=composite(rgb[0,0].float()*a[...,None],a,color[0,...,-1],mr) if method=='expected-depth' else composite_clipped_body(rgb[0,0].float()*a[...,None],a,mr)
                im=Image.fromarray(np.uint8(result['rgb'].cpu().numpy().clip(0,1)*255));im.save(out/f"{v['serial']}_{t:04d}_{method}.png");sheet.paste(im,(col*size,row*(size+24)+24))
            for col,label in enumerate(['NATIVE SOURCE','SAME LHM / expected-depth','SAME LHM / center clipping']):ImageDraw.Draw(sheet).text((col*size+3,row*(size+24)+4),f"{v['serial']} f{t} | {label}",fill='black')
            records.append({'serial':v['serial'],'frame':t,'K_crop':K.cpu().tolist(),'W':v['W'],'native_crop_origin':origin.tolist(),'native_crop_extent':extent,'gaussians_center_clipped':int(blocked.sum())});print(v['serial'],t,'contact reviewed',flush=True)
        sheet.save(out/f"{v['serial']}_contacts.jpg",quality=94)
(out/'report.json').write_text(json.dumps({'success':True,'samples':samples,'records':records,'seconds':time.time()-start,'scope':'Both render panels use bit-identical original LHM neural appearance and identical fitted300-frame body/hands, current candidate helmet geometry/pose/nativeUV. Only compositor differs; actual native-camera source crops supplied. Rendered at504x504 cropped camera resolution, not upscaled old video. No mesh-distilled human; all8 training-view diagnostics, not independent quality or correct-contact acceptance. Center clipping is an approximation for extended Gaussians and neural decoder boundaries.'},indent=2));print('LHM contact review ready',flush=True)
