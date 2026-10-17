"""Frozen neural Gaussian avatar in A-pose, dozens of RGB/alpha target views."""
import os
from pathlib import Path
import sys,json,time,importlib.util,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from PIL import Image,ImageDraw
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',default='avatar_canonical_views_v2');p.add_argument('--avatar',default='avatar_hands_all300_v1');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
bundle=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True)
def move(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    return x
rig=move(bundle['rig']);features=bundle['features'].cuda().float();bg=bundle['background_embedding'][2].cuda().float()
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**bundle['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(bundle['decoder'],strict=True,assign=True);decoder.eval().cuda()
poses={'root_pose':torch.zeros(1,3,device='cuda'),'body_pose':torch.zeros(1,21,3,device='cuda'),'translation_world':torch.zeros(1,3,device='cuda'),'scale':bundle['poses']['scale'].cuda(),'betas':bundle['poses']['betas'].cuda()}
poses['body_pose'][0,15,2]=-.65;poses['body_pose'][0,16,2]=.65
width=height=504;K=np.array([[800.,0,251.5],[0,800.,251.5],[0,0,1.]])
with torch.inference_mode():
    g=animate(rig,poses,0);model=SMPLXSurfaceLandmarks().cuda();aa=torch.cat([poses['root_pose'][:,None],poses['body_pose'],torch.zeros(1,33,3,device='cuda')],1)
    landmarks=model(aa,poses['scale'],poses['translation_world'])['joints55'][0].cpu().numpy()
    xyz=g['xyz'][g['opacity'].flatten()>.05].cpu().numpy();lo=np.minimum(np.quantile(xyz,.001,axis=0),landmarks.min(0))-.07;hi=np.maximum(np.quantile(xyz,.999,axis=0),landmarks.max(0))+.07
    target=(lo+hi)/2;distance=2.5*float((hi-lo).max());cameras=[];sheet=Image.new('RGB',(12*192,6*212),'white')
    for row,elevation in enumerate([-15,10,35]):
        for col,azimuth in enumerate(range(0,360,15)):
            e,a=np.deg2rad([elevation,azimuth]);origin=target+distance*np.array([np.cos(e)*np.sin(a),np.sin(e),np.cos(e)*np.cos(a)])
            forward=target-origin;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            W=np.eye(4);W[:3,:3]=np.stack([right,down,forward]);W[:3,3]=-W[:3,:3]@origin
            color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=torch.tensor(W[None],device='cuda',dtype=torch.float32),Ks=torch.tensor(K[None],device='cuda',dtype=torch.float32),width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
            with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,height,width)
            bodyalpha=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001);image=rgb[0,0].float()*bodyalpha[...,None]+1-bodyalpha[...,None]
            assert torch.isfinite(image).all() and torch.isfinite(bodyalpha).all()
            index=len(cameras);filename=f'{index:03d}';pic=Image.fromarray(np.uint8(image.cpu().numpy().clip(0,1)*255));pic.save(out/f'{filename}.png');Image.fromarray(np.uint8(bodyalpha.cpu().numpy().clip(0,1)*255)).save(out/f'{filename}_alpha.png')
            cameras.append({'id':index,'K':K.tolist(),'W':W.tolist(),'azimuth':azimuth,'elevation':elevation,'rgb':filename+'.png','alpha':filename+'_alpha.png'})
            sheet.paste(pic.resize((192,192)),((col%12)*192,(row*2+col//12)*212+20));ImageDraw.Draw(sheet).text(((col%12)*192+3,(row*2+col//12)*212+3),f'az{azimuth} elev{elevation}',fill='black')
        print('canonical elevation completed',elevation,flush=True)
    torch.save({k:v.cpu() for k,v in poses.items()},out/'pose.pt')
sheet.save(out/'all72_views.jpg');(out/'cameras.json').write_text(json.dumps({'width':width,'height':height,'views':cameras,'coordinate_convention':'Canonical SMPL-X Y-up model frame, frozen avatar scale, zero world translation, CV camera matrices'},indent=2))
(out/'report.json').write_text(json.dumps({'success':True,'views':len(cameras),'seconds':time.time()-start,'source_avatar':o.avatar,'shared_beta_norm':float(bundle['poses']['betas'].norm()),'pose':'A-pose shoulders ±.65rad, neutral hands/face, source shared beta and scale','scope':'Actual frozen Gaussian skinning + saved neural decoder from72 virtual cameras, RGB/soft-alpha targets for mesh distillation. These are model predictions, not new real observations, reconstructed sequence motion or validated mesh/appearance. Camera/time input provenance is inherited from the stated source avatar; this script does not construct an independent held-out reconstruction.'},indent=2));print('canonical72 views complete',flush=True)
