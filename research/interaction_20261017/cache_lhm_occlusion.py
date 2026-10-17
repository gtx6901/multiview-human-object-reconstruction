"""Cache conservative original-LHM occlusion, never claim a Gaussian surface truth."""
import os
from pathlib import Path
import sys,json,time,importlib.util,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016'
sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"])
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='body_head_faces_all300_v1');p.add_argument('--bare-cache',default='body_depth_head_faces_all300_v1');p.add_argument('--frames',type=int,nargs='+');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
def move(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    return x
b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True)
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**b['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(b['decoder'],strict=True,assign=True);decoder.eval().cuda()
rig=move(b['rig']);poses=move(b['poses']);features=b['features'].cuda().float();bg=b['background_embedding'][2].cuda().float()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());frameids=poses['frames'].tolist();selected=frameids if o.frames is None else o.frames
assert len(set(selected))==len(selected) and set(selected)<=set(frameids)
width,height=504,378;A=np.array([[width/2048,0,(width/2048-1)/2],[0,height/1536,(height/1536-1)/2],[0,0,1.]])
records=[]
with torch.inference_mode():
 for v in meta['views']:
    bare=np.load(ROOT/o.bare_cache/f"{v['serial']}.npz")
    K=torch.tensor(A@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
    depths=[];alphas=[];stds=[];decoder_masks=[];changes=[]
    for f in selected:
        t=frameids.index(f);g=animate(rig,poses,t);z=(g['xyz']@W[:3,:3].T+W[:3,3])[:,2]
        # A second depth moment identifies ambiguous foreground/background mixtures.
        colors=torch.cat([g['shs'].squeeze(1),features,z.square()[:,None]],1)
        background=torch.cat([torch.ones(3,device='cuda'),bg,torch.zeros(1,device='cuda')])[None]
        color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=colors,viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=background,packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
        with torch.autocast('cuda',dtype=torch.bfloat16):_,mask=decoder.predict_rgbs(None,color[0,...,3:-2].permute(2,0,1)[None],None,height,width)
        a=alpha[0,...,0];d=color[0,...,-1];variance=(color[0,...,-2]/a.clamp_min(1e-6)-d.square()).clamp_min(0);std=variance.sqrt();m=mask[0,0].float().squeeze()
        reliable=((a>.95)&(m>.95)&(std<.01)&(d>.01)).cpu().numpy().astype('uint8')
        reliable=cv2.erode(reliable,np.ones((3,3),np.uint8))
        resize=lambda value:cv2.resize(value,(256,192),interpolation=cv2.INTER_NEAREST_EXACT)
        core=resize(reliable);depth=resize(d.cpu().numpy());depth[core==0]=0
        assert np.isfinite(depth).all() and (depth[core>0]>0).all()
        depths.append(depth);alphas.append(core);stds.append(resize(std.cpu().numpy()));decoder_masks.append(resize(m.cpu().numpy()))
        old=bare['alpha'][t]>0;new=core>0;changes.append({'frame':f,'reliable_pixels':int(new.sum()),'reliable_outside_bare_pixels':int((new&~old).sum()),'bare_pixels_without_reliable_lhm_depth':int((old&~new).sum())})
    np.savez_compressed(out/f"{v['serial']}.npz",depth=depths,alpha=alphas,hand_unknown=bare['hand_unknown'][[frameids.index(f) for f in selected]],depth_std=stds,decoded_alpha=decoder_masks,frames=np.array(selected),K=bare['K'],W=bare['W'])
    records.append({'serial':v['serial'],'frames':changes});(out/'partial.json').write_text(json.dumps({'completed_views':len(records),'views':records},indent=2));print('LHM occlusion cached',v['serial'],flush=True)
report={'success':True,'frames':selected,'avatar_source':o.avatar,'bare_cache_source':o.bare_cache,'views':records,'seconds':time.time()-start,'internal_resolution':[width,height],'cache_resolution':[256,192],'scope':'Original accepted LHM rig, Gaussian appearance and neural decoder, no helmet clipping. Positive OpenCV Z is opacity-normalized expected Gaussian-center depth, NOT a nearest opaque surface or decoded geometric surface. Extra second-depth moment rejects mixtures with std>=.01 normalized-world units; accumulated and decoded alpha must both exceed.95, then one-pixel eroded. Excluded pixels remain unknown, no bare-depth fallback. Cache hand uncertainty comes unchanged from fitted bare-hand landmarks. Rendering at504x378 and explicit pixel-center K; nearest-exact resampling to existing256x192 cache convention. No metric-unit, correct-contact or novel-view quality claim.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('LHM occlusion cache complete',flush=True)
