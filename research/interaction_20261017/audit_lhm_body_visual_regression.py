"""Same-camera comparison of the original LHM body and fitted body/hand replay."""
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
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import animate,pose_axis_angles
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
parser=argparse.ArgumentParser();parser.add_argument('--out',default='lhm_body_visual_regression_v1');parser.add_argument('--baseline',default='avatar_motion_pickup000_135_v1');parser.add_argument('--current',default='avatar_hands_all300_v1');parser.add_argument('--frames',type=int,nargs='+',default=[0,30,60,90,120,135]);parser.add_argument('--region',choices=['body','head'],default='body');options=parser.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.time()
baseline=torch.load(ROOT/options.baseline/'avatar.pt',weights_only=True);current=torch.load(ROOT/options.current/'avatar.pt',weights_only=True)
identical=[]
for section in ['rig','decoder']:
    assert baseline[section].keys()==current[section].keys()
    for key,value in baseline[section].items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,current[section][key]),(section,key);identical.append(section+'.'+key)
for key in ['features','background_embedding']:
    assert torch.equal(baseline[key],current[key]),key;identical.append(key)
assert baseline['decoder_config']==current['decoder_config']
def cuda(value):
    if isinstance(value,torch.Tensor):return value.cuda()
    if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
    return value
rig=cuda(current['rig']);features=current['features'].cuda().float();bg=current['background_embedding'][2].cuda().float();old=cuda(baseline['poses']);new=cuda(current['poses'])
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**current['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(current['decoder'],strict=True,assign=True);decoder.eval().cuda()
camera=json.loads((ROOT/'avatar_hands_all300_v1/cameras.json').read_text());native=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());real=next(v for v in native['views'] if v['serial']=='21305717')
views=[{'label':'real717','K':real['K'],'W':real['W']},*camera['views'][8:]];samples=options.frames;size=504;records=[]
def joints(poses,t):
    aa=pose_axis_angles(poses,t,'cuda');j,_=batch_rigid_transform(axis_angle_to_matrix(aa),rig['rest_joints'],rig['parents']);return j[0]*poses['scale']+poses['translation_world'][t]
with torch.inference_mode():
    for view in views:
        sheet=Image.new('RGB',(size*3,len(samples)*(size+24)),'white')
        W=torch.tensor(view['W'],device='cuda',dtype=torch.float32);K0=np.array(view['K'])
        for row,t in enumerate(samples):
            a=joints(old,t);b=joints(new,t)
            if options.region=='head':a=a[[12,15,23,24]];b=b[[12,15,23,24]]
            points=torch.cat([a,b]).cpu().numpy();cam=points@np.array(view['W'])[:3,:3].T+np.array(view['W'])[:3,3];q=cam@K0.T;uv=q[:,:2]/q[:,2:];lo=uv.min(0);hi=uv.max(0);center=(hi+lo)/2;extent=float(max(hi-lo)*(2.5 if options.region=='head' else 1.35));origin=center-extent/2;factor=size/extent
            affine=np.array([[factor,0,-factor*origin[0]],[0,factor,-factor*origin[1]],[0,0,1.]])
            K=torch.tensor(affine@K0,device='cuda',dtype=torch.float32)
            if view['label']=='real717':
                source=Image.open(ROOT/'helmet_native000_299_v1/21305717'/f'{t:04d}.png');photo=source.transform((size,size),Image.Transform.AFFINE,(1/factor,0,origin[0],0,1/factor,origin[1]),resample=Image.Resampling.BICUBIC);sheet.paste(photo,(0,row*(size+24)+24))
            for col,poses in enumerate([old,new],1):
                g=animate(rig,poses,t)
                color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=size,height=size,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB',near_plane=.01,far_plane=100.,eps2d=.3)
                with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,size,size)
                a=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001);arr=(rgb[0,0].float()*a[...,None]+1-a[...,None]).cpu().numpy();im=Image.fromarray(np.uint8(arr.clip(0,1)*255));im.save(out/f"{view['label']}_{t:04d}_{'old' if col==1 else 'fitted'}.png");sheet.paste(im,(col*size,row*(size+24)+24))
            draw=ImageDraw.Draw(sheet)
            for col,label in enumerate(['SOURCE (real717 only)','ORIGINAL LHM motion','SAME LHM + fitted body/hands']):draw.text((col*size+5,row*(size+24)+4),f'{view["label"]} f{t} | {label}',fill='black')
            records.append({'view':view['label'],'frame':t,'K':K.cpu().tolist(),'W':view['W'],'same_shared_appearance':True});print(view['label'],t,'compared',flush=True)
        sheet.save(out/f"{view['label']}_comparison.jpg",quality=94)
(out/'report.json').write_text(json.dumps({'success':True,'seconds':time.time()-start,'identical_appearance_tensors':identical,'records':records,'scope':'Original and current bundles have bit-identical Gaussian rig/attributes, feature tensors, background embedding and neural decoder weights/config. Same camera/crop and same neural renderer for old body-motion vs refitted body/hands at6 shared times and3views. Human-only isolation: helmet omitted in BOTH rendered panels; real photo includes the helmet. Does not certify that changed poses/contact or novel-view quality improved. Mesh-distilled body is not used.'},indent=2));print('LHM visual regression audit ready',flush=True)
