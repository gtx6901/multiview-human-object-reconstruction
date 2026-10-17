"""Fixed full-action framing from the ACTUAL LHM Gaussians, no mesh avatar."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from avatar_rig import animate
out=ROOT/'lhm_full_action_framing_v1';out.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.time()
bundle=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True)
def cuda(x):
    if isinstance(x,torch.Tensor):return x.cuda()
    if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
    return x
rig=cuda(bundle['rig']);poses=cuda(bundle['poses']);motion=cuda(torch.load(ROOT/'helmet_material_pose_all300_v1/sequence.pt',weights_only=True));meta=json.loads((ROOT/'avatar_hands_all300_v1/cameras.json').read_text());width,height=meta['width'],meta['height'];newwidth,newheight=1008,756;assert newwidth%7==newheight%7==0
novel=meta['views'][8:];limits=[{'lo':np.array([np.inf,np.inf]),'hi':np.array([-np.inf,-np.inf])} for _ in novel];positive_min=[np.inf]*len(novel)
with torch.inference_mode():
    for t in range(300):
        g=animate(rig,poses,t);keep=g['opacity'].flatten()>.01;xyz=g['xyz'][keep];radius=3*g['scaling'][keep].max(1).values
        obj=motion['vertices_local']@motion['R'][t].T*motion['logscale'].exp()+motion['translation_world'][t]
        for i,v in enumerate(novel):
            W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);K=torch.tensor(v['K'],device='cuda',dtype=torch.float32);pc=xyz@W[:3,:3].T+W[:3,3];positive_min[i]=min(positive_min[i],float(pc[:,2].min()));assert (pc[:,2]>.01).all()
            q=pc@K.T;uv=q[:,:2]/q[:,2:];r=radius*max(float(K[0,0]),float(K[1,1]))/pc[:,2];low=(uv-r[:,None]).min(0).values.cpu().numpy();high=(uv+r[:,None]).max(0).values.cpu().numpy()
            cp=obj@W[:3,:3].T+W[:3,3];oq=cp@K.T;ouv=oq[:,:2]/oq[:,2:];low=np.minimum(low,ouv.min(0).values.cpu().numpy());high=np.maximum(high,ouv.max(0).values.cpu().numpy());limits[i]['lo']=np.minimum(limits[i]['lo'],low);limits[i]['hi']=np.maximum(limits[i]['hi'],high)
        if t%75==0:print('actual Gaussian framing frame',t,flush=True)
    transforms=[]
    # First frame more tightly at the OLD output size, then increase resolution.
    S=np.array([[newwidth/width,0,.5*newwidth/width-.5],[0,newheight/height,.5*newheight/height-.5],[0,0,1.]])
    for i,v in enumerate(novel):
        lo,hi=limits[i]['lo'],limits[i]['hi'];center=(lo+hi)/2;factor=min((width-32)/(hi[0]-lo[0]),(height-32)/(hi[1]-lo[1]));A=np.array([[factor,0,width/2-factor*center[0]],[0,factor,height/2-factor*center[1]],[0,0,1.]])
        v['K']=(S@A@np.array(v['K'])).tolist();v['framing']='Fixed across300 frames from opacity>.01 actual LHM Gaussian means plus3max-scale projected radii and candidate helmet vertices. No SMPL-X mesh substitute. Camera W unchanged.32old-pixel margin.';transforms.append({'label':v['label'],'zoom_factor_before_resolution_increase':factor,'old_pixel_bounds':[lo.tolist(),hi.tolist()],'minimum_gaussian_positive_camera_z':positive_min[i]})
    for v in meta['views'][:8]:v['K']=(S@np.array(v['K'])).tolist()
    meta.update(width=newwidth,height=newheight);(out/'cameras.json').write_text(json.dumps(meta,indent=2));(out/'report.json').write_text(json.dumps({'success':True,'frames':300,'output_resolution':[newwidth,newheight],'sources':{'avatar':'body_head_faces_all300_v1','helmet_motion':'helmet_material_pose_all300_v1'},'fixed_novel_camera_transforms':transforms,'seconds':time.time()-start,'scope':'Resolution/framing candidate only, not geometry or reconstruction-quality improvement. Same two camera extrinsics; real cameras scaled by exact pixel-center affine. Includes all300 actual LHM Gaussian bounds, not just body joints with oversized fixed world padding. Small Gaussian opacity threshold and3scale radius are framing heuristics; updated helmet motion must be checked for viewport coverage.'},indent=2));print('LHM full-action camera candidate ready',flush=True)
