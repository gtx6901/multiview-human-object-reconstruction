"""Portrait framing candidate from actual full-action Gaussian bounds."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from avatar_rig import animate
out=ROOT/'lhm_portrait_framing_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
def cuda(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 return x
b=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True);rig=cuda(b['rig']);poses=cuda(b['poses'])
m=cuda(torch.load(ROOT/'helmet_native_roi_all300_v1/sequence.pt',weights_only=True));meta=json.loads((ROOT/'lhm_full_action_framing_v1/cameras.json').read_text());oldw,oldh=meta['width'],meta['height'];nw,nh=1134,1512
limits=[dict(lo=np.full(2,np.inf),hi=np.full(2,-np.inf)) for _ in meta['views'][8:]]
with torch.inference_mode():
 for t in range(300):
  g=animate(rig,poses,t);keep=g['opacity'].flatten()>.01;xyz=g['xyz'][keep];radius=3*g['scaling'][keep].max(1).values;obj=m['vertices_local']@m['R'][t].T*m['logscale'].exp()+m['translation_world'][t]
  for v,limit in zip(meta['views'][8:],limits):
   W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);K=torch.tensor(v['K'],device='cuda',dtype=torch.float32);pc=xyz@W[:3,:3].T+W[:3,3];assert (pc[:,2]>.01).all();q=pc@K.T;uv=q[:,:2]/q[:,2:];r=radius*max(float(K[0,0]),float(K[1,1]))/pc[:,2]
   oc=obj@W[:3,:3].T+W[:3,3];oq=oc@K.T;ou=oq[:,:2]/oq[:,2:]
   limit['lo']=np.minimum(limit['lo'],np.minimum((uv-r[:,None]).min(0).values.cpu().numpy(),ou.min(0).values.cpu().numpy()));limit['hi']=np.maximum(limit['hi'],np.maximum((uv+r[:,None]).max(0).values.cpu().numpy(),ou.max(0).values.cpu().numpy()))
records=[]
for v,limit in zip(meta['views'][8:],limits):
 lo,hi=limit['lo'],limit['hi'];center=(lo+hi)/2;s=min((nw-64)/(hi[0]-lo[0]),(nh-64)/(hi[1]-lo[1]));A=np.array([[s,0,(nw-1)/2-s*center[0]],[0,s,(nh-1)/2-s*center[1]],[0,0,1.]])
 v['K']=(A@np.array(v['K'])).tolist();records.append({'label':v['label'],'isotropic_pixel_magnification_vs_accepted_video':float(s),'source_pixel_bounds':[lo.tolist(),hi.tolist()]})
# Reference camera images are letterboxed, never stretched to portrait aspect.
s=min(nw/oldw,nh/oldh);A=np.array([[s,0,(nw-1)/2-s*(oldw-1)/2],[0,s,(nh-1)/2-s*(oldh-1)/2],[0,0,1.]])
for v in meta['views'][:8]:v['K']=(A@np.array(v['K'])).tolist()
meta.update(width=nw,height=nh);(out/'cameras.json').write_text(json.dumps(meta,indent=2));(out/'report.json').write_text(json.dumps({'success':True,'seconds':time.time()-start,'resolution':[nw,nh],'frames_bounded':300,'novel_views':records,'scope':'Framing/resolution only, same accepted human and camera extrinsics. Fixed full300 Gaussian means opacity>.01 plus3scale radii and ROI helmet bounds,64pixel margins. No geometry/detail-quality improvement claimed; decoder must be visually tested at this resolution. Source and real cameras letterboxed isotropically.'},indent=2));print('portrait cameras prepared',flush=True)
