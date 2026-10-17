"""Native real-image colors with mesh/body visibility and explicit class checks."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
import torch.nn.functional as F
from PIL import Image
from mesh_renderer import MeshRenderer
def srgb_to_linear(x):return torch.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def linear_to_srgb(x):
    x=x.clamp(0,1);return torch.where(x<=.0031308,x*12.92,1.055*x.pow(1/2.4)-.055)

class NativeHelmetSampler:
    def __init__(self,mesh,motion,meta,body_depth='body_depth_all300_v1'):
        self.vertices=mesh['vertices_local'].cuda();self.faces=mesh['faces'].cuda();self.classes=mesh['face_classes'].cuda().long()
        self.R=motion['R'].cuda();self.t=motion['translation_world'].cuda();self.scale=motion['logscale'].exp().cuda();self.meta=meta
        self.renderer=MeshRenderer();self.maskroot=ROOT/'sam2_all300_v1';m=json.loads((self.maskroot/'report.json').read_text());self.records={(r['serial'],r['frame']):r for r in m['records']}
        self.cache_view=None;self.cache=None;self.body_depth=body_depth
    def sample(self,vi,f,points,normals,classes,debug=False):
        v=self.meta['views'][vi];record=self.records[(v['serial'],f)]
        if record['mask'] is None:return None
        if self.cache_view!=vi:
            d=np.load(ROOT/self.body_depth/f"{v['serial']}.npz")
            self.cache={k:d[k] for k in ['depth','alpha','hand_unknown']};self.cache_view=vi
            if 'decoded_alpha' in d:self.cache['decoded_alpha']=d['decoded_alpha']
        W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);K=torch.tensor(v['K'],device='cuda',dtype=torch.float32)
        vertices=self.vertices@self.R[f].T*self.scale+self.t[f]
        mesh=self.renderer.depth_alpha_batch(vertices[None],self.faces,K[None],W[None],2048,1536)
        world=points@self.R[f].T*self.scale+self.t[f];pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;uv=q[:,:2]/q[:,2:].clamp_min(.001)
        grid=((uv+.5)/torch.tensor([2048,1536],device='cuda')*2-1).reshape(1,-1,1,2)
        def sample_tensor(value,mode='nearest'):
            if value.ndim==2:value=value[None,None]
            return F.grid_sample(value.float(),grid,mode=mode,padding_mode='zeros',align_corners=False)[0,:,:,0].T
        lab=torch.tensor(np.array(Image.open(self.maskroot/record['labels'])),device='cuda')
        depth=sample_tensor(mesh['depth'][0])[:,0];hit=sample_tensor(mesh['hard_alpha'][0])[:,0]>.5
        face=sample_tensor(mesh['face_id'][0])[:,0].long().clamp(0,len(self.classes)-1)
        tag=self.classes[face];normal=normals@self.R[f].T
        camera=-W[:3,:3].T@W[:3,3];direction=F.normalize(camera-world,dim=1);cosine=(normal*direction).sum(1).clamp(0,1)
        bd=sample_tensor(torch.tensor(self.cache['depth'][f],device='cuda'))[:,0]
        ba=sample_tensor(torch.tensor(self.cache['alpha'][f],device='cuda'))[:,0]
        hand=sample_tensor(torch.tensor(self.cache['hand_unknown'][f],device='cuda'))[:,0]
        good=(pc[:,2]>.01)&(uv[:,0]>=0)&(uv[:,0]<2048)&(uv[:,1]>=0)&(uv[:,1]<1536)&hit
        filters={'candidate_label':sample_tensor(lab)[:,0]==1,'mesh_depth':(pc[:,2]-depth).abs()<.002,'surface_class':tag==classes,'view_angle':cosine>.15,'hand_unknown':hand<.5,'body_depth':(ba<.5)|(pc[:,2]<bd+.01)}
        if 'decoded_alpha' in self.cache:
            # Missing reliable depth inside the neural silhouette is UNKNOWN,
            # never evidence that the human is absent (especially helmet lining).
            silhouette=sample_tensor(torch.tensor(self.cache['decoded_alpha'][f],device='cuda'))[:,0]
            filters['body_depth_uncertainty']=~((silhouette>.5)&(ba<.5))
        stats={}
        if debug:stats['projected_hit']={str(c):int((good&(classes==c)).sum()) for c in [0,1,2]}
        for key,keep in filters.items():
            good&=keep
            if debug:stats[key]={str(c):int((good&(classes==c)).sum()) for c in [0,1,2]}
        rgb=torch.tensor(np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB')),device='cuda').permute(2,0,1)[None].float()/255
        colors=sample_tensor(rgb,mode='bilinear')
        # Clipped white is still an observation, but carries less color information.
        saturation=torch.where((colors>.985).all(1),.25,1.)
        weight=cosine.pow(3)*saturation*good
        if debug:return srgb_to_linear(colors),weight,colors,stats
        return srgb_to_linear(colors),weight,colors
