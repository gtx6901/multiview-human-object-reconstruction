"""Differentiable CV-camera mesh rendering with top-left pixel centers and depth."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import torch
import nvdiffrast.torch as dr

class MeshRenderer:
    def __init__(self):
        self.context=dr.RasterizeCudaContext()

    def depth_alpha_batch(self,vertices,faces,K,W,width,height,near=.01,far=100.,colors=None):
        """Instanced opaque depths in the same CV convention as __call__."""
        pc=torch.einsum('bij,bnj->bni',W[:,:3,:3],vertices)+W[:,None,:3,3]
        image=torch.einsum('bij,bnj->bni',K,pc)
        clip=torch.stack([2*(image[...,0]+.5*pc[...,2])/width-pc[...,2],
            pc[...,2]-2*(image[...,1]+.5*pc[...,2])/height,
            (far+near)/(far-near)*pc[...,2]-2*far*near/(far-near),pc[...,2]],-1).contiguous()
        faces=faces.to(torch.int32).contiguous();rast,_=dr.rasterize(self.context,clip,faces,resolution=[height,width])
        hard=(rast[...,3:]>0).float();depth,_=dr.interpolate(pc[...,2:3].contiguous(),rast,faces)
        alpha=dr.antialias(hard,rast,clip,faces)
        result={'alpha':alpha.flip(1)[...,0],'depth':depth.flip(1)[...,0],'hard_alpha':hard.flip(1)[...,0],
                'face_id':rast.flip(1)[...,3].to(torch.long)-1}
        if colors is not None:
            attrs=colors[None].expand(len(vertices),-1,-1).contiguous();rgb,_=dr.interpolate(attrs,rast,faces)
            result['premultiplied_rgb']=dr.antialias(rgb*hard,rast,clip,faces).flip(1)
        return result

    def alpha_batch(self,vertices,faces,K,W,width,height,near=.01,far=100.):
        """Instanced differentiable silhouettes for explicit B independent cameras."""
        assert vertices.ndim==3 and K.ndim==3 and W.ndim==3
        pc=torch.einsum('bij,bnj->bni',W[:,:3,:3],vertices)+W[:,None,:3,3]
        image=torch.einsum('bij,bnj->bni',K,pc)
        clip=torch.stack([2*(image[...,0]+.5*pc[...,2])/width-pc[...,2],
            pc[...,2]-2*(image[...,1]+.5*pc[...,2])/height,
            (far+near)/(far-near)*pc[...,2]-2*far*near/(far-near),pc[...,2]],-1).contiguous()
        faces=faces.to(torch.int32).contiguous()
        rast,_=dr.rasterize(self.context,clip,faces,resolution=[height,width])
        alpha=(rast[...,3:]>0).float()
        return dr.antialias(alpha,rast,clip,faces).flip(1)[...,0]

    def __call__(self,vertices,faces,K,W,width,height,colors=None,near=.01,far=100.,uv=None,texture=None):
        """W maps common world coordinates to OpenCV x-right/y-down/z-forward."""
        pc=vertices@W[:3,:3].T+W[:3,3]
        image=pc@K.T
        clip=torch.stack([2*(image[:,0]+.5*pc[:,2])/width-pc[:,2],
            pc[:,2]-2*(image[:,1]+.5*pc[:,2])/height,
            (far+near)/(far-near)*pc[:,2]-2*far*near/(far-near),pc[:,2]],-1)[None].contiguous()
        faces=faces.to(torch.int32).contiguous()
        rast,_=dr.rasterize(self.context,clip,faces,resolution=[height,width])
        alpha=(rast[...,3:]>0).float()
        depth,_=dr.interpolate(pc[None,:,2:3].contiguous(),rast,faces)
        if uv is not None and texture is not None:
            coordinates,_=dr.interpolate(uv[None].contiguous(),rast,faces)
            coordinates=torch.stack([coordinates[...,0],1-coordinates[...,1]],-1).contiguous()
            rgb=dr.texture(texture[None].contiguous(),coordinates,filter_mode='linear',boundary_mode='clamp')
        else:
            if colors is None:colors=torch.ones_like(vertices)
            rgb,_=dr.interpolate(colors[None].contiguous(),rast,faces)
        rgba=dr.antialias(torch.cat([rgb*alpha,alpha],-1),rast,clip,faces)
        # Rasterizer is bottom-left, input RGB and K are top-left.
        rgba=rgba[0].flip(0);depth=depth[0].flip(0)
        return {'premultiplied_rgb':rgba[...,:3],'alpha':rgba[...,3],
                'depth':depth[...,0],'hard_alpha':alpha[0,...,0].flip(0)}

def probe():
    import json,numpy as np
    from PIL import Image
    out=ROOT/'mesh_renderer_probe_v1';out.mkdir(exist_ok=False)
    # Two quads project to the same independent analytic box, near red/far blue.
    def quad(z):return [[-.2*z,-.2*z,z],[.2*z,-.2*z,z],[.2*z,.2*z,z],[-.2*z,.2*z,z]]
    vertices=torch.tensor(quad(2)+quad(3),device='cuda',dtype=torch.float32,requires_grad=True)
    faces=torch.tensor([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],device='cuda',dtype=torch.int32)
    colors=torch.tensor([[1.,0,0]]*4+[[0.,0,1.]]*4,device='cuda')
    K=torch.tensor([[40.,0,31.5],[0,40.,31.5],[0,0,1]],device='cuda');W=torch.eye(4,device='cuda')
    renderer=MeshRenderer();r=renderer(vertices,faces,K,W,64,64,colors)
    assert torch.allclose(r['premultiplied_rgb'][32,32],torch.tensor([1.,0,0],device='cuda'))
    assert abs(float(r['depth'][32,32])-2)<1e-5
    ys,xs=torch.where(r['hard_alpha']>.5)
    assert [int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())]==[24,39,24,39]
    gx=torch.linspace(0,1,64,device='cuda')[None,:]
    (r['premultiplied_rgb'][...,0]*gx).sum().backward()
    assert torch.isfinite(vertices.grad).all() and vertices.grad.abs().sum()>0
    second=vertices.detach().clone();second[:4]*=2;second[4:]*=.5
    q=renderer(second,faces,K,W,64,64,colors)
    assert torch.allclose(q['premultiplied_rgb'][32,32],torch.tensor([0.,0,1.],device='cuda'))
    assert abs(float(q['depth'][32,32])-1.5)<1e-5
    for n,p in [('near_red',r),('near_blue',q)]:
        rgb=p['premultiplied_rgb']+1-p['alpha'][...,None]
        Image.fromarray(np.uint8(rgb.detach().cpu().numpy().clip(0,1)*255)).resize((512,512)).save(out/(n+'.png'))
    report={'success':True,'analytic_pixel_box_inclusive':[24,39,24,39],'center_depths':[float(r['depth'][32,32]),float(q['depth'][32,32])],
        'position_gradient_l1':float(vertices.grad.abs().sum()),'tests':['CV pixel center projection','top-left output','near surface occlusion','positive camera-space depth','finite nonzero antialias position gradients'],
        'scope':'Independent synthetic geometric fixture verifies mesh coordinate/depth component only, not reconstructed helmet or mesh-Gaussian compositing.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':probe()
