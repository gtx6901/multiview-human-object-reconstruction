"""Depth-ordered two-layer compositing; Gaussian expected depth is approximate.

Mesh depth is an opaque surface, Gaussian body depth summarizes a distribution.
This is a pilot interface, not exact interleaving of arbitrary Gaussian volumes.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch

def mesh_visible_opacities(means,opacities,K,W,mesh,depth_tolerance=0.):
    """Cull Gaussian centers behind the nearest opaque mesh before decoding.

    This is the center-depth approximation: an extended Gaussian can straddle
    the mesh. It is not exact per-ray Gaussian/triangle interleaving.
    Outside-image and unoccluded centers keep their original opacity.
    """
    pc=means@W[:3,:3].T+W[:3,3]
    image=pc@K.T
    uv=image[:,:2]/pc[:,2:3].clamp_min(1e-8)
    height,width=mesh['depth'].shape
    xy=torch.floor(uv+.5).long()
    inside=(pc[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    x=xy[:,0].clamp(0,width-1);y=xy[:,1].clamp(0,height-1)
    md=mesh['depth'][y,x]
    blocked=inside&(mesh['hard_alpha'][y,x]>.5)&(md>0)&(pc[:,2]>md+depth_tolerance)
    return opacities*~blocked.reshape(opacities.shape),blocked

def composite_clipped_body(body_premul,body_alpha,mesh,background=None):
    """Body decoded from mesh-clipped Gaussians, over the opaque mesh layer."""
    if background is None:background=torch.ones(3,device=body_premul.device)
    ma=mesh['alpha'];alpha=body_alpha+(1-body_alpha)*ma
    rgb=body_premul+(1-body_alpha[...,None])*mesh['premultiplied_rgb']+(1-alpha[...,None])*background
    return {'rgb':rgb,'alpha':alpha,'body_first':body_alpha>1e-5}

def composite(body_premul,body_alpha,body_depth,mesh,background=None):
    if background is None:background=torch.ones(3,device=body_premul.device)
    ma=mesh['alpha'];mp=mesh['premultiplied_rgb'];md=mesh['depth']
    body_first=(body_alpha>1e-5)&((ma<=1e-5)|((body_depth>0)&(body_depth<md)))
    front=torch.where(body_first[...,None],body_premul,mp)
    back=torch.where(body_first[...,None],mp,body_premul)
    fa=torch.where(body_first,body_alpha,ma);ba=torch.where(body_first,ma,body_alpha)
    alpha=fa+(1-fa)*ba
    rgb=front+(1-fa[...,None])*back+(1-alpha[...,None])*background
    return {'rgb':rgb,'alpha':alpha,'body_first':body_first}

def probe():
    import importlib.util,json,numpy as np
    from PIL import Image
    from mesh_renderer import MeshRenderer
    spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['gsplat.csrc']=module
    from gsplat import rasterization
    out=ROOT/'hybrid_compositor_probe_v1';out.mkdir(exist_ok=False)
    W=torch.eye(4,device='cuda');K=torch.tensor([[40.,0,31.5],[0,40.,31.5],[0,0,1]],device='cuda')
    color,alpha,_=rasterization(means=torch.tensor([[0.,0,2]],device='cuda'),quats=torch.tensor([[1.,0,0,0]],device='cuda'),
        scales=torch.tensor([[.2,.2,.03]],device='cuda'),opacities=torch.tensor([.9],device='cuda'),colors=torch.tensor([[0.,0,1]],device='cuda'),
        viewmats=W[None],Ks=K[None],width=64,height=64,backgrounds=torch.zeros(1,3,device='cuda'),render_mode='RGB+ED',packed=False)
    body=color[0,...,:3];depth=color[0,...,3];alpha=alpha[0,...,0]
    assert abs(float(depth[32,32])-2)<1e-4
    renderer=MeshRenderer();checks=[];sheet=Image.new('RGB',(1024,512),'white')
    for i,z in enumerate([1.5,2.5]):
        vertices=torch.tensor([[-.5,-.5,z],[.5,-.5,z],[.5,.5,z],[-.5,.5,z]],device='cuda')
        faces=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32)
        mesh=renderer(vertices,faces,K,W,64,64,torch.tensor([[1.,0,0]]*4,device='cuda'))
        result=composite(body,alpha,depth,mesh)
        expected=torch.tensor([1.,0,0],device='cuda') if z<2 else body[32,32]+torch.tensor([1.,0,0],device='cuda')*(1-alpha[32,32])
        error=float((result['rgb'][32,32]-expected).abs().max());assert error<1e-5
        Image.fromarray(np.uint8(result['rgb'].cpu().numpy().clip(0,1)*255)).resize((512,512)).save(out/f'mesh_z{z}.png')
        sheet.paste(Image.open(out/f'mesh_z{z}.png'),(i*512,0))
        checks.append({'mesh_depth':z,'body_depth':float(depth[32,32]),'center_max_rgb_error':error,'body_first':bool(result['body_first'][32,32])})
    sheet.save(out/'depth_order.jpg')
    report={'success':True,'checks':checks,'scope':'Real GS rasterization + real mesh rasterization, analytic near/far color and camera-depth check. Does not verify neural alpha edges, multilayer Gaussian depth, or real helmet/body contact.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':probe()
