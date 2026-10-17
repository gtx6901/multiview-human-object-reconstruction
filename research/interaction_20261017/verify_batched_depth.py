"""Batched CV depth fixture compared with the independently checked single renderer."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import torch
from mesh_renderer import MeshRenderer
out=ROOT/'batched_depth_verification_v2';out.mkdir(exist_ok=False)
device='cuda';K=torch.tensor([[40.,0,31.5],[0,40.,31.5],[0,0,1]],device=device);W=torch.eye(4,device=device)
v=torch.tensor([[[-.4,-.4,2],[.4,-.4,2],[.4,.4,2],[-.4,.4,2]],[[-.6,-.6,3],[.6,-.6,3],[.6,.6,3],[-.6,.6,3]]],device=device,requires_grad=True)
faces=torch.tensor([[0,1,2],[0,2,3]],device=device,dtype=torch.int32);renderer=MeshRenderer();batch=renderer.depth_alpha_batch(v,faces,K[None].expand(2,-1,-1),W[None].expand(2,-1,-1),64,64)
color=torch.tensor([[1.,0,0]]*4,device=device)
colored=renderer.depth_alpha_batch(v,faces,K[None].expand(2,-1,-1),W[None].expand(2,-1,-1),64,64,colors=color)
checks=[]
for i,z in enumerate([2.,3.]):
    single=renderer(v[i],faces,K,W,64,64)
    for key in ['alpha','hard_alpha','depth']:assert torch.allclose(batch[key][i],single[key],atol=1e-6)
    assert abs(float(batch['depth'][i,32,32].detach())-z)<1e-6
    single_color=renderer(v[i],faces,K,W,64,64,color)
    assert torch.allclose(colored['premultiplied_rgb'][i],single_color['premultiplied_rgb'],atol=1e-6)
    checks.append({'camera_z':z,'single_batch_max_depth_difference':float((batch['depth'][i]-single['depth']).abs().max().detach())})
batch['depth'][:,32,32].sum().backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
(out/'report.json').write_text(json.dumps({'success':True,'checks':checks,'gradient_l1':float(v.grad.abs().sum()),'shared_vertex_color_batch_parity':True,'scope':'Synthetic batch/single depth and RGB parity, positive CV Z and finite depth gradients; no reconstruction-quality claim.'},indent=2));print('batched depth/color checks passed',flush=True)
