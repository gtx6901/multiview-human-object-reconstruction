"""Check batched textured RGB/depth parity and geometric gradients on CUDA."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import torch
from mesh_renderer import MeshRenderer
out=ROOT/'batched_uv_verification_v1';out.mkdir(exist_ok=False)
v=torch.tensor([[-.4,-.4,2],[.4,-.4,2],[.4,.4,2],[-.4,.4,2]],device='cuda');vertices=v[None].repeat(2,1,1).requires_grad_();f=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32);uv=torch.tensor([[0.,0],[1,0],[1,1],[0,1]],device='cuda');y,x=torch.meshgrid(torch.linspace(0,1,32,device='cuda'),torch.linspace(0,1,32,device='cuda'),indexing='ij');texture=torch.stack([x,y,1-x],-1).contiguous();K=torch.tensor([[[80.,0,31.5],[0,80.,31.5],[0,0,1]],[[70.,0,30.5],[0,75.,32.5],[0,0,1]]],device='cuda');W=torch.eye(4,device='cuda')[None].repeat(2,1,1);renderer=MeshRenderer();batch=renderer.depth_alpha_batch(vertices,f,K,W,64,64,uv=uv,texture=texture);checks=[]
for i in range(2):
 single=renderer(vertices[i],f,K[i],W[i],64,64,uv=uv,texture=texture);errors={k:float((single[k]-batch[k][i]).abs().max()) for k in ['alpha','depth','premultiplied_rgb']};assert max(errors.values())<1e-5;checks.append(errors)
weight=torch.linspace(0,1,64,device='cuda')[None,None,:,None];(batch['premultiplied_rgb']*weight).sum().backward();assert torch.isfinite(vertices.grad).all() and vertices.grad.abs().sum()>0
(out/'report.json').write_text(json.dumps({'success':True,'single_vs_batch_max_errors':checks,'position_gradient_l1':float(vertices.grad.abs().sum()),'scope':'Two distinct cameras on actual CUDA UV renderer, exact convention parity with independent single-mesh route plus nonzero finite geometry gradients. Component contract only, not helmet pose or texture accuracy.'},indent=2));print('batched UV verified',checks,flush=True)
