"""Independent four-corner OBJ-UV / top-left RGB orientation fixture."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import torch
from mesh_renderer import MeshRenderer
out=ROOT/'uv_rendering_verification_v1';out.mkdir(exist_ok=False)
v=torch.tensor([[-1.,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],device='cuda');f=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32)
uv=torch.tensor([[0.,1],[1,1],[1,0],[0,0]],device='cuda');tex=torch.tensor([[[1.,0,0],[0,1,0]],[[0,0,1],[1,1,0]]],device='cuda')
K=torch.tensor([[40.,0,31.5],[0,40.,31.5],[0,0,1]],device='cuda');W=torch.eye(4,device='cuda')
result=MeshRenderer()(v,f,K,W,64,64,uv=uv,texture=tex)
checks=[]
for y,x,color in [(16,16,[1.,0,0]),(16,47,[0.,1,0]),(47,16,[0.,0,1]),(47,47,[1.,1,0])]:
    error=float((result['premultiplied_rgb'][y,x]-torch.tensor(color,device='cuda')).abs().max());assert error<1e-6
    checks.append({'pixel_xy':[x,y],'expected_rgb':color,'max_error':error})
(out/'report.json').write_text(json.dumps({'success':True,'checks':checks,'scope':'Known RGB corner fixture verifies standard upward-V UVs against top-left stored texture and CV raster output. Not reconstruction-quality evidence.'},indent=2));print('UV corner orientation verified',flush=True)
