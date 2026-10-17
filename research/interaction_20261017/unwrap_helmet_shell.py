"""Actual xatlas UVs on the candidate shell, preserving every triangle position."""
import os
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,xatlas,trimesh
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_uv_all300_v3');p.add_argument('--source',default='helmet_shell_all300_v5');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();source=ROOT/options.source/'mesh.pt';mesh=torch.load(source,weights_only=True)
vertices=mesh['vertices_local'].numpy().astype('float32');faces=mesh['faces'].numpy().astype('uint32')
atlas=xatlas.Atlas();atlas.add_mesh(vertices,faces);pack=xatlas.PackOptions();pack.resolution=1024;pack.padding=8;atlas.generate(pack_options=pack)
mapping,indices,uv=atlas[0];mapping=np.asarray(mapping).copy();indices=np.asarray(indices).copy();uv=np.asarray(uv).copy()
assert np.isfinite(uv).all() and uv.min()>=0 and uv.max()<=1
assert np.array_equal(mapping[indices],faces),'Unexpected face reordering: resolve correspondence before carrying material classes'
newvertices=vertices[mapping];error=float(np.abs(newvertices[indices]-vertices[faces]).max());assert error==0
result={**mesh,'vertices_local':torch.tensor(newvertices),'faces':torch.tensor(indices.astype('int32')),'uv':torch.tensor(uv),'colors':torch.full((len(newvertices),3),.65),'source_vertex':torch.tensor(mapping.astype('int64'))}
torch.save(result,out/'mesh.pt')
tm=trimesh.Trimesh(newvertices,indices,process=False);tm.visual=trimesh.visual.TextureVisuals(uv=uv);tm.export(out/'helmet_uv.obj')
layout=Image.new('RGB',(1024,1024),'white');draw=ImageDraw.Draw(layout);pixels=np.stack([uv[:,0]*1023,(1-uv[:,1])*1023],1)
for face in indices:draw.line([tuple(pixels[k]) for k in [*face,face[0]]],fill=(80,80,80),width=1)
layout.save(out/'uv_layout.png')
report={'success':True,'source_mesh_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.time()-start,'vertices_before':len(vertices),'vertices_after_seam_split':len(newvertices),'faces':len(indices),'atlas_width':atlas.width,'atlas_height':atlas.height,'atlas_utilization':float(atlas.utilization),'texture_resolution_planned':[1024,1024],'padding_requested':8,'max_triangle_position_error':error,'scope':'Real xatlas charting/packing on candidate shell, triangle order and positions exactly preserved, seams duplicate vertices. UV OBJ/mesh contract only; no recovered texture or accepted helmet shape. Keep outer/inner/rim face classes for observed-versus-inferred material bookkeeping.'}
report['source']=options.source;report['charts']=atlas.get_mesh_chart_count(0)
(out/'report.json').write_text(json.dumps(report,indent=2));print('xatlas shell UV generated',flush=True)
