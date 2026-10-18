"""Locate which visibility gates remove real helmet texture evidence."""
import os
from pathlib import Path
import sys,argparse,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
from helmet_texture_sampling import NativeHelmetSampler
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--motion',required=True);p.add_argument('--mesh',required=True);p.add_argument('--body-depth',required=True);p.add_argument('--frames',nargs='+',type=int,default=[30,42,60,120,180,299]);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);torch.set_num_threads(4);start=time.time()
mesh=torch.load(ROOT/o.mesh/'mesh.pt',weights_only=True);motion=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],motion['vertices_local'][motion['faces'].long()],atol=1e-6)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());uvreport=json.loads((ROOT/o.mesh/'report.json').read_text());shellsource=uvreport.get('source_shell',uvreport.get('source'));assert shellsource
original=torch.load(ROOT/shellsource/'mesh.pt',weights_only=True);tm=trimesh.Trimesh(original['vertices_local'].numpy(),original['faces'].numpy(),process=False)
# Match the baker's actual smooth source normals, including UV seam duplication.
normals=torch.tensor(np.array(tm.vertex_normals)[mesh['source_vertex'].numpy()],device='cuda',dtype=torch.float32)
faces=mesh['faces'].long().cuda();vertices=mesh['vertices_local'].cuda();classes=mesh['face_classes'].cuda().long();ids=[]
for c in [0,1,2]:
 group=torch.where(classes==c)[0];ids.append(group[torch.linspace(0,len(group)-1,min(1200,len(group)),device='cuda').long()])
ids=torch.cat(ids);points=vertices[faces[ids]].mean(1);normal=torch.nn.functional.normalize(normals[faces[ids]].mean(1),dim=1);sampleclasses=classes[ids];sampler=NativeHelmetSampler(mesh,motion,meta,body_depth=o.body_depth);rows=[]
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  for f in o.frames:
   result=sampler.sample(vi,f,points,normal,sampleclasses,debug=True)
   if result is None:rows.append({'camera':v['serial'],'frame':f,'annotation_known':False});continue
   color,w,srgb,stats=result;selected={}
   for c in [0,1,2]:
    known=(sampleclasses==c)&(w>.03);selected[str(c)]={'sample_count':int(known.sum()),'mean_srgb':srgb[known].mean(0).cpu().tolist() if known.any() else None}
   rows.append({'camera':v['serial'],'frame':f,'annotation_known':True,'filter_counts':stats,'selected':selected})
  print('sampling gates audited',v['serial'],flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'motion':o.motion,'mesh':o.mesh,'body_depth':o.body_depth,'frames':o.frames,'points_per_class':{str(c):int((sampleclasses==c).sum()) for c in [0,1,2]},'records':rows,'seconds':time.time()-start,'scope':'Diagnostic only, exact baker smooth source normals at deterministic triangle centers. Cumulative and independent visibility-filter survivor counts locate evidence loss; selected RGB means are source-color diagnostics, not albedo or texture correctness. No thresholds relaxed, no source images exported, no assets modified.'},indent=2));print('native sampling audit complete',flush=True)
