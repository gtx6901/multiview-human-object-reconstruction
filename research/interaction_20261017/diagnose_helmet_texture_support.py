"""Find which actual visibility filters discard held lining observations."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
from helmet_texture_sampling import NativeHelmetSampler
out=ROOT/'helmet_inspection_texture_support_v1';out.mkdir(exist_ok=False);torch.set_num_threads(4)
mesh=torch.load(ROOT/'helmet_inspection_uv_v1/mesh.pt',weights_only=True);motion=torch.load(ROOT/'helmet_inspection_shell_fit_all300_v1/sequence.pt',weights_only=True);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());tm=trimesh.Trimesh(mesh['vertices_local'].numpy(),mesh['faces'].numpy(),process=False);ids=[]
for c in [0,1,2]:
 candidates=torch.where(mesh['face_classes']==c)[0];ids.extend(candidates[torch.linspace(0,len(candidates)-1,min(1000,len(candidates))).long()].tolist())
points=torch.tensor(np.array(tm.triangles_center)[ids],device='cuda',dtype=torch.float32);normals=torch.tensor(np.array(tm.face_normals)[ids],device='cuda',dtype=torch.float32);classes=mesh['face_classes'][ids].cuda();sampler=NativeHelmetSampler(mesh,motion,meta,body_depth='lhm_occlusion_all300_v1');records=[]
with torch.inference_mode():
 for f in [60,70,180,299]:
  for vi,v in enumerate(meta['views']):
   result=sampler.sample(vi,f,points,normals,classes,debug=True)
   if result is not None:records.append({'frame':f,'serial':v['serial'],'remaining_points_by_class_after_filter':result[3]})
(out/'report.json').write_text(json.dumps({'success':True,'sampled_face_centers_per_class':{str(c):int((classes==c).sum()) for c in [0,1,2]},'records':records,'scope':'Sequential actual candidate-label/self-depth/class/angle/hand/body filters on up to1000 face-center points per class. Filter order affects attributed removal counts; not causal source-error or texture accuracy proof. Diagnose observation support before relaxing any safety/visibility guard. No source photographs or matrices in this report.'},indent=2));print('texture support diagnosed',flush=True)
