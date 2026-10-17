"""Repair an unused inherited projection field without changing rendered state."""
import os
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from canonical_identity import semantic_identity
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]]);K=torch.tensor(np.array([A@np.array(v['K']) for v in meta['views']]),dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),dtype=torch.float32)
for name in ['shared_shape_body_face_motion_v1','shared_shape_body_face_zero_control_v1']:
 out=ROOT/name;bundle=torch.load(out/'avatar.pt',weights_only=True);p=bundle['poses'];source=torch.load(out/'pose_sequence.pt',weights_only=True);assert all(torch.equal(p[k],source[k]) for k in p)
 before=hashlib.sha256((out/'avatar.pt').read_bytes()).hexdigest();identity=semantic_identity(bundle);cam=torch.einsum('vij,tkj->vtki',W[:,:3,:3],p['joints_world'])+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,cam);derived=q[...,:2]/q[...,2:].clamp_min(.01);difference=float((derived-p['projected']).abs().max());fixed={**p,'projected':derived}
 assert all(torch.equal(p[k],fixed[k]) for k in p if k!='projected');updated={**bundle,'poses':fixed};assert semantic_identity(updated)==identity
 torch.save(updated,out/'avatar.pt');torch.save(fixed,out/'pose_sequence.pt');report={'success':True,'max_derived_projection_change_px1024':difference,'all_other_motion_tensors_exact':True,'canonical_tensors_exact':True,'avatar_file_sha256_before':before,'avatar_file_sha256_after':hashlib.sha256((out/'avatar.pt').read_bytes()).hexdigest(),'scope':'Only inherited poses.projected metadata recalculated from already-saved updated COCO3D joints and calibrated K/W. Actual render/fit inputs, canonical tensors and every other pose tensor unchanged; previous render/depth evidence remains applicable. Original parent result preserved. Fitter source now writes this derived field directly.'};(out/'derived_projection_repair.json').write_text(json.dumps(report,indent=2));print(name,difference,flush=True)
