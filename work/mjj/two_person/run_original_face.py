"""Exact original five-point face stage control, current-person inputs.
Only paths, thin staging, Unicode BOM, and NaN-safe masked arithmetic adapt.
"""
from pathlib import Path
import argparse,json,numpy as np,torch
import pipeline_adapter as pa
pa.configure_runtime_paths()
import run_face_stage as face
ap=argparse.ArgumentParser();ap.add_argument('--person',required=True);ap.add_argument('--avatar',required=True);ap.add_argument('--motion',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
person=Path(a.person).resolve();info=json.loads(person.read_text(encoding='utf8'));out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
stage=face.stage_inputs(person,Path(a.avatar).resolve(),Path(a.motion).resolve(),out)
replacements=[('        err = (uv - xy).norm(dim=-1)','        valid = w > 0\n        err = torch.zeros_like(uv[..., 0])\n        err[valid] = (uv[valid] - xy[valid]).norm(dim=-1)')]
mod,provenance=pa.load_reference_patched(face.REF_FACE,replacements)
mod.SRC=stage['stage'];mod.OUT=out;mod.OBS=Path(info['observations']);mod.KEY=Path(info['keypoints']);mod.MANIFEST=Path(info['manifest'])
torch.set_num_threads(4);mod.main()
provenance['scope']='Exact original5point/noisereg/lr/200steps, only NaN-safe mask arithmetic and current-person path adaptation; no rank gate. Not quality acceptance.'
(out/'reference_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
