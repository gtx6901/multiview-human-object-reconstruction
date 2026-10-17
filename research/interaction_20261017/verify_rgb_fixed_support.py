"""Real raster negative case: hiding the object must not yield zero RGB loss."""
import os
from pathlib import Path
import sys,runpy,json
ROOT=Path(__file__).resolve().parent
sys.argv=[str(ROOT/'fit_phased_helmet_all300.py'),'--out','helmet_rgb_support_negative_probe_v3','--initial','helmet_inspection_shell_fit_all300_v1','--phase-motion','helmet_inspection_shell_fit_all300_v1','--phase-priors','helmet_phase_head_faces_all300_v1','--body-depth','lhm_occlusion_all300_v1','--rgb-loss-weight','2','--rgb-texture','helmet_inspection_texture_surface_v3','--skip-coarse','--crop-observations','--worn-rotation-cap-deg','3','--worn-translation-cap','.003','--observation-frames','60','180','299','--steps','1']
s=runpy.run_path(str(ROOT/'fit_phased_helmet_all300.py'));torch=s['torch']
with torch.no_grad():
 R,t,_,_=s['trajectory']();_,_,_,photo=s['cost'](R,t+torch.tensor([1000.,0,0],device='cuda'),s['allids']);valid=(s['rgb_reference_support'].sum((1,2))>0).cuda();assert valid.any() and torch.isfinite(photo).all()
 assert torch.allclose(photo[valid],torch.full_like(photo[valid],.25),atol=1e-6)
 report={'success':True,'supported_observations':int(valid.sum()),'reference_pixels':int(s['rgb_reference_support'].sum()),'offscreen_photo_mean':float(photo[valid].mean()),'scope':'Actual selected-frame CUDA textured raster negative case after frozen support capture. Translating object1000world numerical units offscreen produces missing-support penalty.25 rather than zero RGB error. This verifies loss loophole correction only; no all-action pose or visual-quality claim.'}
(ROOT/'helmet_rgb_support_negative_probe_v3/negative_case.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
