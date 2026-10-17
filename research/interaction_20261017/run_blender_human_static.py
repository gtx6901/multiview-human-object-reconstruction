"""Use the verified portable Blender in a bounded background process."""
import os
from pathlib import Path
import subprocess,json,time
ROOT=Path(__file__).resolve().parent;out=ROOT/'blender_human_static_v1';out.mkdir(exist_ok=False);start=time.time()
runtime=json.loads((ROOT/'blender_runtime_v1/report.json').read_text());source=ROOT/'human_mesh_textured_v1';assert json.loads((source/'report.json').read_text())['success']
job={'out':str(out),'obj':str(source/'human_textured.obj'),'texture':str(source/'human_basecolor.png')};(out/'job.json').write_text(json.dumps(job,indent=2))
with (out/'blender.log').open('w',encoding='utf-8') as log:result=subprocess.run([runtime['executable'],'--background','--factory-startup','--python',str(ROOT/'blender_human_static.py'),'--',str(out/'job.json')],stdout=log,stderr=subprocess.STDOUT,timeout=600)
assert result.returncode==0 and (out/'report.json').exists(),f'Blender failed; inspect {out}/blender.log'
report=json.loads((out/'report.json').read_text());assert report['success'];print('Blender static candidate verified in',round(time.time()-start,2),'seconds',flush=True)
