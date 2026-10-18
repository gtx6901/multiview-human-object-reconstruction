"""Compare all images on the coherent shaped scene before bounded appearance fit."""
import os
from pathlib import Path
import subprocess,sys,json,time,os
ROOT=Path(__file__).resolve().parent;out=ROOT/'shaped_appearance_pipeline_v1';out.mkdir(exist_ok=False);start=time.time();os.environ['PYTHONDONTWRITEBYTECODE']='1';rows=[]
steps=[('observations',['prepare_shared_human_observations.py','--out','shaped_human_observations_all300_v1','--scene','shared_scene_shaped_body_face_v1'],'shaped_human_observations_all300_v1'),('features',['refine_shared_human_features.py','--out','shaped_human_linear_features_bound005_v1','--scene','shared_scene_shaped_body_face_v1','--observations','shaped_human_observations_all300_v1','--response','linear-reference','--feature-bound','.05'],'shaped_human_linear_features_bound005_v1')]
for name,args,folder in steps:
 print('starting',name,flush=True);tick=time.time()
 with (out/f'{name}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
 p=ROOT/folder/'report.json';okay=r.returncode==0 and p.exists() and json.loads(p.read_text()).get('success',False);rows.append({'stage':name,'success':okay,'returncode':r.returncode,'seconds':time.time()-tick});(out/'status.json').write_text(json.dumps(rows,indent=2));assert okay,f'Inspect {name}.log without repeating successful stages'
 print('completed',name,flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'stages':rows,'seconds':time.time()-start,'scope':'All2400-image fixed-support comparison followed by bounded canonical-feature-only refinement on the coherent nonzero-beta scene initialized from original accepted appearance. Frozen motion, rig and decoder; no automatic visual adoption.'},indent=2))
