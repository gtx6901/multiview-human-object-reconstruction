"""Sequential real observation preparation for pickup/inspection, no publishing."""
import os
from pathlib import Path
import subprocess,sys,json,time,os
ROOT=Path(__file__).resolve().parent;out=ROOT/'pickup_observation_pipeline000_135_v1';out.mkdir(exist_ok=False)
os.environ['PYTHONDONTWRITEBYTECODE']='1';start=time.time();data='helmet_pickup_inspect000_135_v1';deadline=time.time()+180
while not (ROOT/data/'manifest.json').exists():
    if time.time()>deadline:raise RuntimeError('Source preparation is incomplete; inspect its log')
    time.sleep(1)
meta=json.loads((ROOT/data/'manifest.json').read_text());assert len(meta['views'])==8 and meta['sample_frames']==list(range(136))
for v in meta['views']:
    assert all((ROOT/data/v['serial']/f'{f:04d}.png').exists() for f in meta['sample_frames'])
steps=[
 ('detections',['detect_body.py','--data',data,'--out','detections_pickup000_135_v1','--seed-detections','body_detections_v2','--allow-missing']),
 ('nlf',['run_nlf_observations.py','--data',data,'--detections','detections_pickup000_135_v1','--out','nlf_pickup000_135_v1','--all-frames']),
 ('smplh',['fit_nlf_smplh.py','nlf_pickup000_135_v1','nlf_smplh_pickup000_135_v1']),
 ('world_audit',['audit_nlf_world_alignment.py','--fit','nlf_smplh_pickup000_135_v1','--data',data,'--detections','detections_pickup000_135_v1','--out','nlf_world_alignment_pickup000_135_v1'])]
report={'success':False,'steps':[],'scope':'Only continuous all8 human observations/initialization. No helmet tracking, human-object reconstruction completion or public upload.'}
for name,args in steps:
    tick=time.time();print('starting',name,flush=True)
    with (out/f'{name}.log').open('w') as log:result=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
    report['steps'].append({'name':name,'returncode':result.returncode,'seconds':time.time()-tick});(out/'status.json').write_text(json.dumps(report,indent=2))
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {out/name}.log')
    print('completed',name,flush=True)
report.update(success=True,seconds=time.time()-start);(out/'report.json').write_text(json.dumps(report,indent=2))
