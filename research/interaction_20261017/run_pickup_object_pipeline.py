import os
from pathlib import Path
import subprocess,sys,json,time,os
ROOT=Path(__file__).resolve().parent;out=ROOT/'pickup_object_pipeline000_135_v1';out.mkdir(exist_ok=False);os.environ['PYTHONDONTWRITEBYTECODE']='1';start=time.time()
steps=[
 ('track',['track_pickup_helmet.py','--out','helmet_track_pickup000_135_v2','--extra-anchors','pickup_extra_masks_v1']),
 ('rigid_fit',['fit_pickup_object.py']),
 ('colors',['bake_helmet_shortclip.py','helmet_colors_pickup000_135_v1','helmet_pose_pickup000_135_v1','--data','helmet_pickup_inspect000_135_v1','--objects','helmet_track_pickup000_135_v2','--frame','30']),
 ('avatar',['prepare_pickup_replay.py']),
 ('render',['render_dynamic_hybrid.py','dynamic_pickup000_135_v1','NONE','helmet_pose_pickup000_135_v1','helmet_colors_pickup000_135_v1','--avatar','avatar_motion_pickup000_135_v1','--data','helmet_pickup_inspect000_135_v1','--cameras','avatar_motion_pickup000_135_v1/cameras.json']),
 ('independent_replay',['render_dynamic_hybrid.py','dynamic_pickup000_135_replay_v1','dynamic_pickup000_135_v1','helmet_pose_pickup000_135_v1','helmet_colors_pickup000_135_v1','--avatar','avatar_motion_pickup000_135_v1','--data','helmet_pickup_inspect000_135_v1','--cameras','avatar_motion_pickup000_135_v1/cameras.json'])]
report={'success':False,'steps':[]}
for name,args in steps:
    print('starting',name,flush=True);tick=time.time()
    with (out/f'{name}.log').open('w') as log:result=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
    report['steps'].append({'name':name,'returncode':result.returncode,'seconds':time.time()-tick});(out/'status.json').write_text(json.dumps(report,indent=2))
    if result.returncode:raise RuntimeError(f'{name} failed; inspect its stage log')
    print('completed',name,flush=True)
report.update(success=True,seconds=time.time()-start,scope='Prototype execution and independent replay, not visual/geometry acceptance. Inspect actual renders before publication.');(out/'report.json').write_text(json.dumps(report,indent=2))
