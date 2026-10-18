"""One selected-pixel body feature update followed by the SAME integrated review."""
import os
from pathlib import Path
import os,sys,json,time,subprocess
ROOT=Path(__file__).resolve().parent;os.environ['PYTHONDONTWRITEBYTECODE']='1'
out=ROOT/'aligned_body_features_loop_v1';out.mkdir(exist_ok=False)
observations='overhead_body_aligned_observations_all300_v2';source=json.loads((ROOT/observations/'report.json').read_text());assert source['success'] and source['alignment_screen'] and len(source['frames'])==300 and source['observations_compared']==2400
frames=[0,30,42,60,90,120,135,145,155,165,175,180,190,200,210,225,240,260,280,299]
steps=[('refine',['refine_shared_human_features.py','--out','overhead_body_aligned_features_v1','--scene','shared_scene_overhead_body_v2','--observations',observations,'--response','linear-reference','--feature-bound','.05','--freeze-head-hands','--audit-frames',*map(str,frames)]),('render',['render_dynamic_hybrid.py','aligned_body_features_after_review20_v1','NONE','--scene','overhead_body_aligned_features_v1/scene','--data','helmet_native000_299_v1','--occlusion','gaussian-center','--all-input-frames','--frames',*map(str,frames)]),('storyboard',['make_integrated_review_storyboard.py','--out','aligned_body_features_comparison20_v1','--before','overhead_body_appearance_after_review20_v2','--after','aligned_body_features_after_review20_v1','--avatar','grounded_legs_all300_v1'])]
rows=[];start=time.time()
for name,args in steps:
 print('starting',name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
 rows.append({'stage':name,'returncode':r.returncode});(out/'status.json').write_text(json.dumps(rows,indent=2));assert r.returncode==0,f'Inspect {name}.log; do not repeat completed stages'
 print('completed',name,flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'training_frames':300,'training_cameras':8,'review_frames':frames,'stages':rows,'seconds':time.time()-start,'scope':'One shared body feature update from locally alignment-screened nonclipped body pixels across all8x300 observations. Shape/rig/geometry/motion/decoder and head/hand-dominant feature rows fixed; feature bound.05.20phase times used for response calibration and loss audit, independently replayed all8+2 for actual visual comparison. Source-bearing sheets and all tensors remain private. Not geometry refinement or full300 new video; quality adoption requires visual review.'},indent=2))
