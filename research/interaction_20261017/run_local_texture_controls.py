"""Same geometry/poses, compare global-frame vs local texture reliability."""
import os
from pathlib import Path
import subprocess,sys,json,os
ROOT=Path(__file__).resolve().parent;os.environ['PYTHONDONTWRITEBYTECODE']='1';out=ROOT/'local_texture_controls_pipeline_v1';out.mkdir(exist_ok=False)
steps=[
 ['export_shared_scene.py','--out','shared_scene_local_consensus_v1','--avatar','shared_shape_body_face_motion_v1','--motion','helmet_depth_aware_profile_pose_v2','--mesh','helmet_profile_local_consensus_texture_v1'],
 ['export_shared_scene.py','--out','shared_scene_shared_response_v1','--avatar','shared_shape_body_face_motion_v1','--motion','helmet_depth_aware_profile_pose_v2','--mesh','helmet_profile_shared_response_texture_v1'],
 ['evaluate_scene_input_views.py','--out','local_texture_controls_input_comparison_v1','--scenes','shared_scene_depth_aware_profile_v2','shared_scene_local_consensus_v1','shared_scene_shared_response_v1'],
 ['review_lhm_helmet_contacts.py','--out','helmet_profile_shared_response_contacts_v1','--avatar','shared_shape_body_face_motion_v1','--motion','helmet_depth_aware_profile_pose_v2','--mesh','helmet_profile_shared_response_texture_v1','--frames','42','180','299','--framing','shared_shape_body_face_object_contacts_v1']]
for i,args in enumerate(steps):
 print('starting',i,args[0],flush=True)
 with (out/f'{i}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,f'Inspect {i}.log; never repeat successful outputs'
 print('completed',i,flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'steps':steps,'scope':'Controlled candidate renders; same geometry/human/poses, different texture evidence selection or camera response only. No adoption or quality assertion.'},indent=2))
