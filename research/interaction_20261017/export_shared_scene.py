"""Package a verified initializer into shared assets and per-frame states."""
import os
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from shared_scene import load_shared_scene
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='avatar_multitime16_all300_v1');p.add_argument('--motion',default='helmet_lhm_depth_tight_all300_v1');p.add_argument('--mesh',default='helmet_texture_consistent_lhm_v1');p.add_argument('--cameras',default='lhm_portrait_framing_v1/cameras.json');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);avatar=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);m=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);mesh=torch.load(ROOT/o.mesh/'mesh.pt',weights_only=True);poses=avatar['poses'];assert torch.equal(poses['frames'],m['frames'])
assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],m['vertices_local'][m['faces'].long()],atol=1e-6)
model=SMPLXSurfaceLandmarks();beta=poses['betas'];expected_rest=model.rest+torch.einsum('bl,jcl->bjc',beta,model.joint_shapedirs)[0];rig_rest=avatar['rig']['rest_joints'].reshape(55,3);error=float((rig_rest-expected_rest).abs().max());assert error<1e-6,'Shared beta is inconsistent with the actual LHM rest rig; propagate shape before exporting'
human={k:avatar[k] for k in ['rig','features','background_embedding','decoder','decoder_config']};human.update(betas=poses['betas'],scale=poses['scale'])
fields=['root_pose','body_pose','translation_world','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose'];states={k:poses[k] for k in fields if k in poses};assert states['body_pose'].shape[0]==len(poses['frames'])
helmet={**mesh,'logscale':m['logscale']};motion={'frames':poses['frames'],'human':states,'helmet':{'R':m['R'],'translation_world':m['translation_world']}}
torch.save(human,out/'human_canonical.pt');torch.save(helmet,out/'helmet_canonical.pt');torch.save(motion,out/'motion.pt');(out/'cameras.json').write_text((ROOT/o.cameras).read_text())
manifest={'format':'shared-hoi-scene-v1','frames':len(poses['frames']),'shared_models':['human_canonical.pt','helmet_canonical.pt'],'per_frame_state':'motion.pt','sources':vars(o),'assets':[{'path':name,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()} for name in ['human_canonical.pt','helmet_canonical.pt','motion.pt','cameras.json']],'scope':'Explicit shared canonical human appearance/rig/beta/scale and helmet geometry/UV/texture, separate human and helmet motion table. A container/initializer only: no new joint fitting, all-image refinement or quality gain. Rest rig/beta consistency verified; accepted original assets unchanged. Licensed human tensors remain LOCAL and must not enter public repository.'}
(out/'scene.json').write_text(json.dumps(manifest,indent=2));loaded=load_shared_scene(out)
for k in states:assert torch.equal(loaded['avatar']['poses'][k],poses[k])
for k in ['betas','scale']:assert torch.equal(loaded['avatar']['poses'][k],poses[k])
for k in ['R','translation_world']:assert torch.equal(loaded['object_motion'][k],m[k])
(out/'report.json').write_text(json.dumps({'success':True,'frames':len(poses['frames']),'canonical_models':2,'human_motion_fields':list(states),'rig_beta_rest_error':error,'motion_tensor_reconstruction_exact':True,'scope':manifest['scope']},indent=2));print('shared scene initializer exported; no optimization claim',flush=True)
