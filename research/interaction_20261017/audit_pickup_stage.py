"""Frozen stage evidence, including actual source hashes and independent PNG replay."""
import os
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
out=ROOT/'pickup_stage_audit000_135_v1';out.mkdir(exist_ok=False);start=time.time()
def load(name):return json.loads((ROOT/name/'report.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
meta=json.loads((ROOT/'helmet_pickup_inspect000_135_v1/manifest.json').read_text());motion=torch.load(ROOT/'helmet_pose_pickup000_135_v1/sequence.pt',weights_only=True);frames=motion['frames'].tolist();assert frames==list(range(136))
source_checks=[]
for v in meta['views']:
    same=sha(Path(v['source']))==v['source_sha256'];assert same
    source_checks.append({'serial':v['serial'],'source_bytes_unchanged':same})
track=load('helmet_track_pickup000_135_v2');fit=load('helmet_pose_pickup000_135_v1');body=load('body_fused_pickup000_135_v1');render=load('dynamic_pickup000_135_v1');replay=load('dynamic_pickup000_135_replay_v1')
assert render['success'] and replay['success'];assert len(replay['replay_checks'])==136*2+8*3 and all(c['max_u8_difference']==0 for c in replay['replay_checks'])
neutral=load('dynamic_pickup_neutral000_135_v1');neutral_replay=load('dynamic_pickup_neutral000_135_replay_v1');assert neutral_replay['success'] and len(neutral_replay['replay_checks'])==296 and all(c['max_u8_difference']==0 for c in neutral_replay['replay_checks'])
for r in [render,neutral,neutral_replay]:assert all(x['decoded_frames']==136 for x in r['videos'])
R=motion['R'].numpy();relative=np.einsum('tij,tkj->tik',R[1:],R[:-1]);angle=np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(),axis=-1)*180/np.pi;translation=np.linalg.norm(np.diff(motion['translation_world'].numpy(),axis=0),axis=-1)
observed=np.array([sum(x['frame']==f and x['mask'] is not None for x in track['records']) for f in frames]);ious=[v for m in fit['metrics'] for v in m['soft_visible_iou']]
nlf_count=len(load('nlf_pickup000_135_v1')['records']);assert nlf_count==1088
report={'success':True,'frames':frames,'source_cameras':8,'observations':1088,'nlf_predictions':nlf_count,'body_mean_input_reprojection_px1024':body['mean_body_error_px1024'],'object_observed_masks':track['observed_masks'],'object_unknown_masks':track['unknown_masks'],'object_observed_views_per_frame_min':int(observed.min()),'object_observed_views_per_frame_median':float(np.median(observed)),'object_temporal_carry_frames':fit['temporal_carry_frames'],'object_mean_input_soft_visible_iou':float(np.mean(ious)),'object_rotation_step_max_degrees':float(angle.max()),'object_rotation_step_max_frame':int(angle.argmax()+1),'object_translation_step_max_body_height_units':float(translation.max()),'color_observed_vertex_fraction':load('helmet_colors_pickup000_135_v1')['observed_vertex_fraction'],'independent_exact_pngs_per_variant':len(replay['replay_checks']),'video_frames':136,'video_fps':30,'slow_video_fps':10,'video_duration_seconds':136/30,'slow_video_duration_seconds':136/10,'source_integrity_checks':source_checks,'seconds':time.time()-start,'scope':'Execution/integrity/input-fit evidence only. Manual render review required; no unseen-view ground truth, measured metric accuracy, physical contact, cavity/texture acceptance or full300-frame completion.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
