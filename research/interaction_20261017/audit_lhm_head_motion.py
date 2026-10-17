"""Global head-rotation continuity before/after face fitting."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.spatial.transform import Rotation
from avatar_rig import pose_axis_angles
out=ROOT/'lhm_head_motion_audit_v1';out.mkdir(exist_ok=False);parent=torch.load(ROOT/'body_pose_v1/skeleton.pt',weights_only=True)['parents'].tolist();stats={};rows={}
for name in ['body_hands_all300_v1','body_head_faces_all300_v1']:
    p=torch.load(ROOT/name/'pose_sequence.pt',weights_only=True);aa=torch.cat([pose_axis_angles(p,t,'cpu') for t in range(300)]).numpy();R=Rotation.from_rotvec(aa.reshape(-1,3)).as_matrix().reshape(300,55,3,3);globalR=[]
    for j,pr in enumerate(parent):globalR.append(R[:,j] if pr<0 else globalR[pr]@R[:,j])
    head=Rotation.from_matrix(globalR[15]);step=np.rad2deg((head[:-1].inv()*head[1:]).magnitude());stats[name]={'median_deg':float(np.median(step)),'p95_deg':float(np.quantile(step,.95)),'max_deg':float(step.max()),'max_at_frame':int(step.argmax()+1)};rows[name]=step.tolist()
(out/'report.json').write_text(json.dumps({'success':True,'head_global_rotation_step_stats':stats,'step_series_deg':rows,'scope':'Motion continuity diagnostic only. Large changes require source-image audit; a smoother curve is not proof of accurate motion.'},indent=2));print(json.dumps(stats),flush=True)
