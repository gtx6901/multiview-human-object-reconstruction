"""Estimate one common camera-space unit conversion from actual all8 NLF geometry."""
import os
import argparse,json,time
from cari_runtime import configure,ASSETS,ROOT
configure()
import torch,numpy as np
from lib_smpl.body_landmark import BodyLandmarks
p=argparse.ArgumentParser();p.add_argument('--fit',default='nlf_smplh_frame30_v3');p.add_argument('--data',default='helmet_all8_v2');p.add_argument('--detections',default='body_detections_v2');p.add_argument('--out',default='nlf_world_alignment_frame30_v1');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/options.data/'manifest.json').read_text());det=np.load(ROOT/options.detections/'detections.npz');frames_det=det['frames'].tolist()
frames=sorted({int(path.stem.rsplit('_',1)[1]) for path in (ROOT/options.fit).glob('*.pt')});assert frames
landmark=BodyLandmarks(str(ASSETS));body25=[5,2,6,3,7,4,12,9,13,10,14,11]
qs=[];centers=[];valid=[]
for i,v in enumerate(meta['views']):
    world_to_cam=np.array(v['W']);R=world_to_cam[:3,:3];t=world_to_cam[:3,3]
    verts=np.stack([torch.load(ROOT/options.fit/f"{v['serial']}_{f:04d}.pt",weights_only=True)['vertices'][0].numpy() for f in frames])
    joints=landmark.get_body_kpts_batch(verts)[:,body25]
    qs.append(joints@R);centers.append(-R.T@t)
    score=det['score'][i,[frames_det.index(f) for f in frames],5:]
    valid.append((score>3).astype(float))
q=np.array(qs);center=np.array(centers)[:,None,None];weights=np.array(valid)[...,None];assert (weights.sum(0)>=2).all()
qmean=(q*weights).sum(0,keepdims=True)/weights.sum(0,keepdims=True);cmean=(center*weights).sum(0,keepdims=True)/weights.sum(0,keepdims=True)
dq=q-qmean;dc=center-cmean
alpha=float(-(weights*dq*dc).sum()/(weights*dq*dq).sum());assert alpha>0
world=alpha*q+center;fused=(world*weights).sum(0)/weights.sum(0);residual=np.linalg.norm(world-fused[None],axis=-1);mask=weights[...,0]>0
np.savez_compressed(out/'aligned_joints.npz',frames=frames,per_view_world_joints=world,fused_diagnostic_joints=fused,valid=mask,camera_centers=center,common_model_to_world_scale=alpha)
report={'success':True,'views':8,'frames':frames,'common_model_to_world_scale':alpha,'cross_view_mean_dispersion_world_units':float(residual[mask].mean()),'cross_view_median_dispersion_world_units':float(np.median(residual[mask])),'per_view_mean_dispersion_world_units':[float(residual[i][mask[i]].mean()) for i in range(8)],'seconds':time.time()-start,'scope':'One global scalar estimated with fixed real camera rotations/translations, all8 real NLF fitted body joints. No per-camera normalization or camera refinement. Dispersion is consistency between predictions, NOT metric reconstruction accuracy. World scale inherited subject-height normalization; model metres are prior-implied. Fused averages are diagnostics, not an articulated reconstructed human.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
