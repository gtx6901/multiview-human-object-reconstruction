"""Negative controls for single-camera dominance and fabricated coverage."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import torch
from texture_consensus import consistent_texel_weights
# Four independent cases: high-weight black outlier, unsupported black,
# contradictory two cameras, and independently corroborated black material.
w=torch.zeros(4,2,4);c=torch.zeros(4,2,4,3)
w[:,0,0]=torch.tensor([100.,1.,1.,1.]);c[:,0,0]=torch.tensor([[.02]*3,[.8]*3,[.82]*3,[.81]*3])
w[0,0,1]=1;c[0,0,1]=.02
w[:2,0,2]=1;c[0,0,2]=.02;c[1,0,2]=.8
w[:2,0,3]=1;c[:2,0,3]=.02
filtered,valid,report=consistent_texel_weights(w,c)
assert valid.tolist()==[True,False,False,True]
assert filtered[0,:,0].sum()==0 and filtered[1:,:,0].sum()>0
assert filtered[:,:,1:3].sum()==0 and filtered[:,:,3].sum()==2
out=ROOT/'texture_consensus_negative_controls_v1';out.mkdir(exist_ok=False);report={'success':True,'high_weight_single_view_outlier_rejected':True,'single_camera_and_conflicting_pair_remain_unknown':True,'corroborated_dark_material_preserved':True,'scope':'Synthetic failure controls only. Does not establish real-data correspondence, appearance quality or recovered albedo.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
