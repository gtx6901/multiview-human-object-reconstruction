"""Explicit display pose prior for independently inferred backpack actor asset."""
import os
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parent;out=ROOT/'backpack_canonical_pose_v1';out.mkdir(exist_ok=False)
body=torch.zeros(1,21,3);body[0,15,2]=-.65;body[0,16,2]=.65
torch.save({'frames':torch.tensor([0]),'root_pose':torch.zeros(1,3),'body_pose':body,'left_hand_pose':torch.zeros(1,15,3),'right_hand_pose':torch.zeros(1,15,3),'translation_world':torch.zeros(1,3),'betas':torch.zeros(1,10),'scale':torch.tensor(1.)},out/'pose_sequence.pt')
(out/'report.json').write_text(json.dumps({'success':True,'frames':1,'scope':'A-pose display initialization only; beta0/flat hands/unit model scale are priors, not observed backpack motion or physical scale. No helmet-actor motion copied. Actual sequence fitting waits for reliable eight-view geometry.'},indent=2));print('independent canonical display pose prepared')
