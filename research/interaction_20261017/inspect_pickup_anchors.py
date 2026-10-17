"""Image crops for manual visible-object anchors; projection is only a proposal."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from PIL import Image,ImageDraw
data=ROOT/'helmet_pickup_inspect000_135_v1';meta=json.loads((data/'manifest.json').read_text());registry=json.loads((ROOT/'helmet_observations_v1/manifest.json').read_text())['records']
out=ROOT/'pickup_anchor_inspection_v1';out.mkdir(exist_ok=False);centers={}
for f in [30,60,90,120,150]:
    matrices=[];right=[]
    for v in meta['views'][:2]:
        rec=next(p for p in registry if p['serial']==v['serial'] and p['frame']==f);mask=np.asarray(Image.open(ROOT/rec['mask']))>127;yy,xx=np.where(mask)
        K=np.array(v['K']);W=np.array(v['W']);ray=W[:3,:3].T@np.linalg.solve(K,[xx.mean(),yy.mean(),1]);ray/=np.linalg.norm(ray);C=-W[:3,:3].T@W[:3,3];P=np.eye(3)-np.outer(ray,ray);matrices.append(P);right.append(P@C)
    centers[f]=np.linalg.solve(np.sum(matrices,0),np.sum(right,0))
old=torch.load(ROOT/'body_pose_v1/pose_sequence.pt',weights_only=True);wrist=old['joints_world'][0,[4,5]].mean(0).numpy()
v=meta['views'][1];K=np.array(v['K']);W=np.array(v['W']);rec=next(p for p in registry if p['serial']==v['serial'] and p['frame']==0);m=np.asarray(Image.open(ROOT/rec['mask']))>127;yy,xx=np.where(m);C=-W[:3,:3].T@W[:3,3];ray=W[:3,:3].T@np.linalg.solve(K,[xx.mean(),yy.mean(),1]);ray/=np.linalg.norm(ray);centers[0]=C+ray*np.dot(wrist-C,ray)
times=sorted(centers);trajectory=np.array([[np.interp(f,times,[centers[t][d] for t in times]) for d in range(3)] for f in range(136)])
np.savez(out/'proposal_centers.npz',frames=np.arange(136),world_centers=trajectory)
frames=[0,30,60,90,120,135];sheet=Image.new('RGB',(6*256,8*280),'white');records=[]
for vi,v in enumerate(meta['views']):
    for j,f in enumerate(frames):
        p=np.array(v['W'])[:3,:3]@trajectory[f]+np.array(v['W'])[:3,3];uv=np.array(v['K'])@p;uv=uv[:2]/uv[2];x,y=uv;box=[int(round(x-128)),int(round(y-128)),int(round(x+128)),int(round(y+128))]
        pic=Image.open(data/v['serial']/f'{f:04d}.png').crop(box);sheet.paste(pic,(j*256,vi*280+24));ImageDraw.Draw(sheet).text((j*256+3,vi*280+5),f'{v["serial"]} f{f} ({box[0]},{box[1]})',fill='black');records.append({'serial':v['serial'],'frame':f,'crop_xyxy':box,'projected_proposal':uv.tolist()})
sheet.save(out/'all8_keyframe_crops.jpg');(out/'crops.json').write_text(json.dumps(records,indent=2))
