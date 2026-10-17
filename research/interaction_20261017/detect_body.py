"""Eight-view keypoint observations; keep candidates and identity-selection evidence."""
import os
from pathlib import Path
import json,sys,time,hashlib,argparse
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from torchvision.models.detection import keypointrcnn_resnet50_fpn
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='helmet_all8_v2');p.add_argument('--out',default='body_detections_v2');p.add_argument('--allow-missing',action='store_true');p.add_argument('--seed-detections');options=p.parse_args()
    data=ROOT/options.data;meta=json.loads((data/'manifest.json').read_text())
    out=ROOT/options.out;out.mkdir(exist_ok=False)
    weights=Path.home()/'.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
    model=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None).cuda().eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    static=ROOT.parent/'rebuild_20261014/datasets/fcs_frame0150'
    seed=np.load(ROOT/options.seed_detections/'detections.npz') if options.seed_detections else None
    if seed is not None:
        seed_records=json.loads((ROOT/options.seed_detections/'report.json').read_text())['records']
        assert list(dict.fromkeys(r['view'] for r in seed_records))==[v['serial'] for v in meta['views']]
    xy=[];scores=[];boxes=[];records=[];start=time.time()
    with torch.inference_mode():
        for vi,v in enumerate(meta['views']):
            mask=np.array(Image.open(static/'masks'/f"{v['id']:02d}.png"))
            yy,xx=np.where(mask>127);prev=np.array([xx.min(),yy.min(),xx.max(),yy.max()])*1024/mask.shape[1]
            if seed is not None:
                nearest=int(np.argmin(np.abs(seed['frames']-meta['sample_frames'][0])));prev=seed['boxes'][vi,nearest].copy()
            vx=[];vs=[];vb=[]
            for frame in meta['sample_frames']:
                arr=np.asarray(Image.open(data/v['serial']/f'{frame:04d}.png').convert('RGB')).copy()
                pred=model([torch.from_numpy(arr.transpose(2,0,1)).float().cuda()/255])[0]
                b=pred['boxes'].cpu().numpy();s=pred['scores'].cpu().numpy()
                inter=np.maximum(0,np.minimum(b[:,2:],prev[2:])-np.maximum(b[:,:2],prev[:2])).prod(1)
                area=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
                iou=inter/(area+np.prod(prev[2:]-prev[:2])-inter+1e-8)
                centre=np.linalg.norm((b[:,:2]+b[:,2:])/2-(prev[:2]+prev[2:])/2,axis=1)
                rank=s*(.2+iou)*np.exp(-centre/300)
                selected=int(np.argmax(rank));missing=False
                if options.allow_missing:
                    eligible=(s>.5)&((iou>.1)|(centre<max(100,np.linalg.norm(prev[2:]-prev[:2])*.6)))
                    if eligible.any():selected=int(np.argmax(np.where(eligible,rank,-1)))
                    else:missing=True
                else:assert s[selected]>.5
                if missing:
                    kp=np.zeros((17,2),dtype='float32');ks=np.zeros(17,dtype='float32')
                    selected_box=prev.copy()
                else:
                    kp=pred['keypoints'][selected,:,:2].cpu().numpy();ks=pred['keypoints_scores'][selected].cpu().numpy();selected_box=b[selected];prev=selected_box
                vx.append(kp);vs.append(ks);vb.append(selected_box)
                records.append({'view':v['serial'],'frame':frame,'selected':selected,'boxes':b.tolist(),
                                'person_scores':s.tolist(),'selection_scores':rank.tolist(),'missing_keypoints':missing,'box_source':'last reliable same-view box; crop initialization only' if missing else 'actual current-frame detection'})
                pic=Image.fromarray(arr);draw=ImageDraw.Draw(pic)
                draw.rectangle(tuple(prev),outline='yellow',width=2)
                for j,((x,y),q) in enumerate(zip(kp,ks)):
                    c='lime' if q>3 else 'gray';draw.ellipse((x-3,y-3,x+3,y+3),fill=c);draw.text((x+3,y),str(j),fill=c)
                pic.resize((512,384)).save(out/f"{v['serial']}_{frame:04d}.jpg")
            xy.append(vx);scores.append(vs);boxes.append(vb);print(v['serial'],'detected',flush=True)
    np.savez_compressed(out/'detections.npz',xy=xy,score=scores,boxes=boxes,frames=meta['sample_frames'])
    (out/'report.json').write_text(json.dumps({'seconds':time.time()-start,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'score_threshold':3,'selection':'Confidence weighted overlap and center continuity from static subject bounding box. Must visually audit identity; not ground truth.',
        'records':records},indent=2))
    sheet=Image.new('RGB',(len(meta['sample_frames'])*256,8*192),'white')
    for i,v in enumerate(meta['views']):
        for j,f in enumerate(meta['sample_frames']):sheet.paste(Image.open(out/f"{v['serial']}_{f:04d}.jpg").resize((256,192)),(j*256,i*192))
    sheet.save(out/'all8_keypoints.jpg')
if __name__=='__main__':main()
