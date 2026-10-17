"""Real per-frame person masks, identity gated by observed target boxes."""
import os
from pathlib import Path
import json,hashlib,time,argparse
import torch,numpy as np
from PIL import Image,ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
p=argparse.ArgumentParser();p.add_argument('--data',default='helmet_contiguous180_211_v1');p.add_argument('--detections',default='detections_contiguous180_211_v2');p.add_argument('--objects',default='helmet_track180_211_v2');p.add_argument('--out',default='people_contiguous180_211_v1');options=p.parse_args()
ROOT=Path(__file__).resolve().parent;data=ROOT/options.data;meta=json.loads((data/'manifest.json').read_text());det=np.load(ROOT/options.detections/'detections.npz')
obj=ROOT/options.objects;objects={(p['serial'],p['frame']):p for p in json.loads((obj/'report.json').read_text())['records']} if options.objects!='NONE' else {}
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
weights=Path.home()/'.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
model=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None).cuda().eval();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
records=[]
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  view_out=out/v['serial'];view_out.mkdir();sheet=Image.new('RGB',(8*192,((len(meta['sample_frames'])+7)//8)*164),'white')
  for ti,f in enumerate(meta['sample_frames']):
   source=data/v['serial']/f'{f:04d}.png';arr=np.asarray(Image.open(source).convert('RGB')).copy();pred=model([torch.from_numpy(arr.transpose(2,0,1)).cuda().float()/255])[0]
   ids=torch.where((pred['labels']==1)&(pred['scores']>.5))[0];expected=det['boxes'][vi,ti]
   b=pred['boxes'][ids].cpu().numpy();inter=np.maximum(0,np.minimum(b[:,2:],expected[2:])-np.maximum(b[:,:2],expected[:2])).prod(1);area=np.maximum(0,b[:,2:]-b[:,:2]).prod(1);iou=inter/(area+np.prod(expected[2:]-expected[:2])-inter+1e-8)
   valid=iou>.1;row={'serial':v['serial'],'frame':f,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'mask':None,'status':'unknown; no confident identity-matched person mask'};overlay=arr.copy()
   if valid.any():
    rank=iou*pred['scores'][ids].cpu().numpy();selected=int(ids[int(np.argmax(np.where(valid,rank,-1)))])
    person=pred['masks'][selected,0].cpu().numpy()>.5;Image.fromarray(person.astype('uint8')*255).save(view_out/f'{f:04d}_raw.png')
    object_record=objects.get((v['serial'],f),{'mask':None});removed=0
    if object_record['mask'] is not None:
     object_mask=np.asarray(Image.open(obj/object_record['mask']))>127;removed=int((person&object_mask).sum());person&=~object_mask
    Image.fromarray(person.astype('uint8')*255).save(view_out/f'{f:04d}.png');overlay[person]=(overlay[person]*.5+np.array([30,200,30])*.5).astype('uint8')
    row.update(mask=f"{v['serial']}/{f:04d}.png",raw_mask=f"{v['serial']}/{f:04d}_raw.png",pixels=int(person.sum()),selected_score=float(pred['scores'][selected]),object_overlap_removed=removed,status='candidate person mask; object-positive pixels excluded when available')
   pic=Image.fromarray(overlay);sheet.paste(pic.resize((192,144)),(ti%8*192,ti//8*164+20));ImageDraw.Draw(sheet).text((ti%8*192,ti//8*164),str(f),fill='black');records.append(row)
  sheet.save(out/f"{v['serial']}_timeline.jpg");print(v['serial'],'person masks',flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'views':8,'frames':meta['sample_frames'],'records':records,'unknown':sum(p['mask'] is None for p in records),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'seconds':time.time()-start,'scope':'Actual Mask R-CNN per-image person candidates, selected by target box overlap. Raw masks retained. Object candidates can have errors, so subtraction is not ground truth; unknown object masks not treated as empty. No 3D geometry accuracy established.'},indent=2))
