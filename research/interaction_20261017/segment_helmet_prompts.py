"""Segment explicit source-image prompts; unknown observations produce no mask."""
import os
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('prompts');p.add_argument('out');p.add_argument('--data',default='helmet_all8_v2');p.add_argument('--crop-contact',action='store_true');p.add_argument('--allow-empty-unknown',action='store_true');options=p.parse_args()
sys.path[:0]=[str(ROOT.parent/'lhmpp_20261016/deps'),str(ROOT/'vendor/MobileSAM')]
import torch,numpy as np,cv2
torch.set_num_threads(4)
from PIL import Image,ImageDraw
cfg=json.loads((ROOT/options.prompts).read_text());out=ROOT/options.out;out.mkdir(exist_ok=False)
weights=ROOT/'vendor/MobileSAM/weights/mobile_sam.pt'
from mobile_sam import sam_model_registry,SamPredictor
model=sam_model_registry['vit_t'](checkpoint=str(weights)).cuda().eval();predictor=SamPredictor(model)
records=[];sheet=Image.new('RGB',(3*384,((len(cfg['records'])+2)//3)*304),'white')
for i,prompt in enumerate(cfg['records']):
    serial=prompt['serial'];f=prompt['frame'];source=ROOT/options.data/serial/f'{f:04d}.png'
    arr=np.asarray(Image.open(source).convert('RGB'));record=dict(prompt,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    pic=Image.fromarray(arr.copy())
    if prompt['status']=='unknown':
        record['mask']=None
    else:
        coords=np.array(prompt['pos']+prompt.get('neg',[]));labels=np.array([1]*len(prompt['pos'])+[0]*len(prompt.get('neg',[])))
        assert ((coords[:,0]>=0)&(coords[:,0]<1024)&(coords[:,1]>=0)&(coords[:,1]<768)).all()
        predictor.set_image(arr);masks,scores,_=predictor.predict(point_coords=coords,point_labels=labels,box=np.array(prompt['box']),multimask_output=True)
        choice=int(np.argmax(scores));mask=masks[choice]
        _,components=cv2.connectedComponents(mask.astype('uint8'));ids={int(components[int(y),int(x)]) for x,y in prompt['pos']}-{0};mask=np.isin(components,list(ids))
        if not mask.any() and options.allow_empty_unknown:
            record.update(mask=None,status='unknown; no component contains positive prompt')
            pic.save(out/f'{serial}_{f:04d}_overlay.jpg');records.append(record);print(serial,f,'unknown empty candidate',flush=True);continue
        assert mask.any(),(serial,f)
        name=f'{serial}_{f:04d}_mask.png';Image.fromarray(mask.astype('uint8')*255).save(out/name)
        overlay=arr.copy();overlay[mask]=(overlay[mask]*.45+np.array([255,40,30])*.55).astype('uint8');pic=Image.fromarray(overlay)
        dr=ImageDraw.Draw(pic);dr.rectangle(prompt['box'],outline='lime',width=2)
        for (x,y),label in zip(coords,labels):dr.ellipse((x-3,y-3,x+3,y+3),fill='lime' if label else 'blue')
        record.update(mask=name,pixels=int(mask.sum()),candidate_scores=scores.tolist(),chosen=choice,status='candidate needs visual QA',truncated_by_image_boundary=bool(mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any()))
    pic.save(out/f'{serial}_{f:04d}_overlay.jpg')
    if options.crop_contact and prompt['status']!='unknown':
        x0,y0,x1,y1=prompt['box'];cx=(x0+x1)/2;cy=(y0+y1)/2;side=max(x1-x0,y1-y0)+60;pic=pic.crop((int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2)))
    sheet.paste(pic.resize((384,288)),(i%3*384,i//3*304+16));ImageDraw.Draw(sheet).text((i%3*384,i//3*304),f'{serial} f{f} '+prompt['status'],fill='black')
    records.append(record);print(serial,f,record.get('pixels'),flush=True)
sheet.save(out/'contact.jpg')
(out/'report.json').write_text(json.dumps({'prompts':options.prompts,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'records':records,'scope':'Real per-image visible helmet mask candidates only; unknown is no observation. No ground-truth accuracy or temporal 3D reconstruction established.'},indent=2))
