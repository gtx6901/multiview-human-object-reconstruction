"""Generated-only continuous before/after movies and fixed-camera hand crops."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--before',default='grounded_shared_shape_full300_v2');p.add_argument('--after',default='temporal_body_full300_v4');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False)
old=ROOT/o.before;new=ROOT/o.after
for folder in [old,new]:
 r=json.loads((folder/'report.json').read_text());assert r['success'] and r['frames']==list(range(300))
cams=json.loads((new/'cameras.json').read_text());assert cams==json.loads((old/'cameras.json').read_text());b=torch.load(ROOT/'underarm_weight_amount_fit_v3/avatar.pt',weights_only=True);ps=b['poses'];rows=[]
for vi,c in enumerate(cams['views'][8:]):
 W=np.array(c['W']);K=np.array(c['K']);world=ps['joints55_world'][180:,15].numpy();pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;uv=q[:,:2]/q[:,2:];center=np.median(uv,axis=0);radius=float(K[0,0]*ps['scale']/np.median(pc[:,2])*.29);box=(int(center[0]-radius),int(center[1]-radius),int(center[0]+radius),int(center[1]+radius))
 full=imageio.get_writer(str(out/f'novel{vi}_continuous_before_after.mp4'),fps=30,codec='libx264',macro_block_size=2);close=imageio.get_writer(str(out/f'novel{vi}_worn_hands_slow.mp4'),fps=10,codec='libx264',macro_block_size=2);sheet=Image.new('RGB',(768,8*408),'white')
 try:
  for f in range(300):
   pics=[Image.open(folder/f'{f:04d}_{c["label"]}.png').convert('RGB') for folder in [old,new]];pair=Image.new('RGB',(1134,780),'white')
   for k,im in enumerate(pics):pair.paste(im.resize((567,756)),(k*567,24))
   ImageDraw.Draw(pair).text((4,5),f'frame {f}: previous / neck-hands-sleeve candidate',fill='black');full.append_data(np.asarray(pair))
   if f>=180:
    crop=Image.new('RGB',(768,408),'white')
    for k,im in enumerate(pics):crop.paste(im.crop(box).resize((384,384)),(k*384,24))
    ImageDraw.Draw(crop).text((4,5),f'frame {f}: before / after; fixed crop, continuous original frames',fill='black');close.append_data(np.asarray(crop))
    if f in range(200,216,2):sheet.paste(crop,(0,((f-200)//2)*408))
 finally:full.close();close.close()
 sheet.save(out/f'novel{vi}_hands200_214_generated.jpg',quality=94)
 for name,n in [(f'novel{vi}_continuous_before_after.mp4',300),(f'novel{vi}_worn_hands_slow.mp4',120)]:
  reader=imageio.get_reader(str(out/name));count=sum(1 for _ in reader);reader.close();assert count==n;rows.append({'file':name,'decoded_frames':count})
(out/'report.json').write_text(json.dumps({'success':True,'before':o.before,'after':o.after,'videos':rows,'scope':'Pure-generated actual continuous-frame comparisons, identical camera matrices. Full0..299 before/after at30fps; worn180..299 fixed head/hand crop at10fps. Head crop fixed across the entire worn interval to avoid camera-induced jitter. No raw source imagery or camera matrices in public report.'},indent=2));print('Continuous before/after videos decoded and checked.',flush=True)
