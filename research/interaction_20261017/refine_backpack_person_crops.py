"""Reobserve all frames inside each view's full-action person-box envelope."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np,cv2
from PIL import Image,ImageDraw
from rtmlib import RTMPose
out=ROOT/'backpack_wholebody_all99_v2';out.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
data=ROOT/'backpack_native000_098_v2';prior=ROOT/'backpack_wholebody_all99_v1';meta=json.loads((data/'manifest.json').read_text());frames=meta['sample_frames'];det=np.load(prior/'detections.npz');assert det['frames'].tolist()==frames
model=RTMPose(str(ROOT/'wholebody_runtime_v1/rtmw.onnx'),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False)
samples=[0,15,30,50,75,98];sheet=Image.new('RGB',(6*384,8*312),'white');xy=[];scores=[];valid=[];boxes=[];views=[]
for vi,v in enumerate(meta['views']):
 serial=v['serial'];b=det['boxes_native'][vi];lo=b[:,:2].min(0);hi=b[:,2:].max(0);padding=(hi-lo)*.075;box=np.r_[np.maximum(lo-padding,0),np.minimum(hi+padding,[2047,1535])]
 validimage=np.asarray(Image.open(data/serial/'valid.png'))>127;vx=[];vs=[];vv=[]
 for f in frames:
  rgb=np.asarray(Image.open(data/serial/f'{f:04d}.png').convert('RGB'));k,s=model(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),bboxes=[box.tolist()]);points=k[0];score=s[0];pix=np.floor(points+.5).astype(int)
  inside=(pix[:,0]>=0)&(pix[:,0]<2048)&(pix[:,1]>=0)&(pix[:,1]<1536);inside&=validimage[pix[:,1].clip(0,1535),pix[:,0].clip(0,2047)];assert np.isfinite(points).all() and np.isfinite(score).all()
  vx.append(points);vs.append(score);vv.append(inside)
  if f in samples:
   im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle(box.tolist(),outline='yellow',width=4)
   for j,((x,y),q) in enumerate(zip(points,score)):
    if inside[j] and q>2:draw.ellipse((x-3,y-3,x+3,y+3),fill='lime' if j<91 else 'red')
   sheet.paste(im.resize((384,288)),(samples.index(f)*384,vi*312+24));ImageDraw.Draw(sheet).text((samples.index(f)*384+3,vi*312+4),f'{serial} f{f}',fill='black')
 xy.append(vx);scores.append(vs);valid.append(vv);boxes.append(np.repeat(box[None],99,0));views.append({'serial':serial,'native_box':box.tolist(),'source_frame_count':99})
 np.savez_compressed(out/f'{serial}.npz',xy=vx,score=vs,in_image=vv,frames=frames);(out/'status.json').write_text(json.dumps({'completed_views':vi+1,'seconds':time.time()-start},indent=2));print('backpack envelope observed',serial,flush=True)
np.savez_compressed(out/'detections.npz',xy=xy,score=scores,in_image=valid,boxes_native=boxes,frames=frames);sheet.save(out/'all8_identity_and_keypoints.jpg',quality=94)
(out/'report.json').write_text(json.dumps({'success':True,'observations':792,'frames':frames,'views':views,'seconds':time.time()-start,'scope':'Same independently selected backpack actor; per-view envelope of all99 tracked person boxes plus7.5percent padding replaces stale10-frame crop. Re-runs RTMW on native images, avoids upright head being outside an earlier bent-person box. Automatic detector envelopes and keypoints still require visual identity/coverage and geometric screening; not body/soft-bag reconstruction. Original images/calibration-bearing arrays remain private.'},indent=2));print('backpack corrected crops ready',flush=True)
