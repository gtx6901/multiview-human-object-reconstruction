"""Independent backpack actor observations; never inherit helmet actor identity."""
import os
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from rtmlib import RTMPose
from PIL import Image,ImageDraw
out=ROOT/'backpack_wholebody_all99_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
data=ROOT/'backpack_native000_098_v2';meta=json.loads((data/'manifest.json').read_text());frames=meta['sample_frames'];assert frames==list(range(99)) and len(meta['views'])==8
weight=Path.home()/'.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth';detector=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None).eval();detector.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
pose=RTMPose(str(ROOT/'wholebody_runtime_v1/rtmw.onnx'),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False)
samples=[0,15,30,50,75,98];sheet=Image.new('RGB',(6*384,8*312),'white');allxy=[];allscores=[];allvalid=[];allboxes=[];records=[]
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  serial=v['serial'];prev=None;xy=[];scores=[];valids=[];boxes=[]
  validimage=np.asarray(Image.open(data/serial/'valid.png'))>127
  for f in frames:
   rgb=np.asarray(Image.open(data/serial/f'{f:04d}.png').convert('RGB'));assert rgb.shape==(1536,2048,3)
   candidates=None;selection=None
   if f%10==0 or prev is None:
    small=cv2.resize(rgb,(1024,768),interpolation=cv2.INTER_AREA).copy();pred=detector([torch.from_numpy(small).permute(2,0,1).float()/255])[0]
    b=(pred['boxes'].numpy()+.5)*2-.5;s=pred['scores'].numpy();keep=s>.6;assert keep.any(),f'No confident person {serial} {f}'
    area=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    if prev is None:rank=s*np.sqrt(area)
    else:
     inter=np.maximum(0,np.minimum(b[:,2:],prev[2:])-np.maximum(b[:,:2],prev[:2])).prod(1);iou=inter/(area+np.prod(prev[2:]-prev[:2])-inter+1e-6);distance=np.linalg.norm((b[:,:2]+b[:,2:])/2-(prev[:2]+prev[2:])/2,axis=1);rank=s*(.1+iou)*np.exp(-distance/400)
    j=int(np.argmax(np.where(keep,rank,-1)));prev=b[j];candidates={'boxes_native':b[keep].tolist(),'scores':s[keep].tolist()};selection={'selected_confidence':float(s[j]),'rule':'largest confident foreground initially; confidence/overlap/center continuity thereafter'}
   k,q=pose(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),bboxes=[prev.tolist()]);assert k.shape==(1,133,2) and q.shape==(1,133)
   points=k[0];rawscore=q[0];pixels=np.floor(points+.5).astype(int);inside=(pixels[:,0]>=0)&(pixels[:,0]<2048)&(pixels[:,1]>=0)&(pixels[:,1]<1536)
   inside&=validimage[pixels[:,1].clip(0,1535),pixels[:,0].clip(0,2047)];assert np.isfinite(points).all() and np.isfinite(rawscore).all()
   xy.append(points);scores.append(rawscore);valids.append(inside);boxes.append(prev.copy());records.append({'serial':serial,'frame':f,'detector_refresh':candidates,'selection':selection,'body_points_above2':int((rawscore[:17]>2).sum()),'hand_points_above2':int((rawscore[91:]>2).sum())})
   if f in samples:
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle(prev.tolist(),outline='yellow',width=4)
    for j,((x,y),score) in enumerate(zip(points,rawscore)):
     if inside[j] and score>2:draw.ellipse((x-3,y-3,x+3,y+3),fill='lime' if j<91 else 'red')
    sheet.paste(im.resize((384,288)),(samples.index(f)*384,vi*312+24));ImageDraw.Draw(sheet).text((samples.index(f)*384+3,vi*312+4),f'{serial} f{f}',fill='black')
  np.savez_compressed(out/f'{serial}.npz',xy=xy,score=scores,in_image=valids,boxes_native=boxes,frames=frames);allxy.append(xy);allscores.append(scores);allvalid.append(valids);allboxes.append(boxes)
  (out/'status.json').write_text(json.dumps({'completed_views':vi+1,'observations':(vi+1)*99,'seconds':time.time()-start},indent=2));print('independent backpack actor observed',serial,flush=True)
np.savez_compressed(out/'detections.npz',xy=allxy,score=allscores,in_image=allvalid,boxes_native=allboxes,frames=frames);sheet.save(out/'all8_identity_and_keypoints.jpg',quality=94)
report={'success':True,'frames':frames,'views':meta['views'],'observations':792,'detector_weights_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'rtmw_model_sha256':hashlib.sha256((ROOT/'wholebody_runtime_v1/rtmw.onnx').read_bytes()).hexdigest(),'records':records,'seconds':time.time()-start,'scope':'Independent dark-shirt backpack actor. All8x99 actual native undistorted frames, own calibration/order and original PTS; RTMW133 from native crops, CPU person detector refreshed every10frames, no helmet-actor masks or appearance inherited. SimCC raw score not probability. Candidate identity/landmarks need contact-sheet audit and geometric screening; no 3D body, backpack segmentation, rigid/deformation fit or novel-view reconstruction completion claimed. Raw source overlays and calibration-bearing outputs stay private.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('backpack actor observations complete',flush=True)
