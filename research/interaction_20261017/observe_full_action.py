"""Real 300-frame/eight-camera body and native whole-body observations.

Reuse exact existing observations where available, with explicit provenance.
Unobserved frames are inferred, never duplicated from adjacent frames.
"""
import os
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,cv2,numpy as np
from PIL import Image,ImageDraw
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from rtmlib import RTMPose
torch.set_num_threads(4);cv2.setNumThreads(2)
out=ROOT/'observations_all300_v1';out.mkdir(exist_ok=False);start=time.time()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());assert json.loads((ROOT/'native_dataset_audit_v1/report.json').read_text())['success']
body_sources=['detections_pickup000_135_v1','detections_contiguous180_211_v2']
cache={};provenance=[]
for name in body_sources:
    path=ROOT/name/'detections.npz';d=np.load(path)
    for t,f in enumerate(d['frames']):cache[int(f)]=(name,t,d)
    provenance.append({'file':name+'/detections.npz','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':d['frames'].tolist()})
oldwhole=np.load(ROOT/'wholebody_native000_135_v1/detections.npz')
weights=Path.home()/'.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
bodymodel=keypointrcnn_resnet50_fpn(weights=None,weights_backbone=None).cuda().eval();bodymodel.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'),strict=True)
modelpath=ROOT/'wholebody_runtime_v1/rtmw.onnx';wholemodel=RTMPose(str(modelpath),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False)
samples=[0,60,135,150,165,180,210,240,270,299];sheet=Image.new('RGB',(len(samples)*256,8*212),'white')
allbody=[];allbscore=[];allboxes=[];allwhole=[];allwscore=[];allinside=[];records=[];views=[]
affine=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
with torch.inference_mode():
    for vi,v in enumerate(meta['views']):
        cap=cv2.VideoCapture(v['source']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert (w,h)==(2048,1536)
        mx,my=cv2.initUndistortRectifyMap(np.array(v['K_native']),np.array(v['distortion_opencv5']),None,np.array(v['K']),(w,h),cv2.CV_32FC1)
        validimg=cv2.remap(np.ones((h,w),np.uint8),mx,my,cv2.INTER_NEAREST)>0
        vx=[];vs=[];vb=[];wx=[];ws=[];wi=[];prev=None;parity=[]
        for f in range(300):
            ok,bgr=cap.read();assert ok
            native=cv2.remap(bgr,mx,my,cv2.INTER_LINEAR);small=cv2.resize(native,(1024,768),interpolation=cv2.INTER_AREA)
            missing=False
            if f in cache:
                name,t,d=cache[f];kp=d['xy'][vi,t];ks=d['score'][vi,t];box=d['boxes'][vi,t];kind=name
                missing=bool((ks>3).sum()==0)
            else:
                rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
                pred=bodymodel([torch.from_numpy(rgb.transpose(2,0,1).copy()).float().cuda()/255])[0]
                boxes=pred['boxes'].cpu().numpy();scores=pred['scores'].cpu().numpy();assert prev is not None
                if len(boxes):
                    inter=np.maximum(0,np.minimum(boxes[:,2:],prev[2:])-np.maximum(boxes[:,:2],prev[:2])).prod(1)
                    area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1);iou=inter/(area+np.prod(prev[2:]-prev[:2])-inter+1e-8)
                    distance=np.linalg.norm((boxes[:,:2]+boxes[:,2:]-prev[:2]-prev[2:])/2,axis=1)
                    eligible=(scores>.5)&((iou>.1)|(distance<max(100,np.linalg.norm(prev[2:]-prev[:2])*.6)))
                else:eligible=np.zeros(0,dtype=bool)
                if eligible.any():
                    rank=scores*(.2+iou)*np.exp(-distance/300);j=int(np.argmax(np.where(eligible,rank,-1)))
                    kp=pred['keypoints'][j,:,:2].cpu().numpy();ks=pred['keypoints_scores'][j].cpu().numpy();box=boxes[j]
                else:missing=True;kp=np.zeros((17,2),np.float32);ks=np.zeros(17,np.float32);box=prev.copy()
                kind='actual KeypointRCNN current-frame inference'
            if not missing:prev=np.array(box).copy()
            vx.append(kp);vs.append(ks);vb.append(box)
            if f<136:
                points=oldwhole['xy'][vi,f];scores=oldwhole['score'][vi,f];inside=oldwhole['in_image'][vi,f];wkind='wholebody_native000_135_v1'
            else:
                nativebox=(np.array(box).reshape(2,2)+.5)*2-.5
                k,s=wholemodel(native,bboxes=[nativebox.reshape(4).tolist()]);assert k.shape==(1,133,2) and s.shape==(1,133)
                points=k[0];scores=s[0];pixels=np.floor(points+.5).astype(int)
                inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
                inside&=validimg[pixels[:,1].clip(0,h-1),pixels[:,0].clip(0,w-1)]
                if missing:inside[:]=False
                wkind='actual native RTMW current-frame inference'
            assert np.isfinite(points).all() and np.isfinite(scores).all()
            wx.append(points);ws.append(scores);wi.append(inside)
            records.append({'serial':v['serial'],'frame':f,'body_source':kind,'wholebody_source':wkind,'body_missing':missing})
            if f in samples:
                rgb=cv2.cvtColor(native,cv2.COLOR_BGR2RGB);cached=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png'))
                error=int(np.abs(rgb.astype(int)-cached.astype(int)).max());assert error==0;parity.append({'frame':f,'max_u8_difference':error})
                pic=Image.fromarray(rgb);draw=ImageDraw.Draw(pic)
                for j,((x,y),q) in enumerate(zip(points,scores)):
                    if inside[j] and q>2:draw.ellipse((x-3,y-3,x+3,y+3),fill='lime' if j<91 else 'red')
                thumb=pic.resize((256,192));sheet.paste(thumb,(samples.index(f)*256,vi*212+20));ImageDraw.Draw(sheet).text((samples.index(f)*256+2,vi*212+2),f"{v['serial']} f{f}",fill='black')
        cap.release();digest=hashlib.sha256(Path(v['source']).read_bytes()).hexdigest();assert digest==v['source_sha256']
        np.savez_compressed(out/f"{v['serial']}.npz",body_xy=vx,body_score=vs,boxes=vb,xy=wx,score=ws,in_image=wi,frames=np.arange(300))
        allbody.append(vx);allbscore.append(vs);allboxes.append(vb);allwhole.append(wx);allwscore.append(ws);allinside.append(wi)
        views.append({'serial':v['serial'],'K':v['K'],'K_body1024':(affine@np.array(v['K'])).tolist(),'native_to_body1024':affine.tolist(),'W':v['W'],'source_sha256':digest,'width':2048,'height':1536,'parity':parity})
        (out/'status.json').write_text(json.dumps({'completed_views':len(views),'observations':len(views)*300,'seconds':time.time()-start},indent=2));print('completed',v['serial'],flush=True)
np.savez_compressed(out/'body.npz',xy=allbody,score=allbscore,boxes=allboxes,frames=np.arange(300))
np.savez_compressed(out/'wholebody.npz',xy=allwhole,score=allwscore,in_image=allinside,frames=np.arange(300))
sheet.save(out/'all8_action.jpg')
report={'success':True,'frames':list(range(300)),'views':views,'records':records,'reuse_sources':provenance,'wholebody_reuse_sha256':hashlib.sha256((ROOT/'wholebody_native000_135_v1/detections.npz').read_bytes()).hexdigest(),'body_weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'wholebody_weights_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'seconds':time.time()-start,'scope':'Actual eight-view full300-frame observations. Body points in1024×768 undistorted pixels; whole-body in2048×1536 native pixels. Exact historical observations reused with hashes; remaining frames newly inferred, not interpolated. Scores are detector scores, not visibility probabilities. Needs identity/visibility/multiview QA. Original source/calibration unchanged; no fitted motion or reconstruction completion claim.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('all300 observations complete',flush=True)
