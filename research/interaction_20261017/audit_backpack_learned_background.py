"""Cached ALIKED/LightGlue background matches, no camera replacement."""
import os
from pathlib import Path
import sys,json,time,itertools,os
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
if os.environ.get('HOI_SHARED_SITE'):sys.path.append(os.environ['HOI_SHARED_SITE'])
import torch,numpy as np,cv2
from PIL import Image
from lightglue import ALIKED,LightGlue
for name in ['aliked-n16.pth','aliked_lightglue_v0-1_arxiv.pth']:assert (Path.home()/'.cache/torch/hub/checkpoints'/name).exists(),'Only use existing local weights'
out=ROOT/'backpack_learned_background_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
data=ROOT/'backpack_native000_098_v2';m=json.loads((data/'manifest.json').read_text());det=np.load(ROOT/'backpack_wholebody_all99_v2/detections.npz');A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
extractor=ALIKED(max_num_keypoints=4096).eval().cuda();matcher=LightGlue(features='aliked',flash=False).eval().cuda();features=[];images=[];Ks=[];Ws=[]
with torch.inference_mode():
 for vi,v in enumerate(m['views']):
  rgb=cv2.resize(np.asarray(Image.open(data/v['serial']/'0015.png').convert('RGB')),(1024,768),interpolation=cv2.INTER_AREA);mask=cv2.resize(np.asarray(Image.open(data/v['serial']/'valid.png')),(1024,768),interpolation=cv2.INTER_NEAREST)>127
  x0,y0,x1,y1=np.rint(det['boxes_native'][vi,15]*.5).astype(int);mask[max(0,y0-30):min(768,y1+30),max(0,x0-30):min(1024,x1+30)]=False
  f=extractor.extract(torch.tensor(rgb,device='cuda').permute(2,0,1).float()/255,resize=None);points=f['keypoints'][0];pix=points.round().long();keep=torch.tensor(mask,device='cuda')[pix[:,1].clamp(0,767),pix[:,0].clamp(0,1023)];n=len(points)
  f={k:(value[:,keep] if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==n else value) for k,value in f.items()};features.append(f);images.append(rgb);Ks.append(A@np.array(v['K']));Ws.append(np.array(v['W']));print('background features',v['serial'],int(keep.sum()),flush=True)
 rows=[]
 for i,j in itertools.combinations(range(8),2):
  pred=matcher({'image0':features[i],'image1':features[j]});matches=pred['matches'][0].cpu().numpy();row={'cameras':[m['views'][i]['serial'],m['views'][j]['serial']],'learned_matches':len(matches),'usable':False}
  if len(matches)<20:rows.append(row);continue
  p1=features[i]['keypoints'][0][matches[:,0]].cpu().numpy();p2=features[j]['keypoints'][0][matches[:,1]].cpu().numpy();estimated,flag=cv2.findFundamentalMat(p1,p2,cv2.USAC_MAGSAC,2.,.999,10000)
  if estimated is None or flag is None or flag.sum()<20:rows.append(row);continue
  flag=flag.ravel()>0;T=Ws[j]@np.linalg.inv(Ws[i]);t=T[:3,3];S=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);F=np.linalg.inv(Ks[j]).T@S@T[:3,:3]@np.linalg.inv(Ks[i]);a=np.c_[p1[flag],np.ones(flag.sum())];b=np.c_[p2[flag],np.ones(flag.sum())];la=a@F.T;lb=b@F;e=abs((b*la).sum(-1))*.5*(1/np.linalg.norm(la[:,:2],axis=-1).clip(1e-10)+1/np.linalg.norm(lb[:,:2],axis=-1).clip(1e-10))
  row.update(usable=True,estimated_F_inliers=int(flag.sum()),provided_calibration_median_epi_px1024=float(np.median(e)));rows.append(row);np.savez_compressed(out/f'{i}_{j}_matches.npz',points0=p1[flag],points1=p2[flag],estimated_F=estimated)
  canvas=np.concatenate([images[i],images[j]],axis=1).copy()
  for a,b in zip(p1[flag][:40],p2[flag][:40]):cv2.line(canvas,tuple(np.rint(a).astype(int)),tuple(np.rint(b+[1024,0]).astype(int)),(0,255,0),1)
  Image.fromarray(canvas).save(out/f'{i}_{j}_matches.jpg',quality=94);print('background pair',i,j,row['estimated_F_inliers'],flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'pairs':rows,'seconds':time.time()-start,'scope':'Existing cached ALIKED/LightGlue models, frame15 at1024wide, main actor envelope excluded. MAGSAC2px F with>=20inliers screens learned matches before checking supplied camera geometry. Repetitive foliage/tripods, moving background people and planar degeneracy need private match-image review; no original camera changed or recalibration accepted. Matched point arrays/raw source overlays remain private.'},indent=2));print('learned background diagnostic complete',flush=True)
