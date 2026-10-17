"""Static-background matches distinguish calibration from human-motion issues."""
import os
from pathlib import Path
import sys,json,itertools,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import cv2,numpy as np
from PIL import Image
out=ROOT/'backpack_background_geometry_v1';out.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
data=ROOT/'backpack_native000_098_v2';m=json.loads((data/'manifest.json').read_text());det=np.load(ROOT/'backpack_wholebody_all99_v2/detections.npz');A=np.array([[.5,0,-.25],[0,.5,-.25],[0,0,1.]])
sift=cv2.SIFT_create(nfeatures=6000);images=[];features=[];Ks=[];Ws=[]
for vi,v in enumerate(m['views']):
 rgb=np.asarray(Image.open(data/v['serial']/'0015.png').convert('RGB'));gray=cv2.cvtColor(cv2.resize(rgb,(1024,768),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY);mask=cv2.resize(np.asarray(Image.open(data/v['serial']/'valid.png')),(1024,768),interpolation=cv2.INTER_NEAREST)
 b=det['boxes_native'][vi,15]*.5;pad=30;x0,y0,x1,y1=np.rint(b).astype(int);mask[max(0,y0-pad):min(768,y1+pad),max(0,x0-pad):min(1024,x1+pad)]=0
 k,d=sift.detectAndCompute(gray,mask);features.append((k,d));images.append(gray);Ks.append(A@np.array(v['K']));Ws.append(np.array(v['W']))
rows=[];matcher=cv2.BFMatcher()
for i,j in itertools.combinations(range(8),2):
 ki,di=features[i];kj,dj=features[j];row={'cameras':[m['views'][i]['serial'],m['views'][j]['serial']]}
 if di is None or dj is None:rows.append({**row,'usable':False});continue
 forward=matcher.knnMatch(di,dj,k=2);reverse=matcher.knnMatch(dj,di,k=2);reversegood={a.queryIdx:a.trainIdx for a,b in reverse if a.distance<.7*b.distance};good=[a for a,b in forward if a.distance<.7*b.distance and reversegood.get(a.trainIdx)==a.queryIdx];row['mutual_ratio_matches']=len(good)
 if len(good)<20:rows.append({**row,'usable':False});continue
 p1=np.array([ki[x.queryIdx].pt for x in good]);p2=np.array([kj[x.trainIdx].pt for x in good]);estimated,flag=cv2.findFundamentalMat(p1,p2,cv2.USAC_MAGSAC,2.,.999,10000)
 if estimated is None or flag is None or flag.sum()<20:rows.append({**row,'usable':False});continue
 flag=flag.ravel()>0;T=Ws[j]@np.linalg.inv(Ws[i]);t=T[:3,3];S=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);F=np.linalg.inv(Ks[j]).T@S@T[:3,:3]@np.linalg.inv(Ks[i]);a=np.c_[p1[flag],np.ones(flag.sum())];b=np.c_[p2[flag],np.ones(flag.sum())];la=a@F.T;lb=b@F;e=abs((b*la).sum(-1))*.5*(1/np.linalg.norm(la[:,:2],axis=-1).clip(1e-10)+1/np.linalg.norm(lb[:,:2],axis=-1).clip(1e-10))
 row.update(usable=True,estimated_F_inliers=int(flag.sum()),provided_calibration_median_epi_px1024=float(np.median(e)));rows.append(row)
 selected=[x for x,k in zip(good,flag) if k][:40];image=cv2.drawMatches(images[i],ki,images[j],kj,selected,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);cv2.imwrite(str(out/f'{i}_{j}_background_matches.jpg'),image)
(out/'report.json').write_text(json.dumps({'success':True,'pairs':rows,'seconds':time.time()-start,'scope':'One-time static-background diagnostic at frame15,1024wide, own K pixel affine and original W. Main actor envelope excluded; mutual SIFT ratio.7 and MAGSAC F<=2px select matches, >=20inliers required. Provided-calibration errors measured on this independently screened set. Repetitive foliage, planar degeneracy and moving background people remain possible; private match images must be visually checked before attributing failure or proposing new calibration. Does not modify original cameras, estimate temporal offsets or certify source sync.'},indent=2));print(json.dumps(rows),flush=True)
