"""Test real helmet texture correspondences before another pose/texture loop."""
import os
from pathlib import Path
import sys,os,json,time,itertools
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
if os.environ.get('HOI_SHARED_SITE'):sys.path.append(os.environ['HOI_SHARED_SITE'])
import torch,numpy as np,cv2
from PIL import Image
from lightglue import ALIKED,LightGlue
for name in ['aliked-n16.pth','aliked_lightglue_v0-1_arxiv.pth']:assert (Path.home()/'.cache/torch/hub/checkpoints'/name).exists()
out=ROOT/'helmet_image_features_audit_v1';out.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(2);start=time.time()
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};K=np.array([v['K'] for v in meta['views']]);W=np.array([v['W'] for v in meta['views']]);P=K@W[:,:3];centers=-np.einsum('vji,vj->vi',W[:,:3,:3],W[:,:3,3])
extractor=ALIKED(max_num_keypoints=1024).eval().cuda();matcher=LightGlue(features='aliked',flash=False).eval().cuda();reports=[]
with torch.inference_mode():
 for t in [42,90]:
  features={};native={};images={};origins={};extents={};viewrows=[]
  for vi,v in enumerate(meta['views']):
   rec=records[(v['serial'],t)]
   if rec['mask'] is None:continue
   image=np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{t:04d}.png').convert('RGB'));label=np.array(Image.open(ROOT/'sam2_all300_v1'/rec['labels']));positive=label==1
   yy,xx=np.where(positive)
   if len(xx)<100:continue
   extent=int(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)+64);origin=np.floor(np.array([(xx.max()+xx.min())/2,(yy.max()+yy.min())/2])-extent/2).astype(int);crop=Image.fromarray(image).crop((*origin,*(origin+extent)));crop=np.array(crop.resize((512,512),Image.Resampling.BILINEAR));f=extractor.extract(torch.tensor(crop,device='cuda').permute(2,0,1).float()/255,resize=None)
   points=f['keypoints'][0].cpu().numpy();uv=(points+.5)*extent/512-.5+origin;pix=np.rint(uv).astype(int);x=pix[:,0].clip(0,2047);y=pix[:,1].clip(0,1535);core=cv2.erode(positive.astype('uint8'),np.ones((9,9),np.uint8))>0
   hands=np.load(ROOT/'shared_head_occlusion_all300_v3'/f"{v['serial']}.npz")['hand_unknown'][t];cxy=np.floor((uv+.5)*.125).astype(int);keep=(pix[:,0]>=0)&(pix[:,0]<2048)&(pix[:,1]>=0)&(pix[:,1]<1536)&core[y,x]&(hands[cxy[:,1].clip(0,191),cxy[:,0].clip(0,255)]==0);n=len(points);mask=torch.tensor(keep,device='cuda');f={key:(value[:,mask] if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==n else value) for key,value in f.items()}
   features[vi]=f;native[vi]=uv[keep];images[vi]=crop;origins[vi]=origin;extents[vi]=extent;viewrows.append({'serial':v['serial'],'interior_features':int(keep.sum())})
  parent={};edges=[];pairrows=[]
  def find(x):
   parent.setdefault(x,x)
   if parent[x]!=x:parent[x]=find(parent[x])
   return parent[x]
  for u,v in itertools.combinations(features,2):
   if not len(native[u]) or not len(native[v]):continue
   pred=matcher({'image0':features[u],'image1':features[v]});matches=pred['matches'][0].cpu().numpy();scores=pred['scores'][0].cpu().numpy();matches=matches[scores>.2];T=W[v]@np.linalg.inv(W[u]);tx,ty,tz=T[:3,3];S=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]]);F=np.linalg.inv(K[v]).T@S@T[:3,:3]@np.linalg.inv(K[u]);a=np.c_[native[u][matches[:,0]],np.ones(len(matches))];b=np.c_[native[v][matches[:,1]],np.ones(len(matches))];la=a@F.T;lb=b@F;e=abs((b*la).sum(-1))*.5*(1/np.maximum(np.linalg.norm(la[:,:2],axis=-1),1e-10)+1/np.maximum(np.linalg.norm(lb[:,:2],axis=-1),1e-10));accepted=matches[e<3];pairrows.append({'cameras':[meta['views'][u]['serial'],meta['views'][v]['serial']],'learned_score_above_02':len(matches),'provided_epipolar_under_3_native_px':len(accepted)})
   for i,j in accepted:
    a=(u,int(i));b=(v,int(j));parent[find(a)]=find(b);edges.append((a,b))
  components={}
  for node in parent:components.setdefault(find(node),[]).append(node)
  points=[];tracks=[]
  for nodes in components.values():
   cams=[v for v,_ in nodes]
   if len(nodes)<3 or len(set(cams))!=len(nodes):continue
   A=np.concatenate([np.stack([native[v][j,0]*P[v,2]-P[v,0],native[v][j,1]*P[v,2]-P[v,1]]) for v,j in nodes]);_,_,V=np.linalg.svd(A);h=V[-1];xyz=h[:3]/h[3];q=P[cams]@np.r_[xyz,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-np.array([native[v][j] for v,j in nodes]),axis=-1);rays=xyz-centers[cams];rays/=np.linalg.norm(rays,axis=1)[:,None];angle=np.rad2deg(np.arccos(np.clip(rays@rays.T,-1,1))).max()
   if (q[:,2]<=0).any() or error.max()>3 or angle<3:continue
   points.append(xyz);tracks.append({'nodes':nodes,'reprojection_native_px':error.tolist(),'ray_angle_deg':float(angle)})
  np.savez_compressed(out/f'{t:04d}_feature_points_private.npz',world=np.array(points),tracks=np.array([json.dumps(x) for x in tracks]));reports.append({'frame':t,'views':viewrows,'pairs':pairrows,'three_or_more_view_3d_points':len(points),'total_consistent_pair_edges':len(edges)})
  print('helmet feature frame',t,'3view points',len(points),'edges',len(edges),flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'frames':reports,'seconds':time.time()-start,'scope':'Cached ALIKED/LightGlue helmet-interior crop features; actual positive candidate mask eroded4native pixels and fitted-hand uncertainty excluded.512-square crops rescale native pixels explicitly; no shape/model agreement gate. Matchscore>.2, supplied-camera symmetric epipolar<3native px, unique-camera tracks>=3, DLT maximum reprojection3px, positive depth and ray span>=3deg. Tests whether real image detail can constrain shared helmet geometry/pose rather than recycling biased baked colors. Segmentation/detector/feature/calibration errors and white-shell sparsity remain; no replacement pose/geometry adopted, only2times tested. Private point/track arrays and original images excluded from public release.'},indent=2));print('helmet image correspondence audit complete',flush=True)
