"""Fit a candidate planar rim to visually selected face-on inspection outlines."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from PIL import Image,ImageDraw,ImageOps
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_inspection_opening_plane_v1');p.add_argument('--shape',default='helmet_held_native_shape_v1');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
shape=options.shape;m=torch.load(ROOT/shape/'sequence.pt',weights_only=True);vertices=m['vertices_local'].numpy();tri=vertices[m['faces'].numpy()];center=(vertices.min(0)+vertices.max(0))/2
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());v=next(x for x in meta['views'] if x['serial']=='21305717');K=np.array(v['K']);W=np.array(v['W']);cam=-W[:3,:3].T@W[:3,3];scale=float(m['logscale'].exp());records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};observations=[];normals=[]
for f in [60,70]:
 r=records[v['serial'],f];mask=np.array(Image.open(ROOT/'sam2_all300_v1'/r['mask']))>127;contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=cv2.convexHull(max(contours,key=cv2.contourArea)).reshape(-1,2)
 # Resample the convex outline at uniform arc length, avoiding pixel-density bias.
 closed=np.r_[contour,contour[:1]];length=np.linalg.norm(np.diff(closed,axis=0),axis=1);s=np.r_[0,np.cumsum(length)];q=np.linspace(0,s[-1],96,endpoint=False);target=np.stack([np.interp(q,s,closed[:,j]) for j in range(2)],1)
 R=m['R'][f].numpy();t=m['translation_world'][f].numpy();localcam=(cam-t)@R/scale;direction=center-localcam;direction/=np.linalg.norm(direction);normals.append(direction);observations.append(dict(frame=f,R=R,t=t,target=target,tree=cKDTree(target),record=r))
n0=np.mean(normals,axis=0);n0/=np.linalg.norm(n0);basis=np.eye(3)[np.argmin(np.abs(n0))];u=np.cross(n0,basis);u/=np.linalg.norm(u);w=np.cross(n0,u)
def plane(parameters):
 n=n0+parameters[0]*u+parameters[1]*w;n/=np.linalg.norm(n);radius=np.ptp(vertices@n)/2;level=float(center@n+parameters[2]*radius);return n,level
def ring(parameters):
 n,h=plane(parameters);d=tri@n-h;points=[]
 for a,b in [(0,1),(1,2),(2,0)]:
  hit=d[:,a]*d[:,b]<0;fraction=d[hit,a]/(d[hit,a]-d[hit,b]);points.append(tri[hit,a]+fraction[:,None]*(tri[hit,b]-tri[hit,a]))
 return np.concatenate(points)
def project(points,o):
 world=points@o['R'].T*scale+o['t'];pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;return q[:,:2]/q[:,2:]
def cost(parameters):
 points=ring(parameters)
 if len(points)<30:return 1e4
 n,_=plane(parameters);values=[]
 for o,direction in zip(observations,normals):
  if np.dot(n,direction)<.25:return 1e4
  uv=project(points,o);a=o['tree'].query(uv)[0];b=cKDTree(uv).query(o['target'])[0];values.append(.5*(np.mean(np.minimum(a,40))+np.mean(np.minimum(b,40))))
 return float(np.mean(values))
rng=np.random.default_rng(20261018);candidates=np.r_[np.zeros((1,3)),rng.uniform([-1.,-1.,-.8],[1.,1.,.8],(256,3))];rank=sorted([(cost(x),x) for x in candidates],key=lambda item:item[0]);solutions=[]
for _,x in rank[:4]:
 result=minimize(cost,x,method='Powell',bounds=[(-1.2,1.2),(-1.2,1.2),(-.85,.85)],options={'maxiter':45,'xtol':1e-4,'ftol':1e-4});solutions.append((float(result.fun),result.x))
best,parameters=min(solutions,key=lambda item:item[0]);n,h=plane(parameters);points=ring(parameters);sheet=Image.new('RGB',(768,408),'white');draw=ImageDraw.Draw(sheet)
for i,o in enumerate(observations):
 im=Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f"{o['frame']:04d}.png").convert('RGB');d=ImageDraw.Draw(im)
 for xy in project(points,o):d.ellipse((xy[0]-1,xy[1]-1,xy[0]+1,xy[1]+1),fill=(255,0,255))
 for xy in o['target']:d.ellipse((xy[0]-1,xy[1]-1,xy[0]+1,xy[1]+1),fill=(0,255,0))
 sheet.paste(ImageOps.pad(im.crop(tuple(o['record']['roi_xyxy'])),(384,384),color='white'),(i*384,24));draw.text((i*384+5,5),f"f{o['frame']} green outline / magenta candidate rim",fill='black')
sheet.save(out/'rim_projection.jpg',quality=95)
(out/'report.json').write_text(json.dumps({'success':bool(np.isfinite(best)),'seconds':time.time()-start,'shape_source':shape,'serial':v['serial'],'frames':[60,70],'opening_plane_normal_local':n.tolist(),'opening_level_local':h,'mean_clipped_symmetric_outline_distance_native_px':best,'parameter_solution':parameters.tolist(),'seed_normal_disagreement_deg':float(np.rad2deg(np.arccos(np.clip(np.dot(normals[0],normals[1]),-1,1)))),'scope':'Experimental image-constrained planar-opening candidate. Face-on views717 frames60/70 selected after seeing real exposed lining/rim; convex SAM object outlines approximate rim, not certified inner-edge annotations. Optimize intersection curve of fixed held-phase closed hull against those2D outlines, no head/eye anatomical cutting plane. Unknown plane depth, pose/hull bias, visor/rim thickness and non-planar opening remain; two nearby frames from one camera do NOT independently recover true cavity or guarantee worn fit. Inspect projection and full multiview rendering before adopting. No human geometry or motion changed.'},indent=2));print('inspection rim-plane candidate fitted',best,flush=True)
