"""Compare generic model landmarks to actual LHM neural-rendered face evidence."""
import os
from pathlib import Path
import sys,json,itertools,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from rtmlib import RTMPose
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from avatar_rig import pose_axis_angles
p=argparse.ArgumentParser();p.add_argument('--out',default='lhm_face_anchor_correspondence_v1');p.add_argument('--source',default='avatar_canonical_views_v2');p.add_argument('--avatar');options=p.parse_args()
torch.set_num_threads(4);cv2.setNumThreads(2);out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();source=ROOT/options.source
from canonical_identity import semantic_identity
source_report=json.loads((source/'report.json').read_text());avatar_source=options.avatar or source_report.get('source_avatar') or ('avatar_hands_all300_v1' if options.source=='avatar_canonical_views_v2' else None);assert avatar_source,'Explicit avatar provenance required'
if source_report.get('source_avatar'):assert source_report['source_avatar']==avatar_source
bundle=torch.load(ROOT/avatar_source/'avatar.pt',weights_only=True)
meta=json.loads((source/'cameras.json').read_text());pose=torch.load(source/'pose.pt',weights_only=True);assert torch.equal(pose['betas'],bundle['poses']['betas']) and torch.equal(pose['scale'],bundle['poses']['scale']);model=SMPLXSurfaceLandmarks();prediction=model(pose_axis_angles(pose,0,'cpu'),pose['scale'],pose['translation_world'],betas=pose['betas']);proxy=prediction['coco17'][0,:5].detach().numpy();head=prediction['joints55'][0,15].detach().numpy()
detector=RTMPose(str(ROOT/'wholebody_runtime_v1/rtmw.onnx'),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False);points=[];scores=[];views=[];Ps=[];rows=[]
for view in meta['views']:
    W=np.array(view['W']);K=np.array(view['K']);center=-W[:3,:3].T@W[:3,3];direction=center-head;direction/=np.linalg.norm(direction)
    if direction[2]<.3:continue
    index=view['id'];rgbpath=source/view.get('rgb',f'{index:03d}.png');alphapath=source/view.get('alpha',f'{index:03d}_alpha.png')
    if not rgbpath.exists():rgbpath=source/f'{index:03d}.png'
    image=np.array(Image.open(rgbpath).convert('RGB'));alpha=np.array(Image.open(alphapath));yy,xx=np.where(alpha>127);box=[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)]
    xy,score=detector(image[:,:,::-1].copy(),bboxes=[box]);xy=xy[0,:5];score=score[0,:5];P=K@W[:3];q=np.c_[proxy,np.ones(5)]@P.T;uv=q[:,:2]/q[:,2:];error=np.linalg.norm(xy-uv,axis=1);pic=Image.fromarray(image);draw=ImageDraw.Draw(pic)
    for j,(x,y) in enumerate(xy):
        if score[j]>2:draw.ellipse((x-2,y-2,x+2,y+2),fill='lime')
    for x,y in uv:draw.ellipse((x-3,y-3,x+3,y+3),outline='red',width=1)
    pic.save(out/f'{index:03d}_overlay.png');points.append(xy);scores.append(score);Ps.append(P);views.append(view);rows.append({'virtual_view_id':index,'raw_scores':score.tolist(),'render_detected_minus_proxy_error_px504':error.tolist()})
assert len(views)>=6
xy=np.array(points);score=np.array(scores);P=np.array(Ps);anchors=[];reports=[]
def triangulate(ids,j):
    A=np.concatenate([np.stack([xy[i,j,0]*P[i,2]-P[i,0],xy[i,j,1]*P[i,2]-P[i,1]]) for i in ids]);_,_,V=np.linalg.svd(A);h=V[-1];return h[:3]/h[3]
for j in range(5):
    valid=(score[:,j]>2)&(xy[:,j,0]>0)&(xy[:,j,0]<504)&(xy[:,j,1]>0)&(xy[:,j,1]<504);ids=np.flatnonzero(valid);best=None
    for pair in itertools.combinations(ids,2):
        point=triangulate(pair,j);q=P@np.r_[point,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,j],axis=1);keep=valid&(error<5)&(q[:,2]>0);rank=(int(keep.sum()),-float(error[keep].sum()))
        if best is None or rank>best[0]:best=(rank,keep)
    if best is None or best[0][0]<3:anchors.append([np.nan]*3);reports.append({'coco_index':j,'usable':False});continue
    ids=np.flatnonzero(best[1]);point=triangulate(ids,j);q=P@np.r_[point,1];error=np.linalg.norm(q[:,:2]/q[:,2:]-xy[:,j],axis=1);anchors.append(point.tolist());reports.append({'coco_index':j,'usable':True,'support':len(ids),'median_inlier_reprojection_px504':float(np.median(error[ids])),'distance_to_generic_proxy_model_units':float(np.linalg.norm(point-proxy[j])/float(pose['scale']))})
anchors=np.array(anchors);local=(anchors-head)/float(pose['scale']);np.savez_compressed(out/'anchors.npz',canonical_face_local=local,canonical_world=anchors,generic_proxy_world=proxy,head_joint_world=head,raw_xy=xy,raw_score=score,projection_matrices=P)
report={'success':True,'seconds':time.time()-start,'source_avatar':avatar_source,'source_render':options.source,'semantic_identity_sha256':semantic_identity(bundle),'front_virtual_views':len(views),'points':reports,'observations':rows,'scope':'Same pretrained RTMW detector applied to actual stated-LHM neural A-pose renders, restricted to front-facing virtual cameras. Generic SMPL-X surface COCO5 projections compared in identical camera pixels. Five-pixel RANSAC produces candidate semantic3D anchors only where>=3views agree. These are detector/neural-render pseudo-anchors, not observed real surface/geometry ground truth. Canonical tensor identity recorded for safe downstream fitting; no head deformation/pose adoption performed by this audit. No distilled mesh body used.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'front_views':len(views),'points':reports}),flush=True)
