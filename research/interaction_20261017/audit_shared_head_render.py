"""Remeasure actual neural renders; do not validate a proxy using itself."""
import os
from pathlib import Path
import sys,json,argparse,importlib.util,time
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from accelerate import init_empty_weights
from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
from avatar_rig import animate,pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.lbs import batch_rigid_transform
from rtmlib import RTMPose
spec=importlib.util.spec_from_file_location('gsplat_cuda',os.environ["HOI_GSPLAT_EXTENSION"]);backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend);sys.modules['gsplat.csrc']=backend
from gsplat import rasterization
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--candidate',default='shared_head_alignment_clean_v2');p.add_argument('--frames',type=int,nargs='+',default=[0,42,180,299]);p.add_argument('--baseline');p.add_argument('--motion-update',action='store_true');p.add_argument('--face-anchors');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
report=json.loads((ROOT/o.candidate/'report.json').read_text());base=o.baseline or report['source_avatar'];bundles=[torch.load(ROOT/f/'avatar.pt',weights_only=True) for f in [base,o.candidate]]
assert torch.equal(bundles[0]['features'],bundles[1]['features'])
if o.motion_update:
 from canonical_identity import semantic_identity
 assert semantic_identity(bundles[0])==semantic_identity(bundles[1]) and o.face_anchors
else:assert all(torch.equal(bundles[0]['poses'][k],bundles[1]['poses'][k]) for k in bundles[0]['poses'])
def cuda(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:cuda(v) for k,v in x.items()}
 return x
rigs=[cuda(b['rig']) for b in bundles];allposes=[cuda(b['poses']) for b in bundles];poses=allposes[0];features=bundles[0]['features'].cuda().float();bg=bundles[0]['background_embedding'][2].cuda().float()
with init_empty_weights(include_buffers=False):decoder=PatchDPT4DecoderOnly(**bundles[0]['decoder_config'])
decoder.transformer_block=torch.nn.Module();decoder.load_state_dict(bundles[0]['decoder'],strict=True,assign=True);decoder.eval().cuda()
detector=RTMPose(str(ROOT/'wholebody_runtime_v1/rtmw.onnx'),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());screen=np.load(ROOT/'body_head_faces_all300_v1/face_screen.npz');anchor=np.load(ROOT/(o.face_anchors or 'lhm_face_anchor_correspondence_v1')/'anchors.npz')['canonical_face_local'][:3]
if o.motion_update:locals=[anchor,anchor]
else:
 C=axis_angle_to_matrix(torch.tensor(report['shared_rotation_axis_angle_rad'])).numpy();delta=np.array(report['shared_translation_model_units']);locals=[anchor,anchor@C.T+delta]
width,height=1008,756;s=width/2048;affine=np.array([[s,0,(s-1)/2],[0,s,(s-1)/2],[0,0,1.]]);records=[];private=Image.new('RGB',(3*336,8*len(o.frames)*248),'white');generated=Image.new('RGB',(2*336,8*len(o.frames)*248),'white');arrays=[];candidate_label='BODY/FACE MOTION' if o.motion_update else 'SHARED HEAD'
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);K=torch.tensor(affine@v['K'],device='cuda',dtype=torch.float32)
  for t in o.frames:
   images=[];alphas=[]
   for bi,rig in enumerate(rigs):
    g=animate(rig,allposes[bi],t);color,alpha,_=rasterization(means=g['xyz'],quats=g['rotation'],scales=g['scaling'],opacities=g['opacity'].flatten(),colors=torch.cat([g['shs'].squeeze(1),features],1),viewmats=W[None],Ks=K[None],width=width,height=height,backgrounds=torch.cat([torch.ones(3,device='cuda'),bg])[None],packed=False,render_mode='RGB+ED',near_plane=.01,far_plane=100.,eps2d=.3)
    with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=decoder.predict_rgbs(None,color[0,...,3:-1].permute(2,0,1)[None],None,height,width)
    a=mask[0,0].float().squeeze()*(alpha[0,...,0]>.001);im=(rgb[0,0].float()*a[...,None]+1-a[...,None]).cpu().numpy();images.append(np.uint8(im.clip(0,1)*255));alphas.append(a.cpu().numpy())
   yy,xx=np.where((np.maximum(*alphas) if o.motion_update else alphas[0])>.5);box=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float);box[:2]-=8;box[2:]+=8
   source=cv2.resize(np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{t:04d}.png').convert('RGB')),(width,height),interpolation=cv2.INTER_AREA)
   points=[];scores=[]
   for im in [source,*images]:
    q,sc=detector(im[:,:,::-1].copy(),bboxes=[box.tolist()]);points.append(q[0,:3]);scores.append(sc[0,:3])
   aa=pose_axis_angles(poses,t,'cuda');rot=axis_angle_to_matrix(aa);joints,_=batch_rigid_transform(rot,rigs[0]['rest_joints'],rigs[0]['parents']);R=rot[0,0]
   for j in [3,6,9,12,15]:R=R@rot[0,j]
   ht=joints[0,15]*poses['scale']+poses['translation_world'][t];front=anchor[0]-anchor[1:].mean(0);front[1]=0;front/=np.linalg.norm(front);direction=(R@torch.tensor(front,device='cuda',dtype=torch.float32)).cpu().numpy();center=(-W[:3,:3].T@W[:3,3]).cpu().numpy();ray=center-ht.cpu().numpy();ray/=np.linalg.norm(ray);facing=float(direction@ray)>.15
   predictions=[]
   for bi,local in enumerate(locals):
    ap=allposes[bi];ar=axis_angle_to_matrix(pose_axis_angles(ap,t,'cuda'));aj,_=batch_rigid_transform(ar,rigs[bi]['rest_joints'],rigs[bi]['parents']);ah=ar[0,0]
    for j in [3,6,9,12,15]:ah=ah@ar[0,j]
    at=aj[0,15]*ap['scale']+ap['translation_world'][t];pw=at+torch.tensor(local,device='cuda',dtype=torch.float32)@ah.T*ap['scale'];pc=pw@W[:3,:3].T+W[:3,3];q=pc@K.T;predictions.append((q[:,:2]/q[:,2:]).cpu().numpy())
   valid=screen['inlier'][vi,t,:3]&facing&(np.array(scores)>2).all(0);before=np.linalg.norm(points[1]-points[0],axis=1)/s;after=np.linalg.norm(points[2]-points[0],axis=1)/s;proxy_before=np.linalg.norm(points[1]-predictions[0],axis=1)/s;proxy_after=np.linalg.norm(points[2]-predictions[1],axis=1)/s
   row=len(records);facecenter=np.mean(predictions[0],axis=0);extent=160;bounds=(int(facecenter[0]-extent/2),int(facecenter[1]-extent*.55),int(facecenter[0]+extent/2),int(facecenter[1]+extent*.95))
   for col,im in enumerate([source,*images]):
    crop=Image.fromarray(im).crop(bounds).resize((336,224));private.paste(crop,(col*336,row*248+24));ImageDraw.Draw(private).text((col*336+3,row*248+4),f'{v["serial"]} f{t} '+['SOURCE','BASELINE',candidate_label][col],fill='black')
    if col:generated.paste(crop,((col-1)*336,row*248+24));ImageDraw.Draw(generated).text(((col-1)*336+3,row*248+4),f'{v["serial"]} f{t} '+['BASELINE',candidate_label][col-1],fill='black')
   records.append({'serial':v['serial'],'frame':t,'valid_face_points':int(valid.sum()),'front_facing':facing,'before_native_px':before[valid].tolist(),'after_native_px':after[valid].tolist(),'proxy_vs_render_before_native_px':proxy_before[valid].tolist(),'proxy_vs_render_after_native_px':proxy_after[valid].tolist()});print(v['serial'],t,'actual render face measured',flush=True)
private.save(out/'source_comparison_private.jpg',quality=95);generated.save(out/'generated_only_comparison.jpg',quality=95)
def gather(a,b,key):return [x for r in records if a<=r['frame']<b for x in r[key]]
phase={}
for label,a,b in [('clean',0,136),('worn',180,300)]:
 values={k:gather(a,b,k) for k in ['before_native_px','after_native_px','proxy_vs_render_before_native_px','proxy_vs_render_after_native_px']};phase[label]={'points':len(values['before_native_px']),**{k:float(np.mean(v)) if v else None for k,v in values.items()}}
scope='Actual neural body-only renders remeasured with same RTMW and same baseline-alpha-derived full-body bbox for source/baseline/candidate. Only initially front-facing existing RANSAC-inlier nose/eyes with all3 detector scores>2 contribute. Native1008x756 renders and explicit pixel-center camera affine; reported residuals converted back to2048 native pixels. Proxy-vs-render discrepancy is separately measured. Sampled same-input diagnostic, not independent ground truth; no helmet compositing here. Original-source comparison stays private.'
if o.motion_update:scope=scope.replace('baseline-alpha-derived','union-of-both-render-alpha-derived')+' Motion-update comparison with canonical tensor identity held exact; per-avatar actual shaped rest FK and motion used for each semantic projection.'
(out/'report.json').write_text(json.dumps({'success':True,'source_avatar':base,'candidate_avatar':o.candidate,'frames':o.frames,'records':records,'phase_actual_render_errors':phase,'seconds':time.time()-start,'scope':scope},indent=2));print(json.dumps(phase),flush=True)
