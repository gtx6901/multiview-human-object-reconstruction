"""Evaluate one common rotation mode over EVERY frame of a grasp phase.

This is a seed candidate, not an adopted motion or guaranteed SO3 covering.
"""
import os
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2,trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from mesh_renderer import MeshRenderer
from helmet_texture_sampling import srgb_to_linear
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--motion',required=True);p.add_argument('--texture',required=True);p.add_argument('--body-depth',required=True);p.add_argument('--phase',type=int,default=0);p.add_argument('--rotations',type=int,default=4096);o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
m=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);previous=json.loads((ROOT/o.motion/'report.json').read_text());phase=previous['phases'][o.phase];frames=list(range(phase['start'],phase['stop_exclusive']));assert 2<=len(frames)<=150 and o.rotations>=128
mesh=torch.load(ROOT/o.texture/'mesh.pt',weights_only=True);assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],m['vertices_local'][m['faces'].long()],atol=1e-6)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};cc=json.loads((ROOT/o.texture/'camera_color_correction.json').read_text());assert cc['reference_serial']==meta['views'][1]['serial']
vertices=mesh['vertices_local'].cuda();faces=mesh['faces'].cuda();uv=mesh['uv'].cuda();texture=mesh['texture'].cuda();confidence=torch.tensor(np.array(Image.open(ROOT/o.texture/'observed_texels.png')),device='cuda',dtype=torch.float32)/255;scale=m['logscale'].exp().cuda();pivot=(vertices.min(0).values+vertices.max(0).values)/2;renderer=MeshRenderer();size=96;A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
rng=np.random.default_rng(20261019);q=np.concatenate([np.eye(3)[None],Rotation.random(o.rotations-1,random_state=rng).as_matrix()]);Q=torch.tensor(q,device='cuda',dtype=torch.float32)
tm=trimesh.Trimesh(mesh['vertices_local'].numpy(),mesh['faces'].numpy(),process=False);fi=rng.choice(len(tm.faces),256,p=np.array(tm.area_faces)/tm.area_faces.sum());points=torch.tensor(np.array(tm.triangles_center)[fi],device='cuda',dtype=torch.float32)
observations=[];coarse=torch.zeros(o.rotations,device='cuda')
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
  cache=np.load(ROOT/o.body_depth/f"{v['serial']}.npz");gain=torch.tensor(cc['gain_linear_rgb'][vi],device='cuda');bias=torch.tensor(cc['bias_linear_rgb'][vi],device='cuda')
  for f in frames:
   rec=records[v['serial'],f]
   if not rec['mask']:continue
   box=np.array(rec['roi_xyxy'],float);extent=max(box[2:]-box[:2]);origin=(box[:2]+box[2:])/2-extent/2;s=size/extent;C=np.array([[s,0,-s*origin[0]+.5*s-.5],[0,s,-s*origin[1]+.5*s-.5],[0,0,1.]])
   lab=cv2.warpAffine(np.array(Image.open(ROOT/'sam2_all300_v1'/rec['labels'])),C[:2],(size,size),flags=cv2.INTER_NEAREST,borderValue=255);positive=np.argwhere(lab==1)
   if len(positive)<16:continue
   target=torch.tensor(positive[np.linspace(0,len(positive)-1,min(16,len(positive))).astype(int)][:,::-1].copy(),device='cuda',dtype=torch.float32);distance=torch.tensor(cv2.distanceTransform((lab==0).astype('uint8'),cv2.DIST_L2,3),device='cuda');K=torch.tensor(C@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);R0=m['R'][f].cuda();t0=m['translation_world'][f].cuda();Rs=R0[None]@Q;ts=t0+(R0[None]-Rs)@pivot*scale
   for first in range(0,o.rotations,256):
    stop=min(first+256,o.rotations);world=torch.einsum('bij,nj->bni',Rs[first:stop],points)*scale+ts[first:stop,None];pc=world@W[:3,:3].T+W[:3,3];proj=pc@K.T;xy=proj[...,:2]/proj[...,2:].clamp_min(.01);ij=torch.floor(xy+.5).long();inside=(pc[...,2]>.01)&(ij[...,0]>=0)&(ij[...,0]<size)&(ij[...,1]>=0)&(ij[...,1]<size);outside=distance[ij[...,1].clamp(0,size-1),ij[...,0].clamp(0,size-1)];outside=torch.where(inside,outside,torch.full_like(outside,16.)).mean(1);coverage=torch.cdist(target[None].expand(stop-first,-1,-1),xy).amin(-1).clamp_max(16).mean(1);coarse[first:stop]+=outside+.4*coverage
   raw=np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB'));rgb=cv2.warpAffine(raw,C[:2],(size,size),flags=cv2.INTER_LINEAR).astype('float32')/255
   warp=lambda x:cv2.warpAffine(x,(C@np.linalg.inv(A))[:2],(size,size),flags=cv2.INTER_NEAREST)
   ba=warp(cache['alpha'][f]);bd=warp(cache['depth'][f]);hand=warp(cache['hand_unknown'][f]);da=warp(cache['decoded_alpha'][f]);known=(lab==1)&(hand==0)&~((da>.5)&(ba<.5))&~(rgb>.985).all(-1)
   observations.append({'frame':f,'camera':v['serial'],'K':K,'W':W,'lab':torch.tensor(lab,device='cuda'),'truth':(srgb_to_linear(torch.tensor(rgb,device='cuda'))*gain+bias).clamp(0,1),'known':torch.tensor(known,device='cuda'),'ba':torch.tensor(ba,device='cuda'),'bd':torch.tensor(bd,device='cuda')})
  print('shared phase point screen',v['serial'],flush=True)
 assert set(x['frame'] for x in observations)==set(frames)
 top=coarse.topk(min(64,o.rotations),largest=False).indices
 if not (top==0).any():top=torch.cat([torch.zeros(1,device='cuda',dtype=torch.long),top[:-1]])
 costs=torch.zeros(len(top),device='cuda');ious=costs.clone();photos=costs.clone();counts=torch.zeros(len(top),device='cuda');perframe={f:[] for f in frames}
 for obs in observations:
  f=obs['frame'];R0=m['R'][f].cuda();t0=m['translation_world'][f].cuda();Rs=R0[None]@Q[top];ts=t0+(R0[None]-Rs)@pivot*scale
  for first in range(0,len(top),8):
   stop=min(first+8,len(top));B=stop-first;world=torch.einsum('bij,nj->bni',Rs[first:stop],vertices)*scale+ts[first:stop,None];r=renderer.depth_alpha_batch(world,faces,obs['K'][None].expand(B,-1,-1),obs['W'][None].expand(B,-1,-1),size,size,uv=uv,texture=texture,texture_confidence=confidence)
   valid=(obs['lab']!=255).float();target=(obs['lab']==1).float();pred=r['alpha']*valid;intersection=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1);iou=intersection/union
   # Candidate-independent positive source support; disappearing rendered
   # alpha/observed-UV coverage costs .25 rather than erasing RGB errors.
   support=obs['known'].float();available=r['alpha'].clamp(0,1)*r['texture_confidence'].clamp(0,1)*((obs['ba'][None]<.5)|(r['depth']<obs['bd'][None]+.01));weight=support*available;linear=srgb_to_linear((r['premultiplied_rgb']/r['alpha'][...,None].clamp_min(1e-5)).clamp(0,1));photo=(((linear-obs['truth']).abs().mean(-1)*weight)+.25*(support-weight)).sum((1,2))/support.sum().clamp_min(1);costs[first:stop]+=1-iou+.5*photo;ious[first:stop]+=iou;photos[first:stop]+=photo;counts[first:stop]+=(weight>.75).sum((1,2))
  print('shared phase UV comparison',f,obs['camera'],flush=True)
 costs/=len(observations);ious/=len(observations);photos/=len(observations);identity=int(torch.where(top==0)[0][0]);best=int(costs.argmin());chosen=Q[top[best]].cpu();R=m['R'].clone();t=m['translation_world'].clone();pivot=pivot.cpu();scale=scale.cpu()
 for f in frames:R[f]=m['R'][f]@chosen;t[f]=m['translation_world'][f]+(m['R'][f]-R[f])@pivot*scale
 # Update the phase-level initializer too; subsequent phase fitting must
 # not immediately clamp the new seed back to the old relative mode.
 oldrel=torch.tensor(previous['shared_relative_R'][o.phase]);newrel=oldrel@chosen;newrelt=torch.tensor(previous['shared_relative_translation'][o.phase])+(oldrel-newrel)@pivot*scale
 torch.save({**m,'R':R,'translation_world':t},out/'sequence.pt');rows=[{'candidate':int(top[i]),'cost':float(costs[i]),'input_iou':float(ious[i]),'fixed_source_rgb_error':float(photos[i]),'compared_pixels':int(counts[i])} for i in range(len(top))]
 result={**previous,'success':True,'source_motion':o.motion,'source_texture':o.texture,'source_body_depth':o.body_depth,'search_phase':o.phase,'search_frames':frames,'shared_rotation_candidates':o.rotations,'uv_reranked':len(top),'observations':len(observations),'same_protocol_identity':rows[identity],'selected_shared_mode':rows[best],'mode_change_deg':float(np.rad2deg(Rotation.from_matrix(chosen.numpy()).magnitude())),'mode_candidates':rows,'seconds':time.time()-start,'scope':'One common object-local rotation about preserved bounding-box center evaluated jointly on EVERY frame of the stated phase. Haar-random SO3 plus identity, geometry point screen then64 actual textured raster modes, all available calibrated cameras. Fixed positive-source photometric denominator and missing-coverage penalty, frozen geometry/texture/human. New phase-relative seed updated consistently; per-frame states outside phase exact. No automatic adoption, guaranteed rotation coverage, independent truth or boundary continuity claim; full motion refit and actual source/novel render still required.'}
 result['shared_relative_R'][o.phase]=newrel.tolist();result['shared_relative_translation'][o.phase]=newrelt.tolist();(out/'report.json').write_text(json.dumps(result,indent=2));print('shared rotation mode candidate complete',flush=True)
