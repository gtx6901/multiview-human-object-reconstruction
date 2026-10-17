"""Dense SO3 point-screen search, actual UV reranking and temporal Viterbi."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2,trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from mesh_renderer import MeshRenderer
from helmet_texture_sampling import srgb_to_linear
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--start',type=int,default=50);p.add_argument('--stop',type=int,default=81);p.add_argument('--rotations',type=int,default=4096);p.add_argument('--topk',type=int,default=16);p.add_argument('--motion',default='helmet_inspection_shell_fit_all300_v1');p.add_argument('--texture',default='helmet_inspection_texture_surface_v3');o=p.parse_args()
assert 0<=o.start<o.stop<=300 and o.rotations>=128 and 2<=o.topk<=64
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
assert json.loads((ROOT/'batched_uv_verification_v1/report.json').read_text())['success']
m=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);mesh=torch.load(ROOT/o.texture/'mesh.pt',weights_only=True);assert torch.allclose(mesh['vertices_local'][mesh['faces'].long()],m['vertices_local'][m['faces'].long()],atol=1e-6)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};color=json.loads((ROOT/o.texture/'camera_color_correction.json').read_text());gain=torch.tensor(color['gain_linear_rgb'],device='cuda');bias=torch.tensor(color['bias_linear_rgb'],device='cuda')
vertices=mesh['vertices_local'].cuda();faces=mesh['faces'].cuda();uv=mesh['uv'].cuda();texture=mesh['texture'].cuda();texconf=torch.tensor(np.array(Image.open(ROOT/o.texture/'observed_texels.png')),device='cuda',dtype=torch.float32)/255;scale=m['logscale'].exp().cuda();renderer=MeshRenderer()
rng=np.random.default_rng(20261018);Q=np.concatenate([np.eye(3)[None],Rotation.random(o.rotations-1,random_state=rng).as_matrix()]);Q=torch.tensor(Q,device='cuda',dtype=torch.float32)
tm=trimesh.Trimesh(mesh['vertices_local'].numpy(),mesh['faces'].numpy(),process=False);sample_faces=rng.choice(len(tm.faces),256,p=np.array(tm.area_faces)/tm.area_faces.sum());points=torch.tensor(np.array(tm.triangles_center)[sample_faces],device='cuda',dtype=torch.float32)
size=96;A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
caches=[]
for v in meta['views']:
 d=np.load(ROOT/'lhm_occlusion_all300_v1'/f"{v['serial']}.npz");caches.append({k:d[k] for k in ['depth','alpha','hand_unknown','decoded_alpha']})
states=[];data_costs=[];reports=[]
with torch.inference_mode():
 for f in range(o.start,o.stop):
  R0=m['R'][f].cuda();t=m['translation_world'][f].cuda();Rs=R0[None]@Q;observations=[];coarse=torch.zeros(o.rotations,device='cuda')
  for vi,v in enumerate(meta['views']):
   r=records[v['serial'],f]
   if not r['mask']:continue
   box=np.array(r['roi_xyxy'],float);extent=max(box[2:]-box[:2]);origin=(box[:2]+box[2:])/2-extent/2;s=size/extent;C=np.array([[s,0,-s*origin[0]+.5*s-.5],[0,s,-s*origin[1]+.5*s-.5],[0,0,1.]])
   lab=cv2.warpAffine(np.array(Image.open(ROOT/'sam2_all300_v1'/r['labels'])),C[:2],(size,size),flags=cv2.INTER_NEAREST,borderValue=255)
   positive=np.argwhere(lab==1)
   if len(positive)<16:continue
   target=positive[np.linspace(0,len(positive)-1,min(16,len(positive))).astype(int)][:,::-1].copy();target=torch.tensor(target,device='cuda',dtype=torch.float32)
   distance=cv2.distanceTransform((lab==0).astype('uint8'),cv2.DIST_L2,3);distance=torch.tensor(distance,device='cuda');K=torch.tensor(C@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
   # Geometry-only screen, not opaque rasterization: both outside-mask distance
   # and target-to-projected-point coverage prevent trivial centroid solutions.
   for first in range(0,o.rotations,512):
    stop=min(first+512,o.rotations);world=torch.einsum('bij,nj->bni',Rs[first:stop],points)*scale+t;pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;xy=q[...,:2]/q[...,2:].clamp_min(.01);ij=torch.floor(xy+.5).long();inside=(pc[...,2]>.01)&(ij[...,0]>=0)&(ij[...,0]<size)&(ij[...,1]>=0)&(ij[...,1]<size)
    outside=distance[ij[...,1].clamp(0,size-1),ij[...,0].clamp(0,size-1)];outside=torch.where(inside,outside,torch.full_like(outside,16.)).mean(1)
    coverage=torch.cdist(target[None].expand(stop-first,-1,-1),xy).min(-1).values.clamp_max(16).mean(1);coarse[first:stop]+=(outside+.4*coverage)
   raw=np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB'));rgb=cv2.warpAffine(raw,C[:2],(size,size),flags=cv2.INTER_LINEAR).astype('float32')/255
   warp=lambda x:cv2.warpAffine(x,(C@np.linalg.inv(A))[:2],(size,size),flags=cv2.INTER_NEAREST)
   d=caches[vi];ba=warp(d['alpha'][f]);bd=warp(d['depth'][f]);hand=warp(d['hand_unknown'][f]);da=warp(d['decoded_alpha'][f]);conf=(lab==1)&(hand==0)&~((da>.5)&(ba<.5))&~(rgb>.985).all(-1)
   observations.append({'K':K,'W':W,'lab':torch.tensor(lab,device='cuda'),'truth':(srgb_to_linear(torch.tensor(rgb,device='cuda'))*gain[vi]+bias[vi]).clamp(0,1),'conf':torch.tensor(conf,device='cuda'),'ba':torch.tensor(ba,device='cuda'),'bd':torch.tensor(bd,device='cuda')})
  assert len(observations)>=2,f'Not enough observations at{f}'
  coarse/=len(observations);top=coarse.topk(min(64,o.rotations),largest=False).indices
  if not (top==0).any():top=torch.cat([torch.zeros(1,device='cuda',dtype=torch.long),top[:-1]])
  candidates=Rs[top];cost=torch.zeros(len(top),device='cuda');ious=torch.zeros_like(cost);rgb_errors=torch.zeros_like(cost);support=torch.zeros_like(cost)
  for obs in observations:
   for first in range(0,len(top),16):
    stop=min(first+16,len(top));world=torch.einsum('bij,nj->bni',candidates[first:stop],vertices)*scale+t;B=len(world);r=renderer.depth_alpha_batch(world,faces,obs['K'][None].expand(B,-1,-1),obs['W'][None].expand(B,-1,-1),size,size,uv=uv,texture=texture,texture_confidence=texconf)
    valid=(obs['lab']!=255).float();truth=(obs['lab']==1).float();pred=r['alpha']*valid;intersection=(pred*truth).sum((1,2));union=(pred+truth-pred*truth).sum((1,2)).clamp_min(1);iou=intersection/union
    conf=obs['conf'][None]&(r['texture_confidence']>.75)&(r['alpha']>.9)&((obs['ba'][None]<.5)|(r['depth']<obs['bd'][None]+.01));count=conf.sum((1,2));linear=srgb_to_linear((r['premultiplied_rgb']/r['alpha'][...,None].clamp_min(1e-5)).clamp(0,1));error=((linear-obs['truth']).abs().mean(-1)*conf).sum((1,2))/count.clamp_min(1);error=torch.where(count>=16,error,torch.full_like(error,.2))
    cost[first:stop]+=1-iou+2*error;ious[first:stop]+=iou;rgb_errors[first:stop]+=error;support[first:stop]+=count
  cost/=len(observations);order=cost.topk(o.topk,largest=False).indices;identity_index=int(torch.where(top==0)[0][0])
  if not (order==identity_index).any():order[-1]=identity_index
  states.append(candidates[order].cpu().numpy());data_costs.append(cost[order].cpu().numpy())
  reports.append({'frame':f,'views':len(observations),'rotations_point_screened':o.rotations,'rotations_uv_reranked':len(top),'identity_cost':float(cost[identity_index]),'best_cost':float(cost[order[0]]),'best_input_iou':float(ious[order[0]]/len(observations)),'best_rgb_error_with_support_penalty':float(rgb_errors[order[0]]/len(observations)),'best_compared_pixels':int(support[order[0]])});print(json.dumps(reports[-1]),flush=True)
# Dynamic programming over full rotation matrices, with angular transition cost.
dp=data_costs[0].copy();back=[]
if o.start>0:
 cosine=np.clip((np.einsum('ij,bij->b',m['R'][o.start-1].numpy(),states[0])-1)/2,-1,1);dp+=.015*(np.arccos(cosine)/.1)**2
for i in range(1,len(states)):
 cosine=np.clip((np.einsum('aij,bij->ab',states[i-1],states[i])-1)/2,-1,1);angle=np.arccos(cosine);joint=dp[:,None]+.015*(angle/.1)**2;parent=joint.argmin(0);dp=data_costs[i]+joint[parent,np.arange(o.topk)];back.append(parent)
if o.stop<300:
 cosine=np.clip((np.einsum('ij,bij->b',m['R'][o.stop].numpy(),states[-1])-1)/2,-1,1);dp+=.015*(np.arccos(cosine)/.1)**2
path=[int(dp.argmin())]
for parent in back[::-1]:path.append(int(parent[path[-1]]))
path=path[::-1];R=m['R'].clone()
for i,f in enumerate(range(o.start,o.stop)):R[f]=torch.tensor(states[i][path[i]])
torch.save({**m,'R':R},out/'sequence.pt');step=Rotation.from_matrix(R.numpy());angles=np.rad2deg((step[:-1].inv()*step[1:]).magnitude());change=np.rad2deg((Rotation.from_matrix(m['R'].numpy()).inv()*step).magnitude())
report={'success':True,'seconds':time.time()-start,'interval':[o.start,o.stop],'uniform_random_so3_candidates_plus_identity':o.rotations,'coarse_surface_points':256,'topk_viterbi_states':o.topk,'frames':reports,'viterbi_state_indices':path,'frame_rotation_change_from_initializer_deg':change[o.start:o.stop].tolist(),'boundary_step_degrees':{'before_interval':None if o.start==0 else float(angles[o.start-1]),'after_interval':None if o.stop==300 else float(angles[o.stop-1])},'scope':'Selected inspection interval only; same fixed mesh/UV/scale/translation and accepted human. Haar-random SO3 samples plusidentity, geometry point-distance screen then64actual textured raster candidates per frame across all available cameras,16-state temporal Viterbi with squared geodesic transition and original neighboring-frame boundary penalties. Original rotation is explicitly retained among states. Point screen approximates silhouette and can miss modes;4096 samples do NOT guarantee10degree coverage. UV can carry pose/lighting bias. Two-camera/unknown-depth guards and pixel-support penalties documented in source. Boundary jumps and full300 local refinement/rendering must be inspected; this is NOT an adopted full-action result or independent quality metric.'};(out/'report.json').write_text(json.dumps(report,indent=2));print('dense rotation / Viterbi candidate complete',flush=True)
