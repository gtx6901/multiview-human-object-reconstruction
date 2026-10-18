"""Compare a small shared opening/rim family against all native observations.

Outer source hull is fixed. A few time-invariant opening parameters vary;
all300 object poses are fixed during this geometry block. New UV/baking and a
pose block are required afterwards. No per-frame geometry is introduced.
"""
import os
from pathlib import Path
import sys,json,time,argparse,itertools
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2,trimesh
from PIL import Image
from mesh_renderer import MeshRenderer
from helmet_opening_profile import cut_opening
from visible_face_coverage import face_coverage
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--motion',required=True);p.add_argument('--avatar');p.add_argument('--closed-source');p.add_argument('--smoothing-iterations',type=int);p.add_argument('--body-depth',required=True);p.add_argument('--verify-baseline-only',action='store_true');p.add_argument('--level-shifts',type=float,nargs='+',default=[-.2,-.1,0,.1]);p.add_argument('--tilt-shifts',type=float,nargs='+',default=[-.2,0,.2]);p.add_argument('--side-drops',type=float,nargs='+',default=[0.]);p.add_argument('--back-drops',type=float,nargs='+',default=[0.]);p.add_argument('--front-face-visibility',action='store_true');p.add_argument('--face-anchors');p.add_argument('--legacy-raw-face-coverage',action='store_true');p.add_argument('--face-depth-margin',type=float,default=.01)
p.add_argument('--control-mesh',help='Current shell control matching the stated trajectory; original opening parameter frame remains fixed')
p.add_argument('--material-weight',type=float,default=0.);p.add_argument('--material-stop',type=int,default=136)
o=p.parse_args();assert o.material_weight>=0 and 0<o.material_stop<=300
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
motion=torch.load(ROOT/o.motion/'sequence.pt',weights_only=True);assert motion['frames'].tolist()==list(range(300))
ref=ROOT/'helmet_shell_all300_v5';refmesh=torch.load(ref/'mesh.pt',weights_only=True);refreport=json.loads((ref/'report.json').read_text());source=o.closed_source or refreport['source_motion'];closed_source=torch.load(ROOT/source/'sequence.pt',weights_only=True)
assert torch.allclose(motion['logscale'],closed_source['logscale'])
control_mesh=torch.load(ROOT/o.control_mesh/'mesh.pt',weights_only=True) if o.control_mesh else refmesh
assert motion['vertices_local'][motion['faces'].long()].shape==control_mesh['vertices_local'][control_mesh['faces'].long()].shape and torch.allclose(motion['vertices_local'][motion['faces'].long()],control_mesh['vertices_local'][control_mesh['faces'].long()],atol=1e-6),'Current control geometry must match fitted trajectory'
control_report=json.loads((ROOT/o.control_mesh/'report.json').read_text()) if o.control_mesh else refreport
scale=float(motion['logscale'].exp());closed=trimesh.Trimesh(closed_source['vertices_local'].numpy(),closed_source['faces'].numpy(),process=False)
if closed.volume<0:closed.invert()
smoothing_iterations=refreport['smoothing']['iterations'] if o.smoothing_iterations is None else o.smoothing_iterations;assert smoothing_iterations>=0
before=closed.vertices.copy();trimesh.smoothing.filter_taubin(closed,lamb=.5,nu=.53,iterations=smoothing_iterations);delta=closed.vertices-before;size=np.linalg.norm(delta,axis=1)*scale;closed.vertices=before+delta*np.minimum(1,.005/np.maximum(size,1e-12))[:,None]
n0=np.array(refreport['opening_plane_normal_local']);origin0=n0*refreport['opening_level_local'];up=np.array(refreport['opening_up_local']);front=np.array(refreport['opening_front_local'])
def build(level,tilt,side=0.,back=0.):
 normal=n0+tilt*front;normal/=np.linalg.norm(normal);origin=origin0+level*up
 outer=cut_opening(closed,normal,origin,up,front,side,back);v=outer.vertices.copy();f=outer.faces.copy();N=len(v)
 assert N>100 and not outer.is_watertight
 inner=v-.001/scale*outer.vertex_normals;edges=outer.edges;_,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True);edge=edges[counts[inverse]==1];rim=np.concatenate([np.stack([edge[:,1],edge[:,0],edge[:,0]+N],1),np.stack([edge[:,1],edge[:,0]+N,edge[:,1]+N],1)])
 faces=np.concatenate([f,f[:,::-1]+N,rim]);vertices=np.concatenate([v,inner]);shell=trimesh.Trimesh(vertices,faces,process=False);assert shell.is_watertight and shell.is_winding_consistent and shell.volume>0
 colors=np.full_like(vertices,.2);colors[:N]=.9;colors[N:]=.05;classes=np.r_[np.zeros(len(f)),np.ones(len(f)),np.full(len(rim),2)]
 return {'vertices_local':torch.tensor(vertices,dtype=torch.float32),'faces':torch.tensor(faces,dtype=torch.int32),'colors':torch.tensor(colors,dtype=torch.float32),'face_classes':torch.tensor(classes,dtype=torch.int64),'outer_vertex_count':N},normal,float(origin@normal)
if o.closed_source or o.control_mesh:baseline=control_mesh
else:
 baseline,_,_=build(0,0);assert torch.allclose(baseline['vertices_local'][baseline['faces'].long()],refmesh['vertices_local'][refmesh['faces'].long()],atol=1e-6),'Rebuilt zero-parameter shell must reproduce current canonical geometry'
if o.verify_baseline_only:
 assert not o.closed_source and not o.control_mesh and o.smoothing_iterations is None,'Original zero-parameter rebuild control only'
 (out/'report.json').write_text(json.dumps({'success':True,'zero_parameter_geometry_matches_current_shell':True,'scope':'Shared opening-family control only. No observation scoring, optimization or quality claim.'},indent=2));print('opening parameter baseline exactly reproduces current shell',flush=True);sys.exit(0)
human_source=o.avatar or o.motion;cache_report=json.loads((ROOT/o.body_depth/'report.json').read_text());assert cache_report['avatar_source']==human_source,'Refresh visibility from the stated human avatar before geometry scoring'
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());maskroot=ROOT/'sam2_all300_v1';records=json.loads((maskroot/'report.json').read_text())['records'];lookup={(r['serial'],r['frame']):r for r in records};A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
directional_valid=None
if o.front_face_visibility:
 from joint_head_constraints import JointHeadConstraints
 face_guard=JointHeadConstraints(ROOT,human_source,o.body_depth,meta,torch.ones(300,device='cuda',dtype=torch.bool),o.face_anchors,True);directional_valid=face_guard.valid.cpu().numpy();del face_guard
screen=np.load(ROOT/'body_head_faces_all300_v1/face_screen.npz');labels=[];eligibles=[];depths=[];Ks=[];Ws=[];frameids=[];viewids=[];facecoords=[];facevalid=[];material_targets=[];material_known=[]
for vi,v in enumerate(meta['views']):
 cache=np.load(ROOT/o.body_depth/f"{v['serial']}.npz");assert np.allclose(cache['K'],A@np.array(v['K'])) and np.allclose(cache['W'],v['W']);valid_native=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/'valid.png'))
 for f in range(300):
  r=lookup[(v['serial'],f)]
  if not r['mask']:continue
  bounds=np.array(r['roi_xyxy'],dtype=float);extent=max(bounds[2:]-bounds[:2]);origin=(bounds[:2]+bounds[2:])/2-extent/2;s=256/extent;C=np.array([[s,0,-s*origin[0]+.5*s-.5],[0,s,-s*origin[1]+.5*s-.5],[0,0,1.]])
  def warp(x,border=0):return cv2.warpAffine(x,C[:2],(256,256),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=border)
  def cached(x):return cv2.warpAffine(x,(C@np.linalg.inv(A))[:2],(256,256),flags=cv2.INTER_NEAREST)
  lab=warp(np.asarray(Image.open(maskroot/r['labels'])),255)
  if (lab==1).sum()<10:continue
  person=warp(np.asarray(Image.open(maskroot/r['person_mask'])))>127;obj=warp(np.asarray(Image.open(maskroot/r['mask'])))>127;guard=int(np.ceil(8*s));boundary=cv2.dilate(obj.astype('uint8'),np.ones((2*guard+1,2*guard+1),np.uint8))>0;bd=cached(cache['depth'][f]);ba=cached(cache['alpha'][f]);hand=cached(cache['hand_unknown'][f]);valid=warp(valid_native)>127
  yy,xx=np.indices((256,256));nx=(xx-C[0,2])/s;ny=(yy-C[1,2])/s;roi=(nx>=np.floor(bounds[0]))&(nx<np.ceil(bounds[2]))&(ny>=np.floor(bounds[1]))&(ny<np.ceil(bounds[3]))
  eligible=(lab==255)&person&~boundary&(hand==0)&valid&(ba>0)&roi
  face=screen['xy'][vi,f,:3];q=np.c_[face,np.ones(3)]@C.T;q=q[:,:2];pix=np.floor(q+.5).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<256)&(pix[:,1]>=0)&(pix[:,1]<256);x=pix[:,0].clip(0,255);y=pix[:,1].clip(0,255)
  fv=screen['inlier'][vi,f,:3]&inside&(hand[y,x]==0)&(lab[y,x]!=1)
  if directional_valid is not None:fv&=directional_valid[vi,f,:3]
  if o.material_weight:
   known=np.zeros((256,256),bool);bright=known.copy()
   if f<o.material_stop:
    native=np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB'));rgb=cv2.warpAffine(native,C[:2],(256,256),flags=cv2.INTER_LINEAR).astype('float32')/255;luma=rgb@np.array([.2126,.7152,.0722]);white=max(.6,float(np.quantile(luma[lab==1],.95)));normalized=luma/white;bright=normalized>.7;known=(lab==1)&(hand==0)&valid&((normalized>.7)|(normalized<.2))
    if 'decoded_alpha' in cache:known&=~((cached(cache['decoded_alpha'][f])>.5)&(ba<.5))
   material_targets.append(bright);material_known.append(known)
  labels.append(lab);eligibles.append(eligible);depths.append(bd);Ks.append(C@np.array(v['K']));Ws.append(v['W']);frameids.append(f);viewids.append(vi);facecoords.append(pix.clip(0,255));facevalid.append(fv)
 print('opening observations',v['serial'],flush=True)
labels=torch.tensor(np.array(labels));eligibles=torch.tensor(np.array(eligibles));depths=torch.tensor(np.array(depths));Ks=torch.tensor(np.array(Ks),device='cuda',dtype=torch.float32);Ws=torch.tensor(np.array(Ws),device='cuda',dtype=torch.float32);frameids=torch.tensor(frameids,device='cuda');facecoords=torch.tensor(np.array(facecoords),device='cuda');facevalid=torch.tensor(np.array(facevalid),device='cuda');R=motion['R'].cuda();t=motion['translation_world'].cuda();renderer=MeshRenderer();chunks=list(torch.arange(len(labels),device='cuda').split(32));rows=[];best=None
assert len(labels)>1500 and set(frameids.cpu().tolist())==set(range(300))
if o.material_weight:material_targets=torch.tensor(np.array(material_targets));material_known=torch.tensor(np.array(material_known))
with torch.inference_mode():
 for rebuilt,level,tilt,side,back in [(False,0.,0.,0.,0.)]+[(True,*q) for q in itertools.product(o.level_shifts,o.tilt_shifts,o.side_drops,o.back_drops) if o.closed_source or any(x!=0 for x in q)]:
  if rebuilt:mesh,normal,plane=build(level,tilt,side,back)
  else:
   mesh,normal,plane=baseline,np.array(control_report['opening_plane_normal_local']),float(control_report['opening_level_local'])
   if o.control_mesh:
    selection=control_report.get('selected',{});level=selection.get('level_shift',0.);tilt=selection.get('tilt_shift',0.);side=selection.get('side_drop',0.);back=selection.get('back_drop',0.)
  vertices=mesh['vertices_local'].cuda();faces=mesh['faces'].cuda();ious=[];facehits=[];facecounts=[];rawfacehits=[];material_errors=[];material_counts=[]
  for ids in chunks:
   world=torch.einsum('bij,nj->bni',R[frameids[ids]],vertices)*scale+t[frameids[ids],None];render=renderer.depth_alpha_batch(world,faces,Ks[ids],Ws[ids],256,256,colors=mesh['colors'].cuda() if o.material_weight else None);lab=labels[ids.cpu()].cuda();restore=eligibles[ids.cpu()].cuda()&(render['hard_alpha']>.5)&(render['depth']>0)&(render['depth']<depths[ids.cpu()].cuda()-.01);valid=(lab!=255).float()+.5*restore;target=(lab==1).float();pred=render['alpha']*valid;intersection=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1);ious.append(intersection/union)
   if o.material_weight:
    known=material_known[ids.cpu()].cuda();truth=material_targets[ids.cpu()].cuda().float();predicted=((render['premultiplied_rgb'][...,0]/render['alpha'].clamp_min(1e-5)-.05)/.85).clamp(0,1);error=((predicted-truth).abs()*render['alpha']+(1-render['alpha']))*known;material_errors.append(error.sum());material_counts.append(known.sum())
   num,den,raw=face_coverage(render['alpha'],render['depth'],depths[ids.cpu()].cuda(),facecoords[ids],facevalid[ids],o.face_depth_margin);facehits.append(num);facecounts.append(den);rawfacehits.append(raw)
  iou=torch.cat(ious);rawpen=float(torch.stack(rawfacehits).sum()/facevalid.sum().clamp_min(1));facepen=rawpen if o.legacy_raw_face_coverage else float(torch.stack(facehits).sum()/torch.stack(facecounts).sum().clamp_min(1));loss=float(1-iou.mean())+.5*facepen;row={'source_family':('new_closed_hull' if o.closed_source else 'original_closed_hull') if rebuilt else 'current_shell_control','level_shift':level,'tilt_shift':tilt,'side_drop':side,'back_drop':back,'input_iou':float(iou.mean()),'visible_face_coverage':facepen,'raw_projected_face_coverage_diagnostic':rawpen,'depth_supported_face_points':int(torch.stack(facecounts).sum()),'loss':loss,'held_iou':float(iou[frameids<136].mean()),'worn_iou':float(iou[frameids>=180].mean())};rows.append(row);print(json.dumps(row),flush=True)
  if o.material_weight:
   material_error=float(torch.stack(material_errors).sum()/torch.stack(material_counts).sum().clamp_min(1));assert int(torch.stack(material_counts).sum())>1000;loss+=o.material_weight*material_error;row.update(loss=loss,inspection_material_error=material_error,material_support_pixels=int(torch.stack(material_counts).sum()));print('material-aware total',json.dumps(row),flush=True)
  if best is None or loss<best[0]:best=(loss,mesh,normal,plane,row,iou.cpu())
torch.save(best[1],out/'mesh.pt');torch.save({**motion,'vertices_local':best[1]['vertices_local'],'faces':best[1]['faces']},out/'sequence.pt');np.savez_compressed(out/'observation_metrics.npz',frames=frameids.cpu(),views=viewids,selected_iou=best[5]);scope='Two shared canonical opening parameters only: fixed source closed hull,30step bounded smoothing and.001world-numerical-unit shell thickness prior; all8x300 native ROI labels plus visible nose/eye non-coverage evidence. Fixed input poses/body-depth for this geometry block. Full-body segmentation, proxy face visibility and source hull remain imperfect. Selected input score is not real cavity/contact/new-view truth. New UV, texture bake, pose/head refit and actual image comparisons required before adoption; baseline reconstruction is asserted exact.'
report={'success':True,'observations':len(labels),'candidates':rows,'selected':best[4],'opening_plane_normal_local':best[2].tolist(),'opening_level_local':best[3],'opening_up_local':up.tolist(),'opening_front_local':front.tolist(),'source_shape':source,'baseline_shell':ref.name,'sources':vars(o),'seconds':time.time()-start,'scope':scope,'visible_face_points':int(facevalid.sum())}
if any(x!=0 for x in o.side_drops+o.back_drops):
 report['scope']=scope.replace('Two shared canonical opening parameters only','Shared canonical level/tilt and side/back rim-drop family')+' Nonplanar signed-field clipping lowers side/back rim relative to the plane without independently generating each frame. This is a low-dimensional opening-shape prior tested against observations, not measured cavity reconstruction. Plane metadata describes the base plane only; chosen side/back drops are explicit in selected. Surface clipping is linear on each source triangle.'
if o.front_face_visibility:report['scope']+=' Face noncoverage uses frozen initial directional visibility from the stated avatar and semantic anchor identity, excluding hand uncertainty and back-camera hallucinations; still model-dependent screening.'
report['smoothing_iterations']=smoothing_iterations
if o.control_mesh:report['control_shell']=o.control_mesh;report['scope']+=' Current shell control is the separately recorded control_shell, exactly matching trajectory geometry; original opening parameter frame remains fixed. This supports repeated geometry/pose alternation without assuming current geometry is the original shell.'
if o.material_weight:report['scope']+=' Explicit white-outer/dark-inner design prior additionally compared against fixed positive native-image support before material_stop, normalized per-view luminance thresholds>.7 or<.2, hands/unknown depth excluded. Missing rendered coverage penalized rather than dropped. It is a coarse material-class cue to opening geometry, not calibrated RGB/albedo, recovery of detailed lining or unique pose truth.'
if o.closed_source:report['scope']+=' New closed source compared to the unchanged current-shell control under the SAME updated motion/visibility protocol; no exact-zero-rebuild claim for a different source hull. Bounded smoothing iterations explicitly recorded.'
report['face_coverage_protocol']='legacy_raw_projection' if o.legacy_raw_face_coverage else 'mesh_in_front_of_reliable_lhm_depth'
report['scope']+=' Face-coverage selection now excludes helmet surfaces behind the human face: a point contributes only with reliable positive body depth and helmet depth in front by the recorded margin. Fixed body-depth-supported face denominator is independent of candidate coverage; missing body depth is unknown, not bare-depth fallback. Raw alpha projection is retained only as a diagnostic (or explicit legacy reproduction). LHM expected depth remains approximate and this is not physical contact certification.'
(out/'report.json').write_text(json.dumps(report,indent=2));print('shared opening geometry block complete; unadopted',flush=True)
