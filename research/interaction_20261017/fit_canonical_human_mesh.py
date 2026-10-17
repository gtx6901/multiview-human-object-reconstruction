"""SMPL-X topology offsets distilled from frozen avatar A-pose silhouettes."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
from PIL import Image,ImageDraw
from smplx.lbs import batch_rigid_transform
from avatar_rig import pose_axis_angles
from pytorch3d.transforms import axis_angle_to_matrix
from mesh_renderer import MeshRenderer
p=argparse.ArgumentParser();p.add_argument('--out',default='human_mesh_canonical_v2');p.add_argument('--steps',type=int,default=300);options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
source=ROOT/'avatar_canonical_views_v2';assert json.loads((source/'report.json').read_text())['success']
pose=torch.load(source/'pose.pt',weights_only=True);camera=json.loads((source/'cameras.json').read_text());width=camera['width'];height=camera['height']
path=ROOT.parent/'lhmpp_20261016/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz';d=np.load(path,allow_pickle=True)
template=torch.tensor(d['v_template'],device='cuda',dtype=torch.float32);weights=torch.tensor(d['weights'],device='cuda',dtype=torch.float32);Jreg=torch.tensor(d['J_regressor'],device='cuda',dtype=torch.float32);rest=Jreg@template
parent=torch.tensor(d['kintree_table'][0].astype('int64'),device='cuda');parent[0]=-1
faces=torch.tensor(d['f'].astype('int32'),device='cuda');aa=pose_axis_angles({k:v.cuda() for k,v in pose.items()},0,'cuda');rot=axis_angle_to_matrix(aa)
_,A=batch_rigid_transform(rot,rest[None],parent);T=(weights@A[0].reshape(55,16)).reshape(-1,4,4)
posedirs=torch.tensor(d['posedirs'],device='cuda',dtype=torch.float32).reshape(-1,486).T.contiguous();poseoffset=((rot[:,1:]-torch.eye(3,device='cuda')).reshape(1,486)@posedirs).reshape(-1,3)
posedbase=(T[:,:3,:3]@(template+poseoffset)[...,None])[...,0]+T[:,:3,3]
frozen=torch.zeros(len(template),dtype=torch.bool,device='cuda');faceids=np.load(path.parent/'SMPL-X__FLAME_vertex_ids.npy');frozen[torch.tensor(faceids.astype('int64'),device='cuda')]=True
frozen|=(weights[:,[20,21]+list(range(25,55))].sum(1)>.2)
raw=torch.nn.Parameter(torch.zeros_like(template));scale=pose['scale'].cuda()
edges=trimesh.Trimesh(d['v_template'],d['f'],process=False).edges_unique
src=np.r_[edges[:,0],edges[:,1]];dst=np.r_[edges[:,1],edges[:,0]];degree=np.bincount(src,minlength=len(template))
idx=torch.tensor(np.stack([src,dst]),device='cuda');values=torch.tensor(1/degree[src],device='cuda',dtype=torch.float32)
neighbor=torch.sparse_coo_tensor(idx,values,(len(template),len(template))).coalesce()
fg=faces.long();tri=template[fg];reference_cross=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);reference_squared=reference_cross.square().sum(1).clamp_min(1e-20)
edgeids=torch.tensor(edges,device='cuda');reference_lengths=(template[edgeids[:,1]]-template[edgeids[:,0]]).norm(dim=1).clamp_min(1e-8)
active_faces=~frozen[fg].all(1);backtracked=0
def orientation(delta):
    triangles=(template+delta)[fg];cross=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    return (cross*reference_cross).sum(1)/reference_squared
K=torch.tensor([v['K'] for v in camera['views']],device='cuda',dtype=torch.float32);W=torch.tensor([v['W'] for v in camera['views']],device='cuda',dtype=torch.float32)
target=torch.tensor(np.stack([np.asarray(Image.open(source/v['alpha'])).astype('float32')/255 for v in camera['views']]),device='cuda')
renderer=MeshRenderer();chunks=list(torch.arange(len(K),device='cuda').split(8));history=[];best=None
def geometry():
    delta=.08*torch.tanh(raw)*(~frozen)[:,None]
    world=(posedbase+(T[:,:3,:3]@delta[...,None])[...,0])*scale
    return world,delta
def silhouette_loss(world,ids):
    alpha=renderer.alpha_batch(world[None].expand(len(ids),-1,-1),faces,K[ids],W[ids],width,height)
    intersection=(alpha*target[ids]).sum((1,2));union=(alpha+target[ids]-alpha*target[ids]).sum((1,2)).clamp_min(1)
    return 1-intersection/union,alpha
with torch.no_grad():
    world,_=geometry();before=torch.cat([1-silhouette_loss(world,ids)[0] for ids in chunks]).cpu()
opt=torch.optim.Adam([raw],lr=.025)
for step in range(options.steps+1):
    opt.zero_grad();data=0.
    for ids in chunks:
        world,delta=geometry();loss,_=silhouette_loss(world,ids);chunk=loss.sum()/len(K);data+=float(chunk.detach())
        if step<options.steps:chunk.backward()
    _,delta=geometry();lap=delta-torch.sparse.mm(neighbor,delta);ratio=orientation(delta)
    vertex=template+delta;lengths=(vertex[edgeids[:,1]]-vertex[edgeids[:,0]]).norm(dim=1).clamp_min(1e-8)
    regularization=1000*lap.square().mean()+2*delta.square().mean()+.01*(lengths/reference_lengths).log().square().mean()+5*torch.relu(.25-ratio[active_faces]).square().mean()
    value=data+float(regularization.detach())
    if best is None or value<best[0]:best=(value,raw.detach().clone())
    if step<options.steps:
        regularization.backward();old_raw=raw.detach().clone();opt.step()
        # Keep the canonical surface in its original orientation after EVERY
        # step; an input-IoU gain cannot justify exporting inverted triangles.
        with torch.no_grad():
            damped=False
            for attempt in range(16):
                _,candidate=geometry()
                if bool((orientation(candidate)>.05).all()):break
                raw.copy_((raw+old_raw)*.5);damped=True
            else:raw.copy_(old_raw)
            backtracked+=int(damped)
    for group in opt.param_groups:group['lr']=.025*(1-.8*step/options.steps)
    if step%50==0:
        row={'step':step,'loss':value,'synthetic_view_soft_iou':1-data};history.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    raw.copy_(best[1]);world,delta=geometry();after=torch.cat([1-silhouette_loss(world,ids)[0] for ids in chunks]).cpu();assert torch.count_nonzero(delta[frozen])==0 and torch.isfinite(delta).all()
    ratio=orientation(delta);assert (ratio>.05).all()
    torch.save({'offsets_model_space':delta.cpu(),'vertices_template':template.cpu(),'vertices_canonical':(template+delta).cpu(),'vertices_Apose_world':world.cpu(),'faces':faces.cpu(),'official_uv_vertices':torch.tensor(d['vt']),'official_uv_faces':torch.tensor(d['ft'].astype('int64')),'weights':weights.cpu(),'rest_joints':rest.cpu(),'parents':parent.cpu(),'frozen_vertices':frozen.cpu(),'source_pose':pose,'scale':scale.cpu()},out/'mesh.pt')
    # OBJ keeps separate per-loop official UV indices without changing10475 vertices.
    with (out/'human_canonical.obj').open('w',encoding='utf-8') as f:
        for v in (template+delta).cpu().numpy():f.write('v '+' '.join(f'{x:.9g}' for x in v)+'\n')
        for uv in d['vt']:f.write('vt '+' '.join(f'{x:.9g}' for x in uv)+'\n')
        for tri,uv in zip(d['f'],d['ft']):f.write('f '+' '.join(f'{int(v)+1}/{int(t)+1}' for v,t in zip(tri,uv))+'\n')
    sheet=Image.new('RGB',(6*336,2*356),'white')
    for col,index in enumerate([0,4,8,12,16,20]):
        for row,points in enumerate([posedbase*scale,world]):
            alpha=renderer.alpha_batch(points[None],faces,K[index:index+1],W[index:index+1],width,height)[0].cpu().numpy();image=np.stack([target[index].cpu().numpy(),alpha,np.zeros_like(alpha)],-1)
            pic=Image.fromarray(np.uint8(image.clip(0,1)*255)).resize((336,336));sheet.paste(pic,(col*336,row*356+20));ImageDraw.Draw(sheet).text((col*336+3,row*356+2),('BEFORE' if row==0 else 'AFTER')+f' az{camera["views"][index]["azimuth"]} targetR meshG',fill='black')
    sheet.save(out/'silhouette_fit.jpg')
    report={'success':True,'seconds':time.time()-start,'synthetic_views':len(K),'history':history,'mean_synthetic_soft_iou_before':float(before.mean()),'mean_synthetic_soft_iou_after':float(after.mean()),'vertices':len(template),'faces':len(faces),'frozen_face_hand_vertices':int(frozen.sum()),'max_offset_model_units':float(delta.norm(dim=1).max()),'mean_offset_model_units':float(delta.norm(dim=1).mean()),'scope':'Original10475-vertex SMPL-X topology and official UV preserved; canonical bounded per-vertex offsets fitted against72 actual frozen-avatar synthetic A-pose alpha views with uniform Laplacian/magnitude regularization. Face/head FLAME vertices and hand-weighted vertices remain EXACTLY zero offset. Original rig weights/rest skeleton unchanged. Synthetic agreement is distillation, not independent real-image reconstruction quality. No UV color/rigged Blender import/export verified yet; private licensed mesh arrays must not enter public source release.'}
    report.update(canonical_inverted_triangles=int((ratio<=0).sum()),min_canonical_oriented_area_ratio=float(ratio.min()),backtracked_optimization_steps=backtracked)
    report['scope']+=' Additional edge-length/oriented-area penalties and step backtracking prevent canonical triangle inversions; this is not a proof of no global self-intersection or validity under all animations. V1 had982 orientation reversals and is not accepted as final geometry.'
    (out/'report.json').write_text(json.dumps(report,indent=2));print('canonical human mesh fitted',flush=True)
