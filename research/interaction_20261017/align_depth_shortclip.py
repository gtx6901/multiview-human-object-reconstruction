"""CARI depth alignment algorithm with explicit NumPy/SciPy ICP compatibility."""
import os
from pathlib import Path
import sys,json,time
from cari_runtime import configure,ROOT,SMPLH,make_smplh
configure()
import torch,numpy as np,trimesh
from PIL import Image
from scipy.spatial import cKDTree
from smplfitter.pt import BodyModel,BodyFitter
from cari_utils_runtime import load
warp_ascii=Path(os.environ["HOI_WARP_ASCII_ROOT"])
assert (warp_ascii/'warp/native/builtin.h').read_bytes()==(ROOT/'deps/warp/native/builtin.h').read_bytes()
sys.path.insert(0,str(warp_ascii))
import warp as wp
wp.config.kernel_cache_dir=str(warp_ascii/'kernel_cache')
Utils=load();assert Utils.wp is not None,'Actual Warp kernels required'
assert hasattr(Utils,'erode_depth') and hasattr(Utils,'bilateral_filter_depth')
filter_device='cuda' if wp.get_cuda_devices() else 'cpu'
torch.set_num_threads(4)
out=ROOT/sys.argv[1];out.mkdir(exist_ok=False);start=time.time()
def voxel(points,size):
    origin=points.min(0)-size*.5;keys=np.floor((points-origin)/size).astype('int64')
    _,inverse=np.unique(keys,axis=0,return_inverse=True);counts=np.bincount(inverse)
    return np.stack([np.bincount(inverse,weights=points[:,j])/counts for j in range(3)],1)
def icp(source,target,iters=(15,15,15)):
    source=source.copy();translation=np.zeros(3);steps=[]
    for iterations,radius in zip(iters,[.08,.04,.01]):
        src=voxel(source,.01);tgt=voxel(target,.01);tree=cKDTree(tgt);stage=np.zeros(3)
        for i in range(iterations):
            distances,indices=tree.query(src,k=1);valid=distances<radius*2
            if not valid.any():break
            delta=np.array([0.,0.,(tgt[indices[valid],2]-src[valid,2]).mean()])
            if np.linalg.norm(delta)<.001:break
            src+=delta;stage+=delta;translation+=delta
        source+=stage;steps.append({'radius':radius,'inliers':int(valid.sum()),'z_translation':float(stage[2])})
    return translation,steps
# Known z displacement: distinguishes genuine z-only alignment from pass-through.
rng=np.random.default_rng(83);probe=rng.normal(0,.08,(2000,3));probe[:,2]+=3
shift,checks=icp(probe,probe+np.array([0.,0.,.04]));assert abs(shift[2]-.04)<.0015 and np.abs(shift[:2]).max()==0
model=BodyModel('smplh','male',model_root=str(SMPLH),num_betas=10);fitter=BodyFitter(model).eval();official=make_smplh().eval();faces=official.faces
meta=json.loads((ROOT/'helmet_contiguous180_211_v1/manifest.json').read_text());people=ROOT/'people_contiguous180_211_v1'
records={(p['serial'],p['frame']):p for p in json.loads((people/'report.json').read_text())['records']};results=[]
with torch.inference_mode():
 for v in meta['views']:
  serial=v['serial'];K=np.array(v['K'])
  for frame in meta['sample_frames']:
    filename=f'{serial}_{frame:04d}.pt';pred=torch.load(ROOT/'nlf_smplh_contiguous180_211_v1'/filename,weights_only=True);vertices=pred['vertices'][0].numpy()
    depth=np.load(ROOT/'unidepth_contiguous180_211_v2'/serial/f'{frame:04d}.npz')['depth'].astype('float32')
    depth=Utils.bilateral_filter_depth(Utils.erode_depth(depth,radius=2,device=filter_device),radius=2,device=filter_device)
    xyz=Utils.depth2xyzmap(depth,K);mask=np.asarray(Image.open(people/records[(serial,frame)]['mask']))>127
    target=xyz[mask&(depth>.001)&np.isfinite(depth)];assert len(target)>1000
    np.random.seed(1000+frame)
    points=trimesh.Trimesh(vertices,faces,process=False).sample(8000)
    dz=np.median(target[:,2])-np.median(points[:,2]);moved=points+np.array([0,0,dz]);first,steps1=icp(moved,target,(25,10,5))
    scale=(vertices[:,2].mean()+dz+first[2])/vertices[:,2].mean();assert np.isfinite(scale) and scale>0
    dz2=np.median(target[:,2])-np.median(points[:,2])*scale
    moved=points*scale+np.array([0,0,dz2]);second,steps2=icp(moved,target)
    aligned=vertices*scale+np.array([0,0,dz2])+second;fallback=aligned[:,2].mean()<1
    if fallback:aligned=vertices
    fit=fitter.fit(torch.tensor(aligned,dtype=torch.float32)[None],num_iter=3,requested_keys=['shape_betas','trans','pose_rotvecs'],beta_regularizer=1)
    verts=official(fit['pose_rotvecs'],fit['shape_betas'],fit['trans'])[0];assert torch.isfinite(verts).all()
    torch.save({**{k:x.cpu() for k,x in fit.items()},'vertices':verts.cpu(),'aligned_target_vertices':torch.tensor(aligned,dtype=torch.float32)[None]},out/filename)
    record={'serial':serial,'frame':frame,'depth_cloud_points':len(target),'scale':float(scale),'initial_z_shift':float(dz),'final_z_shift':float(dz2+second[2]),'fallback_original':bool(fallback),'fitter_mean_vertex_distance_m':float(torch.linalg.vector_norm(verts-torch.tensor(aligned,dtype=torch.float32),dim=-1).mean()),'icp_first':steps1,'icp_second':steps2};results.append(record)
    print(serial,frame,'scale',round(float(scale),4),flush=True)
report={'success':True,'filter_device':filter_device,'records':results,'seconds':time.time()-start,'probe_z_shift':shift.tolist(),'scope':'Actual cached NLF and UniDepth outputs, original calibrated K, real official Warp erosion/bilateral kernels and SMPL-H fitter. CARI two-stage z-only ICP/scale sequence reproduced in NumPy/SciPy with Open3D-style voxel centroids and exact cKDTree nearest neighbors; not a claim of bit-identical Open3D/PyTorch3D execution. Deterministic 8000 surface samples. Independent per-view learned depth alignment, no measured metric truth or all8 shared output yet. No CoCoNet forward.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('complete',len(results),flush=True)
