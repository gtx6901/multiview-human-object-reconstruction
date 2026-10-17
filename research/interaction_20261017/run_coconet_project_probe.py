"""Real CoCoNet forward on a project clip using explicit custom-data adapters."""
import os
from pathlib import Path
import sys,os,json,time,ast,types,hashlib
from cari_runtime import configure,ROOT,CARI,ASSETS,make_smplh
configure()
import torch,numpy as np,trimesh,kornia
from PIL import Image
from cari_utils_runtime import load
sys.path.insert(0,os.environ["HOI_WARP_ASCII_ROOT"])
import warp as wp
wp.config.kernel_cache_dir=os.path.join(os.environ["HOI_WARP_ASCII_ROOT"], "kernel_cache")
Utils=load()
import nvdiffrast.torch as dr
from lib_smpl.body_landmark import BodyLandmarks
out=ROOT/sys.argv[1];out.mkdir(exist_ok=False);start=time.time();serial=sys.argv[2] if len(sys.argv)>2 else '21305717'
torch.set_num_threads(4)
checkpoint=torch.load(ROOT/'weights/step031397.pth',map_location='cpu',weights_only=False);cfg=checkpoint['cfg']
official_hub=torch.hub.load
def local_dino(repo,name,*args,**kwargs):
    assert repo=='facebookresearch/dinov2';kwargs['pretrained']=False;kwargs['source']='local'
    return official_hub(str(ROOT/'vendor/dinov2'),name,*args,**kwargs)
torch.hub.load=local_dino
from learning.models import get_model
model=get_model(cfg);loaded=model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
model.eval().cuda();assert not model.training
# Execute the exact upstream methods without instantiating dataset/video loaders
# that require unrelated BEHAVE paths. Their ASTs and operators are unmodified.
namespace={'torch':torch,'np':np,'os':os,'Utils':Utils,'kornia':kornia}
method_sources=[]
def methods(file,classname,names):
    path=CARI/file;source=path.read_text();tree=ast.parse(source);cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==classname)
    selected=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(selected)==len(names)
    isolated=ast.Module(body=[ast.ClassDef(name=classname,bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(isolated);exec(compile(isolated,str(path),'exec'),namespace)
    method_sources.append({'file':file,'sha256':hashlib.sha256(source.encode()).hexdigest(),'methods':names})
    return namespace[classname]
Dataset=methods('learning/datasets/video_data.py','VideoDataset',['process_input','get_one_channel_mask'])
Crop=methods('prep/prerender_behave.py','BehaveRenderer',['crop_color_dmap','Kroi_from_corners'])
processor=Dataset();processor.cfg=cfg;processor.split='test';processor.render_h5_root='project-independent-render'
data=ROOT/'helmet_contiguous180_211_v1';meta=json.loads((data/'manifest.json').read_text());v=next(x for x in meta['views'] if x['serial']==serial)
K=np.array(v['K']);W=np.array(v['W']);size=224;cropper=Crop();cropper.K_full=K;cropper.focal=K.diagonal()[:2];cropper.principal_point=K[:2,2];cropper.rend_size=size
offset=int(sys.argv[3]) if len(sys.argv)>3 else 0
frames=meta['sample_frames'][offset:offset+16];assert len(frames)==16;landmark=BodyLandmarks(str(ASSETS));body_model=make_smplh().cuda()
human_records=json.loads((ROOT/'people_contiguous180_211_v1/report.json').read_text())['records'];obj_records=json.loads((ROOT/'helmet_track180_211_v2/report.json').read_text())['records']
motion=torch.load(ROOT/'helmet_hull180_v2/sequence.pt',weights_only=True);mesh=torch.load(ROOT/'helmet_hull_colors180_v1/mesh.pt',weights_only=True)
alpha=json.loads((ROOT/'nlf_depth_world_alignment180_211_v1/report.json').read_text())['common_model_to_world_scale']
fitted=[torch.load(ROOT/'nlf_depthaligned180_211_v5'/f'{serial}_{f:04d}.pt',weights_only=True) for f in frames]
translations=np.stack([p['trans'][0].numpy() for p in fitted]);betas=np.stack([p['shape_betas'][0].numpy() for p in fitted])
with torch.inference_mode():
    zero=body_model(torch.zeros(1,156,device='cuda'),torch.tensor(betas.mean(0)[None],device='cuda'),torch.zeros(1,3,device='cuda'))[0][0].cpu().numpy()
    np.random.seed(0);diameter=Utils.compute_mesh_diameter(model_pts=trimesh.Trimesh(zero,body_model.faces).sample(8000),n_sample=8000)
glctx=dr.RasterizeCudaContext();inputs=[];renders=[];xyzs=[];rxyzs=[];poses=[];joints=[];sheet=Image.new('RGB',(size*3,size*4),'black')
with torch.inference_mode():
 for ti,f in enumerate(frames):
    h=next(x for x in human_records if x['serial']==serial and x['frame']==f);o=next(x for x in obj_records if x['serial']==serial and x['frame']==f)
    hm=np.asarray(Image.open(ROOT/'people_contiguous180_211_v1'/h['mask']))>127;om=np.asarray(Image.open(ROOT/'helmet_track180_211_v2'/o['mask']))>127
    yy,xx=np.where(hm|om);bmin=np.array([xx.min(),yy.min()]);bmax=np.array([xx.max(),yy.max()]);center=(bmin+bmax)/2;radius=np.max(bmax-bmin)*1.1/2;tl=center-radius;br=center+radius
    kroi=cropper.Kroi_from_corners(br,tl);rgb=np.asarray(Image.open(data/serial/f'{f:04d}.png').convert('RGB'));rgbm=np.concatenate([rgb,hm[...,None].astype('uint8')*255,om[...,None].astype('uint8')*255],-1)
    depth=np.load(ROOT/'unidepth_contiguous180_211_v2'/serial/f'{f:04d}.npz')['depth'];xyz,rgbm_crop=cropper.crop_color_dmap(np.r_[tl,br],rgbm,depth*1000,(size,size))
    body=fitted[ti]['vertices'][0].numpy();joints.append(landmark.get_body_kpts_batch(body[None])[0])
    mi=offset+ti
    object_world=mesh['vertices_local'].numpy()@motion['R'][mi].numpy().T*float(motion['logscale'].exp())+motion['translation_world'][mi].numpy();object_cam=(object_world@W[:3,:3].T+W[:3,3])/alpha
    pose=np.eye(4,dtype='float32');pose[:3,:3]=W[:3,:3]@motion['R'][mi].numpy();pose[:3,3]=(W[:3,:3]@motion['translation_world'][mi].numpy()+W[:3,3])/alpha;poses.append(pose)
    combined=trimesh.Trimesh(np.concatenate([body,object_cam]),np.concatenate([body_model.faces,mesh['faces'].numpy()+len(body)]),process=False)
    combined.visual.vertex_colors=np.uint8(np.concatenate([np.full_like(body,.7),mesh['colors'].numpy()])*255)
    objmesh=trimesh.Trimesh(object_cam,mesh['faces'].numpy(),process=False);objmesh.visual.vertex_colors=np.uint8(mesh['colors'].numpy()*255)
    rendered=[]
    for geom in [combined,objmesh]:
        tensors=Utils.make_mesh_tensors(geom);rr,dd,_=Utils.nvdiffrast_render(K=kroi[None],H=size,W=size,ob_in_cams=torch.eye(4,device='cuda')[None],glctx=glctx,mesh_tensors=tensors,output_size=(size,size),use_light=True)
        rendered.append((np.uint8(rr[0].cpu().numpy()*255),dd[0].cpu().numpy()))
    render_rgb,render_depth=rendered[0];object_depth=rendered[1][1];masks=np.stack([(object_depth<=render_depth)&(object_depth>0),object_depth>0],-1)
    rb=Utils.depth2xyzmap(render_depth,kroi);input_data={'rgbmB':rgbm_crop.numpy().astype('uint8'),'xyzB':xyz.numpy().astype('float16')};render_data={'mask_o':masks,'K_roi':kroi}
    xb,xa,color=processor.process_input(torch.tensor(rb).permute(2,0,1),ti,input_data,diameter,translations,pose,render_data,render_rgb,os.path.join('project',f'{f:04d}'),0)
    inputs.append(color);renders.append(torch.tensor(render_rgb/255).permute(2,0,1).float());xyzs.append(xb);rxyzs.append(xa)
    if ti%4==0:
        row=ti//4;original=input_data['rgbmB'][...,:3];overlay=np.uint8(original*.5+render_rgb*.5)
        for col,arr in enumerate([original,render_rgb,overlay]):sheet.paste(Image.fromarray(arr),(col*size,row*size))
    print('prepared',serial,f,flush=True)
sheet.save(out/'actual_inputs_render_overlay.jpg')
T=len(frames);pose_tensor=torch.tensor(np.array(poses),device='cuda')[None];pose_norm=pose_tensor.clone();pose_norm[:,:,:3,3]*=2/diameter
A=torch.cat([torch.stack(renders),torch.stack(rxyzs)],1)[None].cuda().float();B=torch.cat([torch.stack(inputs),torch.stack(xyzs)],1)[None].cuda().float()
batch={'joints_nlf':torch.tensor(np.array(joints),device='cuda').float()[None],'mesh_diameter':torch.full((1,T),diameter,device='cuda'),'pose_perturbed':pose_tensor,'render_xyz':torch.stack(rxyzs)[None].cuda().float()}
assert A.shape==B.shape==(1,16,9,224,224) and torch.isfinite(A).all() and torch.isfinite(B).all()
torch.save({'A':A.cpu(),'B':B.cpu(),'poseA_norm':pose_norm.cpu(),'batch':{k:x.cpu() for k,x in batch.items()},'frames':frames},out/'input_batch.pt')
with torch.inference_mode():
    output=model(A,B,pose_norm,batch)
assert output and all(torch.isfinite(x).all() for x in output.values() if isinstance(x,torch.Tensor))
torch.save({k:x.detach().cpu() for k,x in output.items()},out/'raw_predictions.pt')
report={'success':True,'serial':serial,'frames':frames,'strict_missing':loaded.missing_keys,'strict_unexpected':loaded.unexpected_keys,'input_shape':list(A.shape),'output_shapes':{k:list(x.shape) for k,x in output.items()},'seconds':time.time()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'upstream_methods':method_sources,'model_to_world_scale':alpha,'scope':'Actual official CoCoNet checkpoint and unmodified model forward on real 16-frame project observations. Custom all8 silhouette object initialization and shared scale replace FoundationPose; bounded hull geometry is unaccepted. Actual upstream crop/process_input and nvdiffrast vertex-color rendering; neutral gray human vertices replace official UV template appearance, observed object vertex colors replace UV texture. Float32 inference. Raw predictions only, not full official CARI4D run, accepted contact or shared eight-camera refined sequence.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('actual project forward complete',flush=True)
