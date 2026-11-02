"""Official HaMeR image-conditioned rotation diagnostic, all8 cameras.

Only original predictor modules are executed; training/render package imports
are avoided. No MANO surface model is downloaded or used, and no inferred beta,
depth or palm orientation is copied into the MJJ identity/body. Predictions are
observations, not accepted reconstruction or an optimization objective.
"""
import argparse,json,sys,types,time
import cv2,numpy as np,torch
from PIL import Image
import calibrate as c
from reconstruct import OUT,move
from motion import configure

def namespaces(source):
    # These package initializers import Lightning/training/OpenGL rendering.
    # The actual backbone/head/dataset source modules remain unchanged.
    for name in ['hamer','hamer.models','hamer.models.heads','hamer.models.components','hamer.utils','hamer.datasets']:
        module=types.ModuleType(name);module.__path__=[str(source.joinpath(*name.split('.')))];sys.modules[name]=module

def load():
    configure();torch.set_num_threads(4);cv2.setNumThreads(4);root=c.ROOT/'work/mjj/runtime/hamer';provenance=json.loads((root/'source.json').read_text());checkpoint=root/'_DATA/hamer_ckpts/checkpoints/hamer.ckpt';assert c.digest(checkpoint)==next(x['sha256'] for x in provenance['files'] if x['path'].endswith('hamer.ckpt'));namespaces(root/'source')
    from yacs.config import CfgNode
    from accelerate import init_empty_weights
    from hamer.models.backbones import create_backbone
    from hamer.models.heads.mano_head import build_mano_head
    cfg=CfgNode.load_cfg((root/'_DATA/hamer_ckpts/model_config.yaml').read_text());cfg.MODEL.BBOX_SHAPE=[192,256];cfg.MANO.MEAN_PARAMS=str(root/'_DATA/data/mano_mean_params.npz');cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None);cfg.freeze()
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=create_backbone(cfg);self.mano_head=build_mano_head(cfg)
        def forward(self,x):return self.mano_head(self.backbone(x[:,:,:,32:-32]))
    with init_empty_weights(include_buffers=False):model=Predictor()
    # The official, hash-verified author checkpoint contains Yacs training
    # metadata; only the strict two predictor state prefixes are used.
    state=torch.load(checkpoint,weights_only=False,map_location='cpu')['state_dict'];selected={k:v for k,v in state.items() if k.startswith(('backbone.','mano_head.'))};model.load_state_dict(selected,strict=True,assign=True);assert not any(p.is_meta for p in model.parameters());model.eval().requires_grad_(False);report={'status':'strict_predictor_load_only_no_inference','source_commit':provenance['source_commit'],'checkpoint_sha256':c.digest(checkpoint),'selected_tensor_count':len(selected),'unused_state_prefixes':sorted({k.split('.')[0] for k in state if k not in selected}),'parameter_count':sum(p.numel() for p in model.parameters()),'exact_forward_scope':'Official HAMER.forward_step backbone256x192 crop and MANO head, unchanged original modules. No training losses/discriminator/renderer/MANO surface path executed.'};(root/'load_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return model,cfg,root

def run(frames=(8,55,70,84),load_only=False):
    model,cfg,root=load()
    if load_only:return
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from pytorch3d.transforms import matrix_to_axis_angle
    model.cuda();dest=OUT/'refinement/hand_surface/image_conditioned';dest.mkdir(exist_ok=True);kp=np.load(c.OUT/'keypoints_crop_dynamic.npz');source=OUT/'refinement/body/contract_fit/full_context/expression/continuous';m=torch.load(source/'motion.pt',weights_only=True,map_location='cpu');records=[];mirror=torch.diag(torch.tensor([-1.,1.,1.],device='cuda'));start=time.time()
    with torch.inference_mode():
        for t in frames:
            sid=int(m['source_frames'][t])
            for ci,cam in enumerate(kp['cams']):
                cam=str(cam);path=c.ROOT/f'assets/mjj/dynamic/{cam}/{sid:04d}.jpg';raw=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR);boxes=[];sides=[]
                for side in range(2):
                    ids=np.arange(91+21*side,112+21*side);good=(kp['score'][ci,sid,ids]>.5)&np.isfinite(kp['xy'][ci,sid,ids]).all(-1)
                    if good.sum()<=3:continue
                    xy=kp['xy'][ci,sid,ids][good];boxes.append(np.r_[xy.min(0),xy.max(0)]);sides.append(side)
                if not boxes:continue
                dataset=ViTDetDataset(cfg,raw,np.array(boxes),np.array(sides),rescale_factor=2.5)
                for i,side in enumerate(sides):
                    item=dataset[i];x=torch.as_tensor(item['img'],device='cuda',dtype=torch.float32)[None];params,weakcam,_=model(x);R=params['hand_pose'].reshape(15,3,3);mapped=mirror@R@mirror if side==0 else R;error=float((mapped.transpose(-1,-2)@mapped-torch.eye(3,device='cuda')).abs().max());assert error<1e-5 and (torch.linalg.det(mapped)>.9999).all();normal=x[0].permute(1,2,0).cpu().numpy();patch=np.uint8(np.clip(normal*np.array(cfg.MODEL.IMAGE_STD)+np.array(cfg.MODEL.IMAGE_MEAN),0,1)*255);Image.fromarray(patch).save(dest/f'{cam}_{t:04d}_hand{side}_input.png');records.append({'camera':cam,'native_frame':t,'source_frame':sid,'side':side,'image_sha256':c.digest(path),'source_box':boxes[i].tolist(),'box_center':item['box_center'].tolist(),'box_size':float(item['box_size']),'predicted_mano':move(params,'cpu'),'weak_camera':weakcam.cpu(),'smplx_local_finger_rotations':mapped.cpu(),'smplx_local_finger_axis_angle':matrix_to_axis_angle(mapped).cpu(),'rotation_orthogonality_max':error});print('Image-conditioned hand',t,cam,side,flush=True)
    torch.save({'predictions':records,'source_motion_sha256':c.digest(source/'motion.pt'),'predictor_source':json.loads((root/'load_report.json').read_text())},dest/'observations.pt');(dest/'report.json').write_text(json.dumps({'status':'raw_observations_require_geometry_and_visual_validation_not_adopted','frames':list(frames),'prediction_count':len(records),'seconds':time.time()-start,'scope':'Original HaMeR full-image RGB hand crops, original256/2.5/192x256 preprocessing and >.5/>3 bbox candidate rule; all8 cameras eligible. Left input flip and SRS local rotation mirror, absolute MANOLayer rotations with no PCA or mean addition. No monocular depth, beta, global palm orientation, new subject identity or selected input camera adopted. Original source images stay local. Validate actual SMPL-X finger/side convention and render every input hypothesis before any consensus/fitting.'},indent=2));print('Raw hand image observations saved, no adopted pose',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frames',type=int,nargs='+',default=[8,55,70,84]);p.add_argument('--all',action='store_true');p.add_argument('--load-only',action='store_true');a=p.parse_args();run(list(range(99)) if a.all else a.frames,a.load_only)
