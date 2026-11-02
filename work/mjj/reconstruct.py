"""Gated MJJ static fit, tensor rig export, and eight-camera inspection.

All outputs live in work/mjj/reconstruction. Source assets and FCS are read only.
"""
from pathlib import Path
import sys,os,json,argparse,importlib.util
import numpy as np
import calibrate as c

OUT=c.ROOT/'work/mjj/reconstruction';LHM=c.ROOT/'lhmpp'


def gate():
    r=json.loads((c.OUT/'acceptance.json').read_text())
    if not r['accepted_for_static_fit'] or r['failures']:raise RuntimeError('Calibration gate is closed')
    p=c.ROOT/r['candidate']
    if c.digest(p)!=r['candidate_sha256']:raise RuntimeError('Accepted camera hash changed')
    for name,h in r['validation_input_sha256'].items():
        if c.digest(c.OUT/name)!=h:raise RuntimeError(f'Validation input changed: {name}')
    return r,np.load(p)


def model_load():
    import torch
    torch.set_num_threads(4);sys.dont_write_bytecode=True;sys.path.insert(0,str(LHM))
    source=LHM/'load_model_probe.py';text=source.read_text();old='out.mkdir(parents=True,exist_ok=False)'
    assert text.count(old)==1
    # Same strict loader; only its report directory is reusable inside work/.
    text=text.replace(old,'out.mkdir(parents=True,exist_ok=True)')
    sys.argv=[str(source),'--name',str(OUT/'model_load')]
    scope={'__name__':'__main__','__file__':str(source)}
    exec(compile(text,str(source),'exec'),scope)
    return scope['model']


def move(x,device):
    import torch
    if isinstance(x,torch.Tensor):return x.detach().to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [move(v,device) for v in x]
    return x


def fit_static(model,cal):
    import torch
    from smplx.lbs import batch_rigid_transform
    from pytorch3d.transforms import matrix_to_rotation_6d,rotation_6d_to_matrix,matrix_to_axis_angle
    torch.manual_seed(20261027)
    kp=np.load(c.OUT/'keypoints_crop_static.npz');obs=np.load(c.OUT/'candidate_balanced_a_observations_static.npz')
    if list(kp['cams'])!=list(cal['cams']):raise RuntimeError('Serial ordering changed')
    idx=list(kp['frames']).index(50);smpl=[16,17,18,19,20,21,1,2,4,5,7,8]
    with torch.no_grad():rest=model.renderer.smplx_model.get_zero_pose_human(torch.zeros(1,10),torch.device('cpu'),None,None).detach()
    parents=model.renderer.smplx_model.smplx_layer.parents.cpu()
    X=obs['xyz'][idx];mask=np.isfinite(X).all(1);core=np.array([0,1,6,7,10,11]);core=core[mask[core]]
    src=rest[0,smpl].numpy();a,b=src[core],X[core];ma,mb=a.mean(0),b.mean(0)
    u,s,vt=np.linalg.svd((a-ma).T@(b-mb));sign=np.eye(3);sign[2,2]=np.linalg.det(vt.T@u.T);R=vt.T@sign@u.T
    scale=(s*np.diag(sign)).sum()/np.square(a-ma).sum();trans=mb-scale*(ma@R.T)
    trans+=scale*(rest[0,0].numpy()@R.T-rest[0,0].numpy())
    root=torch.nn.Parameter(matrix_to_rotation_6d(torch.tensor(R,dtype=torch.float32)))
    identity=matrix_to_rotation_6d(torch.eye(3));body=torch.nn.Parameter(identity.repeat(21,1))
    logs=torch.nn.Parameter(torch.tensor(np.log(scale),dtype=torch.float32));tr=torch.nn.Parameter(torch.tensor(trans/scale,dtype=torch.float32))
    W=torch.tensor(cal['W'],dtype=torch.float32);K=torch.tensor(cal['K'],dtype=torch.float32)
    xy=torch.tensor(kp['xy'][:,idx,c.JOINTS],dtype=torch.float32);valid=torch.tensor(obs['inlier'][:,idx])
    target=torch.tensor(np.nan_to_num(X),dtype=torch.float32);good=torch.tensor(mask)
    def evaluate():
        rotations=torch.cat([rotation_6d_to_matrix(root)[None],rotation_6d_to_matrix(body),torch.eye(3).repeat(33,1,1)])[None]
        joints,_=batch_rigid_transform(rotations,rest,parents);sc=logs.exp();world=(joints[0,smpl]+tr)*sc
        pc=torch.einsum('cij,nj->cni',W[:,:3,:3],world)+W[:,:3,3][:,None]
        uv=torch.einsum('cij,cnj->cni',K,pc);uv=uv[...,:2]/uv[...,2:].clamp_min(1e-5)
        err=torch.linalg.norm(uv-xy,dim=-1);robust=2*(torch.sqrt(1+(err/4)**2)-1)
        d=torch.linalg.norm((world-target)/sc,dim=-1);loss3=(torch.sqrt(1+(d[good]/.02)**2)-1).mean()
        aa=matrix_to_axis_angle(rotations[0,1:22]);prior=aa.square().mean()
        loss=robust[valid].mean()+.1*loss3+.02*prior
        return loss,err,uv,world,rotations
    opt=torch.optim.Adam([root,logs,tr],lr=.015)
    for stage,steps in [(0,201),(1,1001)]:
        if stage:opt=torch.optim.Adam([root,body,logs,tr],lr=.007)
        for step in range(steps):
            opt.zero_grad();loss,err,*_=evaluate();loss.backward();opt.step()
            if step%200==0:print('static',stage,step,'loss',float(loss),'median_px',float(err[valid].median()),flush=True)
    with torch.no_grad():
        loss,err,uv,world,rots=evaluate();aa=matrix_to_axis_angle(rots[0]);poses={'root_pose':aa[None,0],'body_pose':aa[None,1:22],
            'scale':logs.exp().detach().reshape(1),'translation_world':(tr*logs.exp()).detach()[None],
            'betas':torch.zeros(1,10),'frames':torch.tensor([50]),'fps':100/11}
        torch.save(move(poses,'cpu'),OUT/'static_pose.pt');torch.save({'rest':rest,'parents':parents},OUT/'skeleton.pt')
        np.savez_compressed(OUT/'static_joints.npz',world=world.numpy(),uv=uv.numpy(),inlier=valid.numpy(),target=xy.numpy())
        rawvalid=(kp['score'][:,idx,c.JOINTS]>=.6)&kp['in_image'][:,idx,c.JOINTS]
        report={'success':True,'frame':50,'calibration_sha256':c.digest(c.ROOT/json.loads((c.OUT/'acceptance.json').read_text())['candidate']),
            'scale':float(poses['scale']),'inlier_observations':int(valid.sum()),'all_candidate_observations':int(rawvalid.sum()),
            'supported_joints':int(good.sum()),'robust_selected_error':c.stats(err[valid].numpy()),'all_detector_candidate_error':c.stats(err.numpy()[rawvalid]),
            'per_camera':{str(cam):c.stats(err[i][valid[i]].numpy()) for i,cam in enumerate(cal['cams'])},
            'scope':'Static 000 frame50 body21/root/global only; beta fixed zero to match decoded MJJ appearance. Hands/face neutral. Geometric observation gate stays5px; SMPL regressor joints are approximate detector landmarks. No dynamic fit yet.'}
        (OUT/'static_fit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return poses


def export(model,poses,destination=None,attributes_file=None,latent_file=None):
    import torch
    from core.outputs.output import GaussianAppOutput
    sys.path.insert(0,str(c.ROOT/'scripts'));from avatar_rig import animate
    model.renderer.cuda();model.renderer.device=torch.device('cuda');skinner=model.renderer.smplx_model
    target=OUT/destination if destination else OUT
    if not target.resolve().is_relative_to(OUT.resolve()):raise ValueError('Avatar outside isolated MJJ output')
    target.mkdir(parents=True,exist_ok=True)
    attributes_path=OUT/attributes_file if attributes_file else LHM/'runs/mjj_decode/gaussian_attributes.pt'
    latent_path=OUT/latent_file if latent_file else LHM/'runs/mjj_latent/latent.pt'
    data=move(torch.load(attributes_path,weights_only=True),'cuda')
    latent=torch.load(latent_path,weights_only=True)['result'];poses=move(poses,'cuda')
    with torch.inference_mode():
        query=data['query']['neutral_coords'][0]+data['attrs'][0]['offset_xyz'];N=len(query)
        weights=skinner.query_voxel_skinning_weights(query[None]).reshape(N,-1)
        fixed=(skinner.is_rhand+skinner.is_lhand+skinner.is_face)>0;weights[fixed]=skinner.skinning_weight[fixed]
        neutral_to_zero=(weights@data['smplx']['transform_mat_neutral_pose'][0].reshape(-1,16)).reshape(N,4,4)
        zero=skinner.lbs(query[None],neutral_to_zero[None],torch.zeros(1,3,device='cuda'))[0]
        if poses['betas'].abs().max()>0:
            from smplx.lbs import blend_shapes
            zero=zero+blend_shapes(poses['betas'],skinner.shape_dirs)[0]
        rest=skinner.get_zero_pose_human(poses['betas'],torch.device('cuda'),None,None)
        rig={'weights':weights.float(),'zero_coords':zero.float(),'neutral_to_zero':neutral_to_zero.float(),'rest_joints':rest.float(),
             'parents':skinner.smplx_layer.parents.cuda(),'neutral_rotation':data['attrs'][0]['rotation'].float(),
             **{k:data['attrs'][0][k].float() for k in ['opacity','scaling','shs']}}
        params={k:torch.zeros(1,3,device='cuda') for k in ['jaw_pose','leye_pose','reye_pose','trans']}
        for key in ['jaw_pose','leye_pose','reye_pose']:
            if key in poses:params[key]=poses[key][:1].reshape(1,3)
        params.update(root_pose=poses['root_pose'][:1],body_pose=poses['body_pose'][:1],betas=poses['betas'],
                      lhand_pose=poses.get('left_hand_pose',torch.zeros(1,15,3,device='cuda'))[:1],
                      rhand_pose=poses.get('right_hand_pose',torch.zeros(1,15,3,device='cuda'))[:1],expr=torch.zeros(1,100,device='cuda'),transform_mat_neutral_pose=data['smplx']['transform_mat_neutral_pose'][0])
        orig,_=model.renderer.animate_gs_model(GaussianAppOutput(**data['attrs'][0]),data['query']['neutral_coords'][0],params,mesh_meta=data['query']['mesh_meta'])
        g=orig[0];replay=animate(rig,poses,0)
        local_error=float(((g.xyz*poses['scale']+poses['translation_world'][0]-replay['xyz'])/poses['scale']).abs().max())
        rot_error=float(1-torch.nn.functional.cosine_similarity(g.rotation.float(),replay['rotation'],dim=-1).abs().min())
        assert local_error<2e-5 and rot_error<2e-5,(local_error,rot_error)
        bundle={'rig':move(rig,'cpu'),'poses':move(poses,'cpu'),'features':latent[0][0].cpu(),'background_embedding':model.renderer.background_embedding.detach().cpu(),
                'decoder':move(model.neural_renderer.state_dict(),'cpu'),'decoder_config':json.loads((LHM/'checkpoint_config.json').read_text())['neural_renderer'],
                'scope':'MJJ independent 8-view decoded appearance; supplied shared beta applied to dense Gaussian shape blend and rest rig. Supplied full pose fields; no FCS identity or motion tensors.'}
        torch.save(bundle,target/'avatar.pt')
        (target/'rig_check.json').write_text(json.dumps({'gaussians':N,'upstream_max_xyz_local_error':local_error,'upstream_max_quaternion_cosine_error':rot_error,
            'shared_betas':poses['betas'].cpu().tolist(),'avatar_sha256':c.digest(target/'avatar.pt'),'sources':[str(attributes_path.relative_to(c.ROOT)),str(latent_path.relative_to(c.ROOT))]},indent=2))
    return move(bundle,'cuda')


def raster_setup():
    spec=importlib.util.spec_from_file_location('gsplat_cuda',r'D:/Document/3dgs/.tmp/rebuild_20261014_runtime/torch_extensions/gsplat_cuda/gsplat_cuda.pyd')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sys.modules['gsplat.csrc']=mod
    from gsplat import rasterization
    return rasterization


def render_static(model,bundle,cal):
    import torch,cv2
    from PIL import Image,ImageDraw
    from avatar_rig import animate
    raster=raster_setup();model.neural_renderer.cuda();w,h=504,756
    kp=np.load(c.OUT/'keypoints_crop_static.npz');fi=list(kp['frames']).index(50)
    sheet=Image.new('RGB',(w*4,h*4+24*4),'white');draw=ImageDraw.Draw(sheet);records=[]
    with torch.inference_mode():
        gs=animate(bundle['rig'],bundle['poses'],0);colors=torch.cat([gs['shs'].squeeze(1),bundle['features']],1)
        bg=torch.cat([torch.ones(3,device='cuda'),bundle['background_embedding'][2]])[None]
        for i,cam in enumerate(cal['cams']):
            x0,y0,x1,y1=kp['boxes'][i,fi];cx,cy=(x0+x1)/2,(y0+y1)/2;hh=max((y1-y0)*1.18,(x1-x0)*1.18*1.5);ww=hh/1.5
            left,top=cx-ww/2,cy-hh/2;s=w/ww
            A=np.array([[s,0,s*(.5-left)-.5],[0,s,s*(.5-top)-.5],[0,0,1.]])
            K=torch.tensor(A@cal['K'][i],device='cuda',dtype=torch.float32);W=torch.tensor(cal['W'][i],device='cuda',dtype=torch.float32)
            color,alpha,_=raster(means=gs['xyz'],quats=gs['rotation'],scales=gs['scaling'],opacities=gs['opacity'].flatten(),colors=colors,
                                viewmats=W[None],Ks=K[None],width=w,height=h,backgrounds=bg,packed=False,render_mode='RGB')
            with torch.autocast('cuda',dtype=torch.bfloat16):rgb,mask=model.neural_renderer.predict_rgbs(None,color[0,...,3:].permute(2,0,1)[None],None,h,w)
            pred=(rgb[0,0].float().clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            source=cv2.imdecode(np.fromfile(c.ROOT/f'assets/mjj/static/{cam}/0050.png',np.uint8),cv2.IMREAD_COLOR)[:,:,::-1]
            gt=cv2.warpAffine(source,A[:2],(w,h),borderValue=(255,255,255))
            mp=cv2.imdecode(np.fromfile(c.ROOT/f'assets/mjj/masks/sam2/static/{cam}/0050.png',np.uint8),0)
            target=cv2.warpAffine(mp,A[:2],(w,h),flags=cv2.INTER_NEAREST)>0
            am=alpha[0,...,0].cpu().numpy()>.5;union=(am|target).sum();iou=float((am&target).sum()/max(union,1))
            Image.fromarray(pred).save(OUT/f'static_{i:02d}.png');Image.fromarray(gt).save(OUT/f'static_gt_{i:02d}.png')
            row,col=divmod(i,4);yy=row*(2*h+48);xx=col*w
            sheet.paste(Image.fromarray(gt),(xx,yy+24));sheet.paste(Image.fromarray(pred),(xx,yy+h+48))
            draw.text((xx+5,yy+5),f'{cam} / GT',fill='black');draw.text((xx+5,yy+h+29),f'MJJ static / alpha IoU {iou:.3f}',fill='black')
            records.append({'camera':str(cam),'alpha_iou':iou,'native_to_render':A.tolist(),'K':K.cpu().tolist()});print('static render',cam,'alpha_iou',iou,flush=True)
    sheet.resize((1008,1560)).save(OUT/'static_contact.jpg',quality=94)
    (OUT/'static_render.json').write_text(json.dumps({'records':records,'scope':'All 8 GT/render camera crops with exact pixel-center transform; alpha IoU is silhouette diagnostic, not calibration quality.'},indent=2))


def main():
    r,cal=gate();OUT.mkdir(parents=True,exist_ok=True);model=model_load()
    poses=fit_static(model,cal);bundle=export(model,poses);render_static(model,bundle,cal)


if __name__=='__main__':main()
