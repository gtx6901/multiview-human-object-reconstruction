"""All-eight-camera MJJ001 body motion with true capture time and fixed identity.

23 landmarks = 12 skeletal body points + 5 head surface points + 6 foot vertices.
No fingers fitted. Outputs and intermediates remain under work/mjj/reconstruction.
"""
import sys,json
import numpy as np
import calibrate as c
from reconstruct import OUT,LHM,gate,move


def configure():
    sys.path.insert(0,str(LHM));from runtime_imports import configure as config
    config();sys.path.insert(0,str(c.ROOT/'scripts'))


def fit(observations_file=None,output_subdir=None,initial_motion_file=None):
    gate();inspection=json.loads((OUT/'static_inspection.json').read_text())
    if not inspection['allowed_to_attempt_body_motion'] or c.digest(OUT/'avatar.pt')!=inspection['avatar_sha256']:raise RuntimeError('Static inspection gate closed')
    configure()
    import torch
    from smplx.lbs import batch_rigid_transform
    from smplx.vertex_ids import vertex_ids
    from pytorch3d.transforms import axis_angle_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d,matrix_to_axis_angle
    from screen_observations import solve
    torch.set_num_threads(4);torch.manual_seed(20261027);device='cuda'
    accepted,cal=gate();kp=np.load(c.OUT/'keypoints_crop_dynamic.npz');xy=kp['xy'][:,:,:23].astype(float)
    valid=(kp['score'][:,:,:23]>=.6)&kp['in_image'][:,:,:23]
    dest=OUT/output_subdir if output_subdir else OUT
    if not dest.resolve().is_relative_to(OUT.resolve()):raise ValueError('Output outside isolated MJJ directory')
    dest.mkdir(parents=True,exist_ok=True)
    if observations_file:
        path=OUT/observations_file
        if not path.resolve().is_relative_to(OUT.resolve()):raise ValueError('Observations outside isolated MJJ directory')
        observation=np.load(path)
        assert np.array_equal(observation['cameras'],kp['cams']) and np.array_equal(observation['frames'],kp['frames'])
        X=observation['xyz'][:,:23];support=observation['support'][:,:23];inlier=observation['inlier'][:,:,:23]
    else:
        X,support,inlier,err=solve(cal['K'],cal['W'],xy,valid)
        np.savez_compressed(dest/'motion_observations.npz',xyz=X,support=support,inlier=inlier,error_px=err,frames=kp['frames'],cams=kp['cams'])
    skel=torch.load(OUT/'skeleton.pt',weights_only=True);rest=skel['rest'].to(device);parents=skel['parents'].to(device)
    d=np.load(LHM/'pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',allow_pickle=True)
    template=torch.tensor(d['v_template'],dtype=torch.float32,device=device)
    ids=vertex_ids['smplx'];head=[ids[n] for n in ['nose','leye','reye','lear','rear']]
    feet=[ids[n] for n in ['LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']]
    vi=head+feet;vertices=template[vi];weights=torch.tensor(d['weights'],dtype=torch.float32,device=device)[vi]
    expected=torch.tensor(d['J_regressor'],dtype=torch.float32,device=device)@template
    assert torch.allclose(expected[:55],rest[0],atol=1e-6)
    static=move(torch.load(OUT/'static_pose.pt',weights_only=True),device);T=len(kp['frames']);fps=100/11;dt=1/fps
    root_init=matrix_to_rotation_6d(axis_angle_to_matrix(static['root_pose'])).repeat(T,1)
    body_init=matrix_to_rotation_6d(axis_angle_to_matrix(static['body_pose'])).repeat(T,1,1)
    initial=None
    if initial_motion_file:
        initial=move(torch.load(OUT/initial_motion_file,weights_only=True),device)
        assert len(initial['frames'])==T and torch.allclose(initial['scale'],static['scale']) and torch.allclose(initial['betas'],static['betas'])
        root_init=matrix_to_rotation_6d(axis_angle_to_matrix(initial['root_pose']))
        body_init=matrix_to_rotation_6d(axis_angle_to_matrix(initial['body_pose']))
    root=torch.nn.Parameter(root_init.clone());body=torch.nn.Parameter(body_init.clone());scale=static['scale'].detach()
    # Pelvis translation initialization interpolates missing triangulations only;
    # interpolated points never enter the measurement loss.
    hips=X[:,11:13].mean(1);good=np.isfinite(hips).all(1)
    if good.sum()<2:raise RuntimeError('Insufficient pelvis support')
    hips=np.stack([np.interp(np.arange(T),np.flatnonzero(good),hips[good,j]) for j in range(3)],1)
    with torch.no_grad():
        rr=axis_angle_to_matrix(torch.cat([static['root_pose'][:,None],static['body_pose'],torch.zeros(1,33,3,device=device)],1))
        sj,_=batch_rigid_transform(rr,rest,parents);localhip=sj[0,[1,2]].mean(0)
    trans=torch.nn.Parameter(initial['translation_world']/scale if initial is not None else torch.tensor(hips,dtype=torch.float32,device=device)/scale-localhip)
    K=torch.tensor(cal['K'],dtype=torch.float32,device=device);W=torch.tensor(cal['W'],dtype=torch.float32,device=device)
    target=torch.tensor(xy,dtype=torch.float32,device=device);m=torch.tensor(inlier,device=device)
    target3=torch.tensor(np.nan_to_num(X),dtype=torch.float32,device=device);m3=torch.tensor(np.isfinite(X).all(-1),device=device)
    smpl=[16,17,18,19,20,21,1,2,4,5,7,8]
    def evaluate():
        rot=torch.cat([rotation_6d_to_matrix(root)[:,None],rotation_6d_to_matrix(body),torch.eye(3,device=device).repeat(T,33,1,1)],1)
        joints,A=batch_rigid_transform(rot,rest.expand(T,-1,-1),parents)
        transform=(weights@A.reshape(T,55,16)).reshape(T,len(vi),4,4)
        surface=(transform[:,:,:3,:3]@vertices[None,:,:,None])[...,0]+transform[:,:,:3,3]
        points=torch.cat([surface[:,:5],joints[:,smpl],surface[:,5:]],1);world=(points+trans[:,None])*scale
        pc=torch.einsum('cij,tnj->ctni',W[:,:3,:3],world)+W[:,:3,3][:,None,None]
        uv=torch.einsum('cij,ctnj->ctni',K,pc);uv=uv[...,:2]/uv[...,2:].clamp_min(1e-5)
        e=torch.linalg.norm(uv-target,dim=-1);data=(2*(torch.sqrt(1+(e[m]/4)**2)-1)).mean()
        d3=torch.linalg.norm((world-target3)/scale,dim=-1);loss3=(torch.sqrt(1+(d3[m3]/.02)**2)-1).mean()
        velocity=(rot[1:,:22]-rot[:-1,:22])/dt;accel=(rot[2:,:22]-2*rot[1:-1,:22]+rot[:-2,:22])/dt**2
        tv=(trans[1:]-trans[:-1])/dt;ta=(trans[2:]-2*trans[1:-1]+trans[:-2])/dt**2
        temporal=.002*velocity.square().mean()+.00002*accel.square().mean()+.005*tv.square().mean()+.0001*ta.square().mean()
        prior=.005*(rot[:,1:22]-torch.eye(3,device=device)).square().mean()+.001*(root-root_init).square().mean()
        return data+.1*loss3+temporal+prior,e,uv,world,rot,{'data':data,'3d':loss3,'temporal':temporal,'prior':prior}
    opt=torch.optim.Adam([root,body,trans],lr=.012)
    for step in range(1801):
        if step==1000:
            for group in opt.param_groups:group['lr']=.003
        if step==1500:
            for group in opt.param_groups:group['lr']=.001
        opt.zero_grad();loss,e,uv,world,rot,parts=evaluate();loss.backward();opt.step()
        if step%200==0:print('motion',step,'loss',float(loss),'median',float(e[m].median()),{k:float(v) for k,v in parts.items()},flush=True)
    with torch.no_grad():
        loss,e,uv,world,rot,parts=evaluate();aa=matrix_to_axis_angle(rot)
        poses={'root_pose':aa[:,0].detach(),'body_pose':aa[:,1:22].detach(),'translation_world':(trans*scale).detach(),
               'scale':scale,'betas':torch.zeros(1,10,device=device),'frames':torch.tensor(kp['frames'],device=device),'fps':fps}
        torch.save(move(poses,'cpu'),dest/'motion.pt')
        import shutil
        shutil.copy2(dest/'motion.pt',dest/'motion_body.pt')
        np.savez_compressed(dest/'motion_joints.npz',world=world.cpu().numpy(),uv=uv.cpu().numpy(),inlier=inlier,target=xy)
        ee=e.cpu().numpy();vel=torch.linalg.norm((world[1:]-world[:-1])/scale/dt,dim=-1).cpu().numpy()
        report={'success':True,'frames':T,'cameras':list(map(str,kp['cams'])),'fps':fps,'dt_seconds':dt,
            'calibration_sha256':accepted['candidate_sha256'],'avatar_sha256':c.digest(OUT/'avatar.pt'),'motion_sha256':c.digest(dest/'motion.pt'),
            'observations_source':str(observations_file) if observations_file else 'MJJ original5px screening',
            'initialization_source':str(initial_motion_file) if initial_motion_file else 'single static pose repeated for all frames',
            'landmark_scope':'Body12 SMPL joint centers, head5 generic SMPL surface vertices, feet6 generic vertices. Betas/scale/appearance fixed to static MJJ. No hand/finger fit.',
            'measurement_coverage':float(m3.float().mean()),'body_coverage':float(m3[:,5:17].float().mean()),
            'selected_measurement_error':c.stats(ee[inlier]),'all_detector_candidate_error':c.stats(ee[valid]),
            'per_camera':{str(cam):{'selected':c.stats(ee[i][inlier[i]]),'all_candidates':c.stats(ee[i][valid[i]])} for i,cam in enumerate(kp['cams'])},
            'per_frame_selected': [c.stats(ee[:,t][inlier[:,t]]) for t in range(T)],
            'joint_speed_model_units_per_second_p50_p95_max':np.percentile(vel,[50,95,100]).tolist(),
            'temporal_loss_weights':{'rotation_velocity':.002,'rotation_acceleration':.00002,'translation_velocity':.005,'translation_acceleration':.0001},
            'scope':'Motion fitting experiment, not quality acceptance. All eight cameras participate; missing joints inferred by skeleton/time priors. Inputs may contain fast-motion blur; raw detector metrics retained.'}
        (dest/'motion_fit.json').write_text(json.dumps(report,indent=2));print({k:report[k] for k in ['frames','measurement_coverage','selected_measurement_error','all_detector_candidate_error']},flush=True)


if __name__=='__main__':
    raise SystemExit('Historical per-frame fit caused visible jitter. Use fit_continuous_motion.py. Importing configure() remains supported.')
