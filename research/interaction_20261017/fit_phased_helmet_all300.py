"""Multiview rigid helmet fitting with shared attachment phases and depth-aware unknowns."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle
from mesh_renderer import MeshRenderer
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_phased_all300_v1');p.add_argument('--steps',type=int,default=100);p.add_argument('--initial',default='helmet_initial_all300_v1');p.add_argument('--phase-motion');p.add_argument('--skip-coarse',action='store_true');p.add_argument('--material-loss-weight',type=float,default=0.);p.add_argument('--material-mesh');p.add_argument('--phase-priors',default='helmet_phase_priors_all300_v1');p.add_argument('--body-depth',default='body_depth_all300_v1');p.add_argument('--opening-direction-weight',type=float,default=0.);p.add_argument('--crop-observations',action='store_true');p.add_argument('--worn-rotation-cap-deg',type=float);p.add_argument('--worn-translation-cap',type=float)
p.add_argument('--rgb-loss-weight',type=float,default=0.);p.add_argument('--rgb-texture');p.add_argument('--rgb-missing-penalty',type=float,default=.25,help='Fixed initial photo support;0 reproduces legacy variable-support loss');p.add_argument('--observation-frames',type=int,nargs='+');p.add_argument('--joint-head-avatar');p.add_argument('--joint-head-weight',type=float,default=.1);p.add_argument('--joint-head-face-anchors');p.add_argument('--joint-head-front-visibility',action='store_true');p.add_argument('--joint-head-lr',type=float,default=.01);p.add_argument('--hand-contact-avatar');p.add_argument('--hand-contact-weight',type=float,default=.03);options=p.parse_args()
assert not (options.rgb_loss_weight and options.material_loss_weight),'Use real RGB or proxy material loss separately'
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
maskroot=ROOT/'sam2_all300_v1';observations=json.loads((maskroot/'report.json').read_text());assert observations['success']
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());initial=torch.load(ROOT/options.initial/'sequence.pt',weights_only=True)
phaseinfo=json.loads((ROOT/options.phase_priors/'report.json').read_text());anchor=torch.load(ROOT/options.phase_priors/'anchors.pt',weights_only=True)
vertices=initial['vertices_local'].cuda();faces=initial['faces'].cuda();scale=initial['logscale'].exp().cuda();renderer=MeshRenderer();width,height=256,192
if options.crop_observations:width=height=256
A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
Ks=torch.tensor(np.array([A@np.array(v['K']) for v in meta['views']]),device='cuda',dtype=torch.float32);Ws=torch.tensor([v['W'] for v in meta['views']],device='cuda',dtype=torch.float32)
records={(r['serial'],r['frame']):r for r in observations['records']};frames=[];views=[];labels=[];eligible=[];bodydepth=[];bodyalpha=[];photo_targets=[];photo_confidence=[]
observation_K=[];crop_records=[]
material_colors=None;opening_normal=None
rgb_uv=rgb_texture=rgb_confidence=None;photo_rgb=[]
if options.rgb_loss_weight:
    assert options.rgb_texture
    from helmet_texture_sampling import srgb_to_linear
    texture_source=ROOT/options.rgb_texture;texmesh=torch.load(texture_source/'mesh.pt',weights_only=True)
    assert torch.allclose(texmesh['vertices_local'][texmesh['faces'].long()],initial['vertices_local'][initial['faces'].long()],atol=1e-6)
    vertices=texmesh['vertices_local'].cuda();faces=texmesh['faces'].cuda();rgb_uv=texmesh['uv'].cuda();rgb_texture=texmesh['texture'].cuda()
    rgb_confidence=torch.tensor(np.array(Image.open(texture_source/'observed_texels.png')),device='cuda',dtype=torch.float32)/255
    correction=json.loads((texture_source/'camera_color_correction.json').read_text());assert correction['reference_serial']==meta['views'][1]['serial']
    rgb_gain=torch.tensor(correction['gain_linear_rgb'],device='cuda',dtype=torch.float32);rgb_bias=torch.tensor(correction['bias_linear_rgb'],device='cuda',dtype=torch.float32)
if options.material_loss_weight:
    assert options.material_mesh
    materialmesh=torch.load(ROOT/options.material_mesh/'mesh.pt',weights_only=True)
    assert torch.allclose(materialmesh['vertices_local'],initial['vertices_local']) and torch.equal(materialmesh['faces'],initial['faces'])
    material_colors=torch.full_like(vertices,.05);material_colors[:materialmesh['outer_vertex_count']]=.9
    opening_normal=torch.tensor(json.loads((ROOT/options.material_mesh/'report.json').read_text())['opening_plane_normal_local'],device='cuda',dtype=torch.float32)
for vi,v in enumerate(meta['views']):
    cache=np.load(ROOT/options.body_depth/f"{v['serial']}.npz");bd=cache['depth'];ba=cache['alpha'];hand=cache['hand_unknown']
    assert np.allclose(cache['K'],Ks[vi].cpu().numpy()) and np.allclose(cache['W'],Ws[vi].cpu().numpy())
    valid_native=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/'valid.png'))
    valid=cv2.resize(valid_native,(256,192),interpolation=cv2.INTER_NEAREST)>127
    for f in range(300):
        if options.observation_frames is not None and f not in options.observation_frames:continue
        r=records[(v['serial'],f)]
        if not r['mask']:continue
        rawlab=np.asarray(Image.open(maskroot/r['labels']));rawperson=np.asarray(Image.open(maskroot/r['person_mask']));rawobj=np.asarray(Image.open(maskroot/r['mask']))
        if options.crop_observations:
            roi_bounds=np.array(r['roi_xyxy'],dtype=float);extent=max(roi_bounds[2:]-roi_bounds[:2]);assert extent>0
            origin=(roi_bounds[:2]+roi_bounds[2:])/2-extent/2;s=width/extent
            C=np.array([[s,0,-s*origin[0]+.5*s-.5],[0,s,-s*origin[1]+.5*s-.5],[0,0,1.]])
            def warp_native(image,border=0,mode=cv2.INTER_NEAREST):return cv2.warpAffine(image,C[:2],(width,height),flags=mode,borderMode=cv2.BORDER_CONSTANT,borderValue=border)
            def warp_cache(image):return cv2.warpAffine(image,(C@np.linalg.inv(A))[:2],(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            lab=warp_native(rawlab,255);person=warp_native(rawperson)>127;obj=warp_native(rawobj)>127;currentvalid=warp_native(valid_native)>127
            bdepth=warp_cache(bd[f]);balpha=warp_cache(ba[f]);hunknown=warp_cache(hand[f]);native_roi=np.zeros_like(valid_native);x0,y0,x1,y1=roi_bounds
            native_roi[max(0,int(np.floor(y0))):min(1536,int(np.ceil(y1))),max(0,int(np.floor(x0))):min(2048,int(np.ceil(x1)))]=1;roi=warp_native(native_roi)>0
            guard=int(np.ceil(8*s));boundary_guard=cv2.dilate(obj.astype('uint8'),np.ones((2*guard+1,2*guard+1),np.uint8))>0;currentK=C@np.array(v['K'])
        else:
            lab=cv2.resize(rawlab,(width,height),interpolation=cv2.INTER_NEAREST);person=cv2.resize(rawperson,(width,height),interpolation=cv2.INTER_NEAREST)>127;obj=cv2.resize(rawobj,(width,height),interpolation=cv2.INTER_NEAREST)>127
            currentvalid=valid;bdepth=bd[f];balpha=ba[f];hunknown=hand[f];boundary_guard=cv2.dilate(obj.astype('uint8'),np.ones((3,3),np.uint8))>0
            roi=np.zeros_like(valid);x0,y0,x1,y1=np.array(r['roi_xyxy'])*.125;roi[max(0,int(np.floor(y0))):min(height,int(np.ceil(y1))),max(0,int(np.floor(x0))):min(width,int(np.ceil(x1)))]=True;currentK=Ks[vi].cpu().numpy()
        if (lab==1).sum()<10:continue
        restore=(lab==255)&person&~boundary_guard&(hunknown==0)&currentvalid&roi&(balpha>0)
        frames.append(f);views.append(vi);labels.append(lab);eligible.append(restore);bodydepth.append(bdepth);bodyalpha.append(balpha);observation_K.append(currentK)
        if options.crop_observations:crop_records.append({'serial':v['serial'],'frame':f,'native_origin':origin.tolist(),'native_square_extent':float(extent),'native_to_crop':C.tolist(),'foreground_crop_pixels':int((lab==1).sum())})
        if options.material_loss_weight or options.rgb_loss_weight:
            native_rgb=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB'))
            rgb=(warp_native(native_rgb,0,cv2.INTER_LINEAR) if options.crop_observations else cv2.resize(native_rgb,(width,height),interpolation=cv2.INTER_AREA)).astype('float32')/255
            luminance=rgb@np.array([.2126,.7152,.0722]);white=max(.6,float(np.quantile(luminance[lab==1],.95)));normalized=luminance/white
            if options.material_loss_weight:
                photo_targets.append(normalized>.7);photo_confidence.append((lab==1)&((normalized>.7)|(normalized<.2))&(hunknown==0))
            else:
                confidence=(lab==1)&(hunknown==0)&currentvalid&~(rgb>.985).all(-1)
                if 'decoded_alpha' in cache:
                    decoded=warp_cache(cache['decoded_alpha'][f]) if options.crop_observations else cache['decoded_alpha'][f]
                    confidence&=~((decoded>.5)&(balpha<.5))
                photo_confidence.append(confidence);photo_rgb.append(np.uint8(np.clip(rgb,0,1)*255))
    print('prepared masks/depth',v['serial'],flush=True)
labels=torch.tensor(np.array(labels));eligible=torch.tensor(np.array(eligible));bodydepth=torch.tensor(np.array(bodydepth));N=len(frames)
observation_K=torch.tensor(np.array(observation_K),device='cuda',dtype=torch.float32)
if crop_records:(out/'observation_crops.json').write_text(json.dumps(crop_records,indent=2))
if options.material_loss_weight:photo_targets=torch.tensor(np.array(photo_targets));photo_confidence=torch.tensor(np.array(photo_confidence))
if options.rgb_loss_weight:photo_rgb=torch.tensor(np.array(photo_rgb));photo_confidence=torch.tensor(np.array(photo_confidence));bodyalpha=torch.tensor(np.array(bodyalpha))
if options.observation_frames is None:assert N>1500 and set(frames)==set(range(300))
else:assert N>=2 and set(frames)==set(options.observation_frames)
frameids=torch.tensor(frames,device='cuda');viewids=torch.tensor(views,device='cuda');R0=initial['R'].cuda();t0=initial['translation_world'].cuda();phases=phaseinfo['phases'];P=len(phases)
rgb_support_counts=torch.zeros(N,device='cuda',dtype=torch.int64)
rgb_reference_support=torch.zeros_like(labels,dtype=torch.bool) if options.rgb_loss_weight else None
capture_rgb_reference=False
rgb_reference_ready=False
phaseids=torch.full((300,),-1,device='cuda',dtype=torch.long);ar=torch.eye(3,device='cuda').expand(300,-1,-1).clone();at=torch.zeros(300,3,device='cuda')
relR=[];relt=[]
for i,phase in enumerate(phases):
    a,b,j=phase['start'],phase['stop_exclusive'],phase['joint'];assert (phaseids[a:b]<0).all()
    phaseids[a:b]=i;ar[a:b]=anchor['global_joint_R'][a:b,j].cuda();at[a:b]=anchor['joints_world'][a:b,j].cuda()
    relR.append(phase['relative_R']);relt.append(phase['relative_translation'])
relR=torch.tensor(relR,device='cuda',dtype=torch.float32);relt=torch.tensor(relt,device='cuda',dtype=torch.float32)
attached=phaseids>=0;pid=phaseids.clamp_min(0);rraw=torch.nn.Parameter(torch.zeros(300,3,device='cuda'));traw=torch.nn.Parameter(torch.zeros(300,3,device='cuda'));pr=torch.nn.Parameter(torch.zeros(P,3,device='cuda'));pt=torch.nn.Parameter(torch.zeros(P,3,device='cuda'))
rotation_cap=torch.full((300,1),.35,device='cuda');translation_cap=torch.full((300,1),.015,device='cuda')
for phase in phases:
    if phase['joint']==15:
        a,b=phase['start'],phase['stop_exclusive']
        if options.worn_rotation_cap_deg is not None:
            assert options.worn_rotation_cap_deg>0
            rotation_cap[a:b]=np.deg2rad(options.worn_rotation_cap_deg)/np.sqrt(3)
        if options.worn_translation_cap is not None:
            assert options.worn_translation_cap>0
            translation_cap[a:b]=options.worn_translation_cap/np.sqrt(3)
if options.phase_motion:
    previous=json.loads((ROOT/options.phase_motion/'report.json').read_text())
    assert [(x['start'],x['stop_exclusive'],x['joint']) for x in previous['phases']]==[(x['start'],x['stop_exclusive'],x['joint']) for x in phases]
    relR=torch.tensor(previous['shared_relative_R'],device='cuda',dtype=torch.float32);relt=torch.tensor(previous['shared_relative_translation'],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        priorR=ar@relR[pid];priort=at+(ar@relt[pid,...,None])[...,0]
        residual=matrix_to_axis_angle(R0@priorR.transpose(-1,-2))
        rraw[attached]=torch.atanh((residual[attached]/rotation_cap[attached]).clamp(-.999999,.999999))
        traw[attached]=translation_cap[attached]*torch.atanh(((t0-priort)[attached]/translation_cap[attached]).clamp(-.999999,.999999))
joint_head=None
if options.joint_head_avatar:
    from joint_head_constraints import JointHeadConstraints
    worn=torch.zeros(300,device='cuda',dtype=torch.bool)
    for phase in phases:
        if phase['joint']==15:worn[phase['start']:phase['stop_exclusive']]=True
    joint_head=JointHeadConstraints(ROOT,options.joint_head_avatar,options.body_depth,meta,worn,options.joint_head_face_anchors,options.joint_head_front_visibility)
fitparams=[rraw,traw,pr,pt]+([] if joint_head is None else [joint_head.raw])

def trajectory():
    current_ar,current_at=ar,at
    if joint_head is not None:
        headR,headt,_,_,_=joint_head.state(surface=False);current_ar=torch.where(joint_head.worn[:,None,None],headR,ar);current_at=torch.where(joint_head.worn[:,None],headt,at)
    phaseR=current_ar@(axis_angle_to_matrix(pr[pid])@relR[pid]);phaset=current_at+(current_ar@(relt[pid]+pt[pid])[...,None])[...,0]
    residualR=torch.where(attached[:,None],rotation_cap*torch.tanh(rraw),rraw)
    residualt=torch.where(attached[:,None],translation_cap*torch.tanh(traw/translation_cap),traw)
    R=axis_angle_to_matrix(residualR)@torch.where(attached[:,None,None],phaseR,R0)
    t=torch.where(attached[:,None],phaset,t0)+residualt
    return R,t,residualR,residualt
def cost(R,t,ids):
    vi=viewids[ids];f=frameids[ids]
    world=torch.einsum('bij,nj->bni',R[f],vertices)*scale+t[f,None]
    render=renderer.depth_alpha_batch(world,faces,observation_K[ids],Ws[vi],width,height,colors=material_colors,uv=rgb_uv,texture=rgb_texture,texture_confidence=rgb_confidence)
    lab=labels[ids.cpu()].cuda();restore_eligible=eligible[ids.cpu()].cuda();bd=bodydepth[ids.cpu()].cuda()
    # Only restore pixels confidently in front of the fitted body, away from hands/edges.
    restore=restore_eligible&(render['hard_alpha']>.5)&(render['depth']>0)&(render['depth']<bd-.01)
    valid=(lab!=255).float()+.5*restore.float();target=(lab==1).float();pred=render['alpha']*valid
    intersection=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1)
    iou=intersection/union
    photo=torch.zeros_like(iou)
    if options.material_loss_weight:
        confidence=photo_confidence[ids.cpu()].cuda();truth=photo_targets[ids.cpu()].cuda().float()
        predicted=((render['premultiplied_rgb'][...,0]/render['alpha'].clamp_min(1e-5)-.05)/.85).clamp(0,1)
        photo=((predicted-truth).abs()*confidence*render['alpha']).sum((1,2))/confidence.sum((1,2)).clamp_min(1)
    if options.rgb_loss_weight:
        truth=srgb_to_linear(photo_rgb[ids.cpu()].cuda().float()/255)
        truth=(truth*rgb_gain[vi,None,None]+rgb_bias[vi,None,None]).clamp(0,1)
        predicted=srgb_to_linear((render['premultiplied_rgb']/render['alpha'][...,None].clamp_min(1e-5)).clamp(0,1))
        confidence=photo_confidence[ids.cpu()].cuda()&(render['texture_confidence']>.75)&(render['alpha']>.9)
        confidence&=(bodyalpha[ids.cpu()].cuda()<.5)|(render['depth']<bd+.01)
        if capture_rgb_reference:rgb_reference_support[ids.cpu()]=confidence.detach().cpu()
        if options.rgb_missing_penalty>0 and rgb_reference_ready and not capture_rgb_reference:
            reference=rgb_reference_support[ids.cpu()].cuda().float()
            available=render['texture_confidence'].clamp(0,1)*render['alpha'].clamp(0,1)
            available=available*((bodyalpha[ids.cpu()].cuda()<.5)|(render['depth']<bd+.01)).float()
            weight=reference*available;rgb_support_counts[ids]=(weight>.75).sum((1,2)).detach()
            photo=((predicted-truth).abs().mean(-1)*weight+options.rgb_missing_penalty*(reference-weight).clamp_min(0)).sum((1,2))/reference.sum((1,2)).clamp_min(1)
        else:
            weight=confidence.float()*render['alpha'].detach();rgb_support_counts[ids]=confidence.sum((1,2)).detach()
            photo=((predicted-truth).abs().mean(-1)*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1)
    return 1-iou+(options.material_loss_weight+options.rgb_loss_weight)*photo,iou,restore.sum((1,2)),photo
hand_contact=None
if options.hand_contact_avatar:
    from hand_helmet_contact import HandHelmetContact
    with torch.no_grad():R,t,_,_=trajectory()
    hand_contact=HandHelmetContact(ROOT,options.hand_contact_avatar,vertices,faces,scale,R,t)
# Coarse rotation search for phase representatives and the donning transition.
# Never use held-out views here; this is the all8 main experiment.
coarse=[];group=torch.tensor(Rotation.create_group('O').as_matrix(),device='cuda',dtype=torch.float32)
with torch.no_grad():
    R,t,_,_=trajectory()
    coarse_frames=[] if options.skip_coarse else sorted(set([150,165]+[(x['start']+min(x['stop_exclusive'],212)-1)//2 for x in phases]))
    for f in coarse_frames:
        obs=torch.where(frameids==f)[0]
        if len(obs)<2:continue
        best=None
        for q in group:
            proposed=R.clone();proposed[f]=R[f]@q
            loss,_,_,_=cost(proposed,t,obs);value=float(loss.mean())
            if best is None or value<best[0]:best=(value,q.clone())
        i=int(phaseids[f])
        if i>=0:
            # R_world = R_anchor @ R_relative; compose the tested local increment.
            relR[i]=relR[i]@best[1]
        else:
            R0[f]=R[f]@best[1]
            # Blend correction over the transition, avoid a single-frame seed jump.
            from scipy.spatial.transform import Slerp
            a=max(136,f-10);b=min(179,f+10)
            if a<f<b:
                seed=Rotation.from_matrix(np.stack([R0[a].cpu().numpy(),R0[f].cpu().numpy(),R0[b].cpu().numpy()]))
                R0[a:b+1]=torch.tensor(Slerp([a,f,b],seed)(np.arange(a,b+1)).as_matrix(),device='cuda',dtype=torch.float32)
        R,t,_,_=trajectory();coarse.append({'frame':f,'orientations_tested':24,'selected_input_loss':best[0]})
allids=torch.arange(N,device='cuda');chunks=list(allids.split(32));history=[];best=None
def metrics():
    R,t,_,_=trajectory();ious=[];restores=[];photos=[]
    for ids in chunks:
        _,i,r,photo=cost(R,t,ids);ious.append(i);restores.append(r);photos.append(photo)
    return torch.cat(ious),torch.cat(restores),torch.cat(photos)
capture_rgb_reference=bool(options.rgb_loss_weight)
with torch.no_grad():before,restored_before,beforephoto=metrics();before=before.cpu();beforephoto=beforephoto.cpu()
capture_rgb_reference=False
rgb_reference_ready=bool(options.rgb_loss_weight)
if options.rgb_loss_weight and options.rgb_missing_penalty>0:
    with torch.no_grad():before,restored_before,beforephoto=metrics();before=before.cpu();beforephoto=beforephoto.cpu()
rgb_support_counts_before=rgb_support_counts.detach().cpu().clone()
opt=torch.optim.Adam([{'params':[rraw,pr],'lr':.018},{'params':[traw,pt],'lr':.0015}])
if joint_head is not None:opt.add_param_group({'params':[joint_head.raw],'lr':options.joint_head_lr})
for step in range(options.steps+1):
    opt.zero_grad();data_sum=0.;iou_sum=0.;photo_sum=0.
    for ids in chunks:
        R,t,_,_=trajectory();loss,iou,_,photo=cost(R,t,ids);chunkloss=loss.sum()/N;data_sum+=float(chunkloss.detach());iou_sum+=float(iou.detach().sum())/N;photo_sum+=float(photo.detach().sum())/N
        if step<options.steps:chunkloss.backward()
    R,t,rr,tt=trajectory()
    regularization=2*(R[2:]-2*R[1:-1]+R[:-2]).square().mean()+1000*(t[2:]-2*t[1:-1]+t[:-2]).square().mean()+.002*rr.square().mean()+10*tt.square().mean()
    if joint_head is not None:regularization+=options.joint_head_weight*joint_head.face_loss()[0]
    if hand_contact is not None:regularization+=options.hand_contact_weight*hand_contact.loss(R,t)
    if opening_normal is not None:
        worn=torch.zeros(300,dtype=torch.bool,device='cuda')
        for phase in phases:
            if phase['joint']==15:worn[phase['start']:phase['stop_exclusive']]=True
        up=torch.nn.functional.normalize(anchor['joints_world'][:,15].cuda()-anchor['joints_world'][:,12].cuda(),dim=1)
        alignment=((R@opening_normal)*up).sum(1)
        regularization+=.1*torch.relu(.45-alignment[worn]).square().mean()
        if options.opening_direction_weight:
            front=anchor['joints_world'][:,[23,24]].mean(1).cuda()-anchor['joints_world'][:,15].cuda();front=front-(front*up).sum(1,keepdim=True)*up;front=torch.nn.functional.normalize(front,dim=1)
            target_normal=torch.nn.functional.normalize(up-.6*front,dim=1)
            regularization+=options.opening_direction_weight*(1-((R@opening_normal)*target_normal).sum(1))[worn].mean()
    value=data_sum+float(regularization.detach())
    if best is None or value<best[0]:best=(value,[x.detach().clone() for x in fitparams])
    if step<options.steps:regularization.backward();opt.step()
    for paramgroup,lr in zip(opt.param_groups,[.018,.0015]):paramgroup['lr']=lr*(1-.8*step/options.steps)
    if step%20==0:
        row={'step':step,'loss':value,'input_soft_iou':iou_sum,('rgb_photometric_error' if options.rgb_loss_weight else 'material_class_error'):photo_sum};history.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    for x,b in zip(fitparams,best[1]):x.copy_(b)
    R,t,rr,tt=trajectory();after,restored,afterphoto=metrics();assert torch.isfinite(R).all() and torch.isfinite(t).all()
    assert torch.allclose(torch.linalg.det(R),torch.ones(300,device='cuda'),atol=1e-4)
    sequence={**initial,'R':R.cpu(),'translation_world':t.cpu()};torch.save(sequence,out/'sequence.pt')
    np.savez_compressed(out/'observation_metrics.npz',frames=frames,views=views,before_iou=before,after_iou=after.cpu(),restored_negative_pixels=restored.cpu())
    report={'success':True,'frames':300,'observations':N,'seconds':time.time()-start,'history':history,'coarse_search':coarse,'phases':phases,'mean_input_soft_iou_before':float(before.mean()),'mean_input_soft_iou_after':float(after.mean()),'restored_negative_pixels':int(restored.sum()),'shared_relative_R':(axis_angle_to_matrix(pr)@relR).cpu().tolist(),'shared_relative_translation':(relt+pt).cpu().tolist(),'max_rotation_residual_rad':float(rr[attached].norm(dim=1).max()),'max_translation_residual':float(tt[attached].norm(dim=1).max()),'scope':'All8 main experiment; every300 frame uses real SAM2 candidate masks. Proposed stable-grasp/head-relative phases share transforms with small bounded per-frame residuals; other intervals free.24-way coarse rotations at selected phase/donning frames, original calibrated cameras/fixed shared scale. Tri-state silhouette loss restores half-weight negative evidence only when the predicted object is >.01 normalized-world units in front of a fitted bare-body mesh, away from hands and boundaries. Model/mask/rotation ambiguity remains. Before/after use same current mask/depth protocol after coarse initialization, not prior masks or held-out evaluation. Geometry remains old filled hull; no cavity/UV/texture or physical-contact accuracy claim.'}
    report['sources']=vars(options)
    if options.rgb_loss_weight:report['fixed_photo_support']={'enabled':options.rgb_missing_penalty>0,'reference_pixels':int(rgb_reference_support.sum()),'before_current_pixels':int(rgb_support_counts_before.sum()),'after_current_pixels':int(rgb_support_counts.sum()),'missing_penalty':options.rgb_missing_penalty,'scope':'Reference support captured after coarse initialization and fixed through motion updates. New loss denominator remains initial pixel count; soft mesh alpha/observed-UV confidence weights the missing-coverage penalty. Observations initially without RGB support remain unknown. Legacy penalty0 retains variable-support behavior. New metric protocol differs from earlier RGB trial.'}
    if joint_head is not None:report['joint_head']=joint_head.save(out)
    if hand_contact is not None:report['hand_contact']=hand_contact.report(R,t)
    report['observed_frames']=sorted(set(frames))
    if options.observation_frames is not None:report['scope']='SELECTED-FRAME DIAGNOSTIC ONLY: saved300 motion is an initializer plus limited-frame fitting, NOT a full-action reconstruction. '+report['scope'].replace('every300 frame uses','selected frames use')
    if options.worn_rotation_cap_deg is not None or options.worn_translation_cap is not None:
        report['scope']+=' Worn-phase per-frame axis limits divided by sqrt(3) bound the TOTAL residual rotation/translation norms by the specified values; shared phase transform is still optimized. This enforces an attachment prior, not measured contact or metric-unit accuracy. Tight bounds may expose frozen-head motion errors.'
    cache_report=json.loads((ROOT/options.body_depth/'report.json').read_text())
    if 'avatar_source' in cache_report:
        report['scope']=report['scope'].replace('fitted bare-body mesh','conservative LHM expected-depth core')
        report['body_occlusion_scope']=cache_report['scope']
    report['observation_render_resolution']=[width,height]
    if options.crop_observations:report['scope']+=' Native-image helmet ROI observations mapped to256-square with explicit per-observation pixel-center affine K, preserving much more helmet silhouette detail than256-wide full-frame fitting. Body depth remains the existing coarse cache, conservatively resampled; no extra body-depth accuracy claimed. Before/after are THIS ROI protocol only, not directly comparable to full-frame IoU.'
    report['scope']=report['scope'].replace('Geometry remains old filled hull; no cavity/UV/texture or physical-contact accuracy claim.','Saved canonical geometry from the stated initialization is reused; candidate geometry/material provenance is in its source reports. No physical-contact accuracy claim.')
    if options.opening_direction_weight:report['scope']+=' Explicit worn-helmet opening-direction DESIGN prior follows fitted head-up and eye-forward directions; not a measured cavity or independently validated contact constraint.'
    if options.material_loss_weight:
        report.update(material_class_error_before=float(beforephoto.mean()),material_class_error_after=float(afterphoto.mean()))
        report['scope']+=' Additional actual-image bright/dark material-class loss with an explicit white-outer/dark-lining prior, ambiguous middle luminance/hand pixels ignored; weak opening-up penalty only while worn. This disambiguates candidate orientation modes, not calibrated full photometric matching or true-pose evidence. Input silhouette and material metrics reported separately.'
    if options.skip_coarse:report['scope']=report['scope'].replace('24-way coarse rotations at selected phase/donning frames','Coarse search skipped during warm-started shape/pose alternation')
    if options.rgb_loss_weight:
        report.update(rgb_photometric_error_before=float(beforephoto.mean()),rgb_photometric_error_after=float(afterphoto.mean()))
        report['rgb_support']={'observations_with_pixels':int((rgb_support_counts>0).sum()),'frames_with_pixels':sorted(set(frameids[rgb_support_counts>0].cpu().tolist())),'compared_pixels_across_observations':int(rgb_support_counts.sum())}
        report['scope']+=' Actual baked UV RGB photometric term in assumed linear RGB after saved camera gain/bias. Only observed UV texels, positive nonclipped candidate pixels away from hands and conservative body-depth uncertainty, and rendered foreground are compared. Fixed texture can encode pose bias or lighting; lower input RGB loss does not validate true pose, novel-view quality or physical contact. White/dark proxy disabled in this run.'
    (out/'report.json').write_text(json.dumps(report,indent=2));print('phased full300 fit complete',flush=True)
