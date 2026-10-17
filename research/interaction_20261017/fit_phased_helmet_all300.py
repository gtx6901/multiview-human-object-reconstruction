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
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_phased_all300_v1');p.add_argument('--steps',type=int,default=100);p.add_argument('--initial',default='helmet_initial_all300_v1');p.add_argument('--phase-motion');p.add_argument('--skip-coarse',action='store_true');p.add_argument('--material-loss-weight',type=float,default=0.);p.add_argument('--material-mesh');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
maskroot=ROOT/'sam2_all300_v1';observations=json.loads((maskroot/'report.json').read_text());assert observations['success']
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());initial=torch.load(ROOT/options.initial/'sequence.pt',weights_only=True)
phaseinfo=json.loads((ROOT/'helmet_phase_priors_all300_v1/report.json').read_text());anchor=torch.load(ROOT/'helmet_phase_priors_all300_v1/anchors.pt',weights_only=True)
vertices=initial['vertices_local'].cuda();faces=initial['faces'].cuda();scale=initial['logscale'].exp().cuda();renderer=MeshRenderer();width,height=256,192
A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
Ks=torch.tensor(np.array([A@np.array(v['K']) for v in meta['views']]),device='cuda',dtype=torch.float32);Ws=torch.tensor([v['W'] for v in meta['views']],device='cuda',dtype=torch.float32)
records={(r['serial'],r['frame']):r for r in observations['records']};frames=[];views=[];labels=[];eligible=[];bodydepth=[];bodyalpha=[];photo_targets=[];photo_confidence=[]
material_colors=None;opening_normal=None
if options.material_loss_weight:
    assert options.material_mesh
    materialmesh=torch.load(ROOT/options.material_mesh/'mesh.pt',weights_only=True)
    assert torch.allclose(materialmesh['vertices_local'],initial['vertices_local']) and torch.equal(materialmesh['faces'],initial['faces'])
    material_colors=torch.full_like(vertices,.05);material_colors[:materialmesh['outer_vertex_count']]=.9
    opening_normal=torch.tensor(json.loads((ROOT/options.material_mesh/'report.json').read_text())['opening_plane_normal_local'],device='cuda',dtype=torch.float32)
for vi,v in enumerate(meta['views']):
    cache=np.load(ROOT/'body_depth_all300_v1'/f"{v['serial']}.npz");bd=cache['depth'];ba=cache['alpha'];hand=cache['hand_unknown']
    assert np.allclose(cache['K'],Ks[vi].cpu().numpy()) and np.allclose(cache['W'],Ws[vi].cpu().numpy())
    valid=cv2.resize(np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/'valid.png')),(width,height),interpolation=cv2.INTER_NEAREST)>127
    for f in range(300):
        r=records[(v['serial'],f)]
        if not r['mask']:continue
        lab=cv2.resize(np.asarray(Image.open(maskroot/r['labels'])),(width,height),interpolation=cv2.INTER_NEAREST)
        if (lab==1).sum()<10:continue
        person=cv2.resize(np.asarray(Image.open(maskroot/r['person_mask'])),(width,height),interpolation=cv2.INTER_NEAREST)>127
        obj=cv2.resize(np.asarray(Image.open(maskroot/r['mask'])),(width,height),interpolation=cv2.INTER_NEAREST)>127
        boundary_guard=cv2.dilate(obj.astype('uint8'),np.ones((3,3),np.uint8))>0
        roi=np.zeros_like(valid);x0,y0,x1,y1=np.array(r['roi_xyxy'])*.125
        roi[max(0,int(np.floor(y0))):min(height,int(np.ceil(y1))),max(0,int(np.floor(x0))):min(width,int(np.ceil(x1)))]=True
        restore=(lab==255)&person&~boundary_guard&(hand[f]==0)&valid&roi&(ba[f]>0)
        frames.append(f);views.append(vi);labels.append(lab);eligible.append(restore);bodydepth.append(bd[f]);bodyalpha.append(ba[f])
        if options.material_loss_weight:
            rgb=cv2.resize(np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB')),(width,height),interpolation=cv2.INTER_AREA).astype('float32')/255
            luminance=rgb@np.array([.2126,.7152,.0722]);white=max(.6,float(np.quantile(luminance[lab==1],.95)));normalized=luminance/white
            photo_targets.append(normalized>.7);photo_confidence.append((lab==1)&((normalized>.7)|(normalized<.2))&(hand[f]==0))
    print('prepared masks/depth',v['serial'],flush=True)
labels=torch.tensor(np.array(labels));eligible=torch.tensor(np.array(eligible));bodydepth=torch.tensor(np.array(bodydepth));N=len(frames)
if options.material_loss_weight:photo_targets=torch.tensor(np.array(photo_targets));photo_confidence=torch.tensor(np.array(photo_confidence))
assert N>1500 and set(frames)==set(range(300))
frameids=torch.tensor(frames,device='cuda');viewids=torch.tensor(views,device='cuda');R0=initial['R'].cuda();t0=initial['translation_world'].cuda();phases=phaseinfo['phases'];P=len(phases)
phaseids=torch.full((300,),-1,device='cuda',dtype=torch.long);ar=torch.eye(3,device='cuda').expand(300,-1,-1).clone();at=torch.zeros(300,3,device='cuda')
relR=[];relt=[]
for i,phase in enumerate(phases):
    a,b,j=phase['start'],phase['stop_exclusive'],phase['joint'];assert (phaseids[a:b]<0).all()
    phaseids[a:b]=i;ar[a:b]=anchor['global_joint_R'][a:b,j].cuda();at[a:b]=anchor['joints_world'][a:b,j].cuda()
    relR.append(phase['relative_R']);relt.append(phase['relative_translation'])
relR=torch.tensor(relR,device='cuda',dtype=torch.float32);relt=torch.tensor(relt,device='cuda',dtype=torch.float32)
attached=phaseids>=0;pid=phaseids.clamp_min(0);rraw=torch.nn.Parameter(torch.zeros(300,3,device='cuda'));traw=torch.nn.Parameter(torch.zeros(300,3,device='cuda'));pr=torch.nn.Parameter(torch.zeros(P,3,device='cuda'));pt=torch.nn.Parameter(torch.zeros(P,3,device='cuda'))
if options.phase_motion:
    previous=json.loads((ROOT/options.phase_motion/'report.json').read_text())
    assert [(x['start'],x['stop_exclusive'],x['joint']) for x in previous['phases']]==[(x['start'],x['stop_exclusive'],x['joint']) for x in phases]
    relR=torch.tensor(previous['shared_relative_R'],device='cuda',dtype=torch.float32);relt=torch.tensor(previous['shared_relative_translation'],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        priorR=ar@relR[pid];priort=at+(ar@relt[pid,...,None])[...,0]
        residual=matrix_to_axis_angle(R0@priorR.transpose(-1,-2))
        rraw[attached]=torch.atanh((residual[attached]/.35).clamp(-.999999,.999999))
        traw[attached]=.015*torch.atanh(((t0-priort)[attached]/.015).clamp(-.999999,.999999))
def trajectory():
    phaseR=ar@(axis_angle_to_matrix(pr[pid])@relR[pid]);phaset=at+(ar@(relt[pid]+pt[pid])[...,None])[...,0]
    residualR=torch.where(attached[:,None],.35*torch.tanh(rraw),rraw)
    residualt=torch.where(attached[:,None],.015*torch.tanh(traw/.015),traw)
    R=axis_angle_to_matrix(residualR)@torch.where(attached[:,None,None],phaseR,R0)
    t=torch.where(attached[:,None],phaset,t0)+residualt
    return R,t,residualR,residualt
def cost(R,t,ids):
    vi=viewids[ids];f=frameids[ids]
    world=torch.einsum('bij,nj->bni',R[f],vertices)*scale+t[f,None]
    render=renderer.depth_alpha_batch(world,faces,Ks[vi],Ws[vi],width,height,colors=material_colors)
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
    return 1-iou+options.material_loss_weight*photo,iou,restore.sum((1,2)),photo
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
with torch.no_grad():before,restored_before,beforephoto=metrics();before=before.cpu();beforephoto=beforephoto.cpu()
opt=torch.optim.Adam([{'params':[rraw,pr],'lr':.018},{'params':[traw,pt],'lr':.0015}])
for step in range(options.steps+1):
    opt.zero_grad();data_sum=0.;iou_sum=0.;photo_sum=0.
    for ids in chunks:
        R,t,_,_=trajectory();loss,iou,_,photo=cost(R,t,ids);chunkloss=loss.sum()/N;data_sum+=float(chunkloss.detach());iou_sum+=float(iou.detach().sum())/N;photo_sum+=float(photo.detach().sum())/N
        if step<options.steps:chunkloss.backward()
    R,t,rr,tt=trajectory()
    regularization=2*(R[2:]-2*R[1:-1]+R[:-2]).square().mean()+1000*(t[2:]-2*t[1:-1]+t[:-2]).square().mean()+.002*rr.square().mean()+10*tt.square().mean()
    if opening_normal is not None:
        worn=torch.zeros(300,dtype=torch.bool,device='cuda')
        for phase in phases:
            if phase['joint']==15:worn[phase['start']:phase['stop_exclusive']]=True
        up=torch.nn.functional.normalize(anchor['joints_world'][:,15].cuda()-anchor['joints_world'][:,12].cuda(),dim=1)
        alignment=((R@opening_normal)*up).sum(1)
        regularization+=.1*torch.relu(.45-alignment[worn]).square().mean()
    value=data_sum+float(regularization.detach())
    if best is None or value<best[0]:best=(value,[x.detach().clone() for x in [rraw,traw,pr,pt]])
    if step<options.steps:regularization.backward();opt.step()
    for paramgroup,lr in zip(opt.param_groups,[.018,.0015]):paramgroup['lr']=lr*(1-.8*step/options.steps)
    if step%20==0:
        row={'step':step,'loss':value,'input_soft_iou':iou_sum,'material_class_error':photo_sum};history.append(row);print(json.dumps(row),flush=True)
with torch.no_grad():
    for x,b in zip([rraw,traw,pr,pt],best[1]):x.copy_(b)
    R,t,rr,tt=trajectory();after,restored,afterphoto=metrics();assert torch.isfinite(R).all() and torch.isfinite(t).all()
    assert torch.allclose(torch.linalg.det(R),torch.ones(300,device='cuda'),atol=1e-4)
    sequence={**initial,'R':R.cpu(),'translation_world':t.cpu()};torch.save(sequence,out/'sequence.pt')
    np.savez_compressed(out/'observation_metrics.npz',frames=frames,views=views,before_iou=before,after_iou=after.cpu(),restored_negative_pixels=restored.cpu())
    report={'success':True,'frames':300,'observations':N,'seconds':time.time()-start,'history':history,'coarse_search':coarse,'phases':phases,'mean_input_soft_iou_before':float(before.mean()),'mean_input_soft_iou_after':float(after.mean()),'restored_negative_pixels':int(restored.sum()),'shared_relative_R':(axis_angle_to_matrix(pr)@relR).cpu().tolist(),'shared_relative_translation':(relt+pt).cpu().tolist(),'max_rotation_residual_rad':float(rr[attached].norm(dim=1).max()),'max_translation_residual':float(tt[attached].norm(dim=1).max()),'scope':'All8 main experiment; every300 frame uses real SAM2 candidate masks. Proposed stable-grasp/head-relative phases share transforms with small bounded per-frame residuals; other intervals free.24-way coarse rotations at selected phase/donning frames, original calibrated cameras/fixed shared scale. Tri-state silhouette loss restores half-weight negative evidence only when the predicted object is >.01 normalized-world units in front of a fitted bare-body mesh, away from hands and boundaries. Model/mask/rotation ambiguity remains. Before/after use same current mask/depth protocol after coarse initialization, not prior masks or held-out evaluation. Geometry remains old filled hull; no cavity/UV/texture or physical-contact accuracy claim.'}
    report['sources']=vars(options)
    if options.material_loss_weight:
        report.update(material_class_error_before=float(beforephoto.mean()),material_class_error_after=float(afterphoto.mean()))
        report['scope']+=' Additional actual-image bright/dark material-class loss with an explicit white-outer/dark-lining prior, ambiguous middle luminance/hand pixels ignored; weak opening-up penalty only while worn. This disambiguates candidate orientation modes, not calibrated full photometric matching or true-pose evidence. Input silhouette and material metrics reported separately.'
    if options.skip_coarse:report['scope']=report['scope'].replace('24-way coarse rotations at selected phase/donning frames','Coarse search skipped during warm-started shape/pose alternation')
    (out/'report.json').write_text(json.dumps(report,indent=2));print('phased full300 fit complete',flush=True)
