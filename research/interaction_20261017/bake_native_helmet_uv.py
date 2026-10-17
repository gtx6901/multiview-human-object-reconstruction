"""Native multitime/multicamera helmet UV baking, with measured-coverage masks."""
import os
from pathlib import Path
import sys,json,time,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')];sys.path.append(os.environ.get("HOI_SHARED_SITE", ""))
import torch,numpy as np,trimesh,nvdiffrast.torch as dr
from PIL import Image
from scipy.ndimage import distance_transform_edt
from helmet_texture_sampling import NativeHelmetSampler,linear_to_srgb
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_texture_native_all300_v2');p.add_argument('--stride',type=int,default=3);p.add_argument('--motion',default='helmet_shell_pose_all300_v1');p.add_argument('--uv',default='helmet_uv_all300_v1');p.add_argument('--policy',choices=['all-time','stage-separated'],default='stage-separated');p.add_argument('--body-depth',default='body_depth_all300_v1');p.add_argument('--consistent-frames',action='store_true');options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
assert options.stride>0 and json.loads((ROOT/'uv_rendering_verification_v1/report.json').read_text())['success']
meshpath=ROOT/options.uv/'mesh.pt';mesh=torch.load(meshpath,weights_only=True);motion=torch.load(ROOT/options.motion/'sequence.pt',weights_only=True);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text())
uvmeta=json.loads((ROOT/options.uv/'report.json').read_text());shellsource=uvmeta.get('source','helmet_shell_all300_v3');shellpath=ROOT/shellsource/'mesh.pt'
assert hashlib.sha256(shellpath.read_bytes()).hexdigest()==uvmeta['source_mesh_sha256']
original=torch.load(shellpath,weights_only=True);tm=trimesh.Trimesh(original['vertices_local'].numpy(),original['faces'].numpy(),process=False)
assert np.allclose(mesh['vertices_local'][mesh['faces'].long()],motion['vertices_local'][motion['faces'].long()],atol=1e-6)
normals=torch.tensor(np.array(tm.vertex_normals)[mesh['source_vertex'].numpy()],device='cuda',dtype=torch.float32)
uv=mesh['uv'].cuda();faces=mesh['faces'].cuda();verts=mesh['vertices_local'].cuda();H=W=1024
context=dr.RasterizeCudaContext();clip=torch.cat([uv*2-1,torch.zeros(len(uv),1,device='cuda'),torch.ones(len(uv),1,device='cuda')],1)[None]
rast,_=dr.rasterize(context,clip.contiguous(),faces.contiguous(),resolution=[H,W]);rast=rast.flip(1).contiguous()
positions,_=dr.interpolate(verts[None].contiguous(),rast,faces);ns,_=dr.interpolate(normals[None].contiguous(),rast,faces)
faceid=rast[0,...,3].long()-1;mask=faceid>=0;y,x=torch.where(mask);fid=faceid[mask];points=positions[0][mask];normal=torch.nn.functional.normalize(ns[0][mask],dim=1);classes=mesh['face_classes'].cuda()[fid].long();M=len(points)
sampler=NativeHelmetSampler(mesh,motion,meta,body_depth=options.body_depth)
# Fit a small approximate camera-response correction using same-surface,
# same-time correspondences. Camera717 fixes the global color gauge.
calib_frames=[30,60,90,120,150,180,210,240,270,299] if options.policy=='all-time' else [180,210,240,270,299]
outer=torch.where(classes==0)[0];sampleids=outer[torch.linspace(0,len(outer)-1,min(2000,len(outer)),device='cuda').long()]
S=len(sampleids);cs=torch.zeros(len(calib_frames),8,S,3,device='cuda');cw=torch.zeros(len(calib_frames),8,S,device='cuda')
with torch.inference_mode():
    for vi,v in enumerate(meta['views']):
        for ti,f in enumerate(calib_frames):
            result=sampler.sample(vi,f,points[sampleids],normal[sampleids],classes[sampleids])
            if result is None:continue
            color,weight,srgb=result
            weight=weight*((srgb>.03).any(1))*(~(srgb>.98).all(1))
            if options.policy=='stage-separated':weight*=((srgb>.5).all(1)&((srgb.max(1).values-srgb.min(1).values)<.2))
            cs[ti,vi]=color;cw[ti,vi]=weight
        print('camera calibration samples',v['serial'],flush=True)
pa=[];pb=[];ca=[];cb=[];pairweights=[];support=np.zeros((8,8),int)
for a in range(8):
    for b in range(a+1,8):
        good=(cw[:,a]>.03)&(cw[:,b]>.03);support[a,b]=support[b,a]=int(good.sum())
        if not good.any():continue
        c1=cs[:,a][good];c2=cs[:,b][good];weights=(cw[:,a][good]*cw[:,b][good]).sqrt()
        pa.append(torch.full((len(c1),),a,device='cuda',dtype=torch.long));pb.append(torch.full((len(c1),),b,device='cuda',dtype=torch.long));ca.append(c1);cb.append(c2);pairweights.append(weights)
assert ca,'No usable multiview color correspondences'
ia=torch.cat(pa);ib=torch.cat(pb);c1=torch.cat(ca);c2=torch.cat(cb);pw=torch.cat(pairweights)
rawgain=torch.nn.Parameter(torch.zeros(8,3,device='cuda'));rawbias=torch.nn.Parameter(torch.zeros(8,3,device='cuda'));reference=1;gate=torch.ones(8,1,device='cuda');gate[reference]=0
priorgain=torch.ones(8,3,device='cuda');white_support=[]
if options.policy=='stage-separated':
    values=cs[:,reference][cw[:,reference]>.03];assert len(values)>10,'Reference lacks white-shell exposure evidence'
    refwhite=values.median(0).values
    for vi in range(8):
        values=cs[:,vi][cw[:,vi]>.03];white_support.append(len(values))
        if len(values)>10:priorgain[vi]=(refwhite/values.median(0).values.clamp_min(.02)).clamp(.25,4)
    priorgain[reference]=1
def correction():
    if options.policy=='all-time':return torch.exp(np.log(2)*torch.tanh(rawgain)*gate),.1*torch.tanh(rawbias)*gate
    return priorgain*torch.exp(.25*torch.tanh(rawgain)*gate),.02*torch.tanh(rawbias)*gate
def objective(g,b):
    residual=c1*g[ia]+b[ia]-c2*g[ib]-b[ib]
    return (torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.03,reduction='none')*pw[:,None]).sum()/(3*pw.sum())
opt=torch.optim.Adam([rawgain,rawbias],lr=.025)
with torch.no_grad():before=float(objective(*correction()))
for step in range(201):
    opt.zero_grad();g,b=correction()
    regularization=.002*(g-1).square().mean()+.02*b.square().mean() if options.policy=='all-time' else .1*(g/priorgain-1).square().mean()+2*b.square().mean()
    loss=objective(g,b)+regularization
    if step<200:loss.backward();opt.step()
gain,bias=[v.detach() for v in correction()];after=float(objective(gain,bias).detach())
calib={'reference_serial':meta['views'][reference]['serial'],'frames':calib_frames,'correspondences':len(c1),'pair_support':support.tolist(),'gain_linear_rgb':gain.cpu().tolist(),'bias_linear_rgb':bias.cpu().tolist(),'robust_color_loss_before':before,'robust_color_loss_after':after,'scope':'Approximate affine response fit in assumed linearized sRGB, same-time surface correspondences, robust loss, fixed reference camera and bounded coefficients. Not spectral/colorimetric calibration or recovered albedo; clipping, pose error and view-dependent lighting remain.'}
(out/'camera_color_correction.json').write_text(json.dumps(calib,indent=2))
calib['appearance_policy']=options.policy;calib['white_shell_support']=white_support;calib['exposure_prior_gain']=priorgain.cpu().tolist()
if options.policy=='stage-separated':
    calib['scope']+=' Restricted to unclipped near-neutral white exterior patches while worn; robust white-patch exposure priors and strong gain/bias regularization address false dark/white correspondences. Material/illumination assumptions explicitly introduced; not unrestricted measured camera calibration.'
(out/'camera_color_correction.json').write_text(json.dumps(calib,indent=2))
topweight=torch.zeros(8,2,M,device='cuda');topcolor=torch.zeros(8,2,M,3,device='cuda');topframe=torch.full((8,2,M),-1,device='cuda',dtype=torch.int16)
frames=sorted(set(list(range(0,300,options.stride))+[299]));used=[]
if options.policy=='stage-separated':frames=[f for f in frames if f<=135 or f>=180]
frame_consistency=[]
if options.consistent_frames:
    assert options.policy=='all-time','Consistency selection must not silently retain stage/color gates'
    with torch.inference_mode():
        for f in frames:
            colors=torch.zeros(8,S,3,device='cuda');weights=torch.zeros(8,S,device='cuda')
            for vi in range(8):
                sample=sampler.sample(vi,f,points[sampleids],normal[sampleids],classes[sampleids])
                if sample is None:continue
                c,w,_=sample;colors[vi]=(c*gain[vi]+bias[vi]).clamp(0,1);weights[vi]=w
            known=weights>.03;supported=known.sum(0)>=3;support=int(supported.sum())
            score=None
            if support>=64:
                masked=colors.masked_fill(~known[...,None],float('nan'));median=torch.nanmedian(masked,dim=0).values.nan_to_num(0)
                error=(colors-median[None]).abs().mean(-1);w=weights*known*supported[None]
                score=float((error*w).sum()/w.sum().clamp_min(1e-8))
            frame_consistency.append({'frame':f,'surface_points_seen_by_at_least3_cameras':support,'corrected_linear_rgb_disagreement':score})
            print('frame color consistency',f,score,flush=True)
    eligible=sorted([r for r in frame_consistency if r['corrected_linear_rgb_disagreement'] is not None and r['corrected_linear_rgb_disagreement']<=.08],key=lambda r:r['corrected_linear_rgb_disagreement'])
    assert eligible,'No sufficiently supported cross-camera-consistent frames; do not bake a fabricated reliable texture'
    chosen={r['frame'] for r in eligible[:max(1,int(np.ceil(.6*len(eligible))))]};frames=sorted(chosen)
    (out/'frame_color_consistency.json').write_text(json.dumps({'records':frame_consistency,'selected_frames':frames,'maximum_disagreement':.08,'kept_fraction_of_eligible':.6,'scope':'Same-time corrected RGB disagreement on common candidate outer-surface points with at least3 visible cameras; at least64 supported points. Top60percent among frames passing.08 assumed-linear-RGB error gate. This filters appearance consistency, not true registration or geometric accuracy; common bias can pass. No held/worn stage restriction or lining dark-pixel gate.'},indent=2))
with torch.inference_mode():
    for vi,v in enumerate(meta['views']):
        count=0
        for f in frames:
            result=sampler.sample(vi,f,points,normal,classes)
            if result is None:continue
            color,weight,srgb=result;color=(color*gain[vi]+bias[vi]).clamp(0,1)
            if options.policy=='stage-separated':
                # Observed design: white exterior when worn; dark lining during inspection.
                # This prevents an uncertain silhouette-only orientation from mixing both.
                admissible=((classes==0)&(f>=180))|((classes==1)&(f<=135)&(srgb.mean(1)<.4))|(classes==2)
                weight*=admissible
            scores=torch.cat([topweight[vi],weight[None]],0);newweight,order=scores.topk(2,dim=0)
            colors=torch.cat([topcolor[vi],color[None]],0);frameids=torch.cat([topframe[vi],torch.full((1,M),f,device='cuda',dtype=torch.int16)],0)
            topweight[vi]=newweight;topcolor[vi]=torch.gather(colors,0,order[...,None].expand(-1,-1,3));topframe[vi]=torch.gather(frameids,0,order);count+=1
        used.append({'serial':v['serial'],'sampled_candidate_frames':count});print('native texture camera complete',v['serial'],flush=True)
    weights=topweight.reshape(16,M);colors=topcolor.reshape(16,M,3);total=weights.sum(0);observed=total>.01;channels=[]
    for channel in range(3):
        values,order=colors[:,:,channel].sort(0);sortedweights=torch.gather(weights,0,order);median_index=(sortedweights.cumsum(0)>=total[None]*.5).long().argmax(0)
        channels.append(torch.gather(values,0,median_index[None])[0])
    rgb=linear_to_srgb(torch.stack(channels,1)).cpu().numpy();observed=observed.cpu().numpy();classids=classes.cpu().numpy();yy=y.cpu().numpy();xx=x.cpu().numpy();faceids=fid.cpu().numpy()
    atlas=np.zeros((H,W,3),np.float32);atlas[yy,xx]=rgb;observedimage=np.zeros((H,W),bool);observedimage[yy,xx]=observed
    classimage=np.full((H,W),-1,np.int16);classimage[yy,xx]=classids;priors=[]
    fallback=[[.85,.85,.85],[.03,.03,.03],[.25,.25,.25]]
    for cls in [0,1,2]:
        region=classimage==cls;known=region&observedimage;unknown=region&~observedimage
        prior=np.median(atlas[known],axis=0) if known.any() else np.array(fallback[cls]);atlas[unknown]=prior
        if known.any():
            distance,nearest=distance_transform_edt(~known,return_indices=True);fill=unknown&(distance<=32);atlas[fill]=atlas[nearest[0][fill],nearest[1][fill]]
        priors.append({'class':cls,'observed_texels':int(known.sum()),'unobserved_texels':int(unknown.sum()),'fallback_or_observed_class_median_srgb':prior.tolist()})
    validimage=classimage>=0;distance,nearest=distance_transform_edt(~validimage,return_indices=True);pad=(~validimage)&(distance<=4);atlas[pad]=atlas[nearest[0][pad],nearest[1][pad]]
    texture=np.uint8(np.clip(atlas,0,1)*255);Image.fromarray(texture).save(out/'helmet_basecolor.png');Image.fromarray(observedimage.astype('uint8')*255).save(out/'observed_texels.png');Image.fromarray((validimage&~observedimage).astype('uint8')*255).save(out/'inferred_texels.png')
    evidence=topframe.cpu().numpy();np.savez_compressed(out/'texture_evidence.npz',texel_y=yy,texel_x=xx,source_frames_per_camera=evidence,source_weights=topweight.cpu().numpy().astype('float16'),classes=classids,observed=observed)
    result={**mesh,'texture':torch.tensor(texture.copy(),dtype=torch.float32)/255};torch.save(result,out/'mesh.pt')
    export=trimesh.Trimesh(mesh['vertices_local'].numpy(),mesh['faces'].numpy(),process=False);export.visual=trimesh.visual.TextureVisuals(uv=mesh['uv'].numpy(),image=Image.fromarray(texture));export.export(out/'helmet_textured.obj')
    area=np.array(tm.area_faces);counts=np.bincount(faceids,minlength=len(area));observedcounts=np.bincount(faceids[observed],minlength=len(area));area_coverage=float((area*observedcounts/np.maximum(counts,1)).sum()/area.sum())
    report={'success':True,'seconds':time.time()-start,'source_mesh_sha256':hashlib.sha256(meshpath.read_bytes()).hexdigest(),'source_motion':options.motion,'texture_resolution':[W,H],'frame_stride':options.stride,'sampled_frames':frames,'views':used,'uv_texels':M,'observed_uv_texels':int(observed.sum()),'observed_uv_fraction':float(observed.mean()),'estimated_area_weighted_observed_fraction':area_coverage,'class_priors':priors,'color_correction':calib,'scope':'Real original native PNG multiframe/multicamera UV appearance, two best-angle observations per camera then channelwise weighted median across up to16 samples. Positive candidate labels, native mesh self-depth and surface-class agreement, conservative body depth and hand-unknown exclusion. Unobserved texels explicitly marked and repaired by nearby observed same-class colors or class medians/design fallbacks. Not calibrated albedo, relighting or proof of correct geometry/texture; pose/segmentation/lighting errors remain. Requires actual real/novel-view rendering and held-out evaluation. No raw footage/model weights published.'}
    report['appearance_policy']=options.policy
    report['source_body_depth']=options.body_depth
    if options.consistent_frames:
        report['frame_consistency_selection']=json.loads((out/'frame_color_consistency.json').read_text())
        report['scope']+=' Candidate frames first ranked by same-time corrected cross-camera color consistency, with explicit support/error gates. No stage or observed-dark-lining selection; common pose/color bias can still survive.'
    report['source_uv']=options.uv;report['source_shell']=shellsource
    if options.policy=='stage-separated':report['scope']+=' Phase/material evidence selection: exterior from worn180–299, lining from held0–135 dark observed pixels, rim from both; donning excluded. This is an explicit visible-material prior for this white-shell/black-lining helmet. Lower observed coverage is reported honestly. No invented logo or claim of resolved photometric pose ambiguity.'
    (out/'report.json').write_text(json.dumps(report,indent=2));print('native multiview UV bake completed',flush=True)
