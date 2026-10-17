"""All-time/all-view object-coordinate silhouette voting with unknown-aware support."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')];sys.path.append(os.environ.get("HOI_SHARED_SITE", ""))
import torch,numpy as np,cv2,trimesh,pymeshlab
from PIL import Image
from scipy.ndimage import label,binary_fill_holes
from skimage.measure import marching_cubes
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_shape_vote_all300_v1');p.add_argument('--motion',default='helmet_phased_all300_v1');p.add_argument('--grid',type=int,default=96);p.add_argument('--ratio',type=float,default=.85);p.add_argument('--bounds',default='helmet_initial_all300_v1');p.add_argument('--body-depth',default='body_depth_all300_v1');p.add_argument('--native-observations',action='store_true');p.add_argument('--frame-range',type=int,nargs=2,default=[0,300]);options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
motion=torch.load(ROOT/options.motion/'sequence.pt',weights_only=True);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());maskroot=ROOT/'sam2_all300_v1';obs=json.loads((maskroot/'report.json').read_text());records={(r['serial'],r['frame']):r for r in obs['records']}
base=torch.load(ROOT/options.bounds/'sequence.pt',weights_only=True)['vertices_local'];lo=base.min(0).values;hi=base.max(0).values;extent=(hi-lo)*1.2;origin=(lo+hi-extent)/2;n=options.grid;spacing=extent/(n-1)
grid=torch.stack(torch.meshgrid(*[torch.arange(n,device='cuda') for _ in range(3)],indexing='ij'),-1).reshape(-1,3);points=origin.cuda()+grid*spacing.cuda();M=len(points)
positive=torch.zeros(M,device='cuda');known=torch.zeros_like(positive);view_support=torch.zeros(M,device='cuda',dtype=torch.int16)
R=motion['R'].cuda();t=motion['translation_world'].cuda();scale=motion['logscale'].exp().cuda();width,height=256,192;A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
if options.native_observations:width,height=2048,1536;A=np.eye(3)
assert 0<=options.frame_range[0]<options.frame_range[1]<=300
observations=[]
with torch.inference_mode():
    for v in meta['views']:
        K=torch.tensor(A@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
        cache=np.load(ROOT/options.body_depth/f"{v['serial']}.npz");depth=cache['depth'];alpha=cache['alpha'];hands=cache['hand_unknown']
        valid=cv2.resize(np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/'valid.png')),(width,height),interpolation=cv2.INTER_NEAREST)>127
        hits=torch.zeros(M,device='cuda',dtype=torch.bool);used=0
        for f in range(*options.frame_range):
            rec=records[(v['serial'],f)]
            if rec['mask'] is None:continue
            labels=cv2.resize(np.asarray(Image.open(maskroot/rec['labels'])),(width,height),interpolation=cv2.INTER_NEAREST)
            person=cv2.resize(np.asarray(Image.open(maskroot/rec['person_mask'])),(width,height),interpolation=cv2.INTER_NEAREST)>127
            mask=cv2.resize(np.asarray(Image.open(maskroot/rec['mask'])),(width,height),interpolation=cv2.INTER_NEAREST)>127
            roi=np.zeros((height,width),dtype=bool);x0,y0,x1,y1=np.array(rec['roi_xyxy'])*(1 if options.native_observations else .125)
            roi[max(0,int(np.floor(y0))):min(height,int(np.ceil(y1))),max(0,int(np.floor(x0))):min(width,int(np.ceil(x1)))]=True
            guard=17 if options.native_observations else 3
            restore=(labels==255)&person&~(cv2.dilate(mask.astype('uint8'),np.ones((guard,guard),np.uint8))>0)&valid&roi
            if not options.native_observations:restore&=(hands[f]==0)&(alpha[f]>0)
            lab=torch.tensor(labels,device='cuda');elig=torch.tensor(restore,device='cuda');bd=torch.tensor(depth[f],device='cuda')
            world=points@R[f].T*scale+t[f];pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;uv=q[:,:2]/q[:,2:].clamp_min(.001);xy=torch.floor(uv+.5).long()
            inside=(pc[:,2]>.01)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height);x=xy[:,0].clamp(0,width-1);y=xy[:,1].clamp(0,height-1)
            sampled=lab[y,x];pos=inside&(sampled==1);neg=inside&(sampled==0)
            if options.native_observations:
                cxy=torch.floor((uv+.5)*.125).long();cx=cxy[:,0].clamp(0,255);cy=cxy[:,1].clamp(0,191);ha=torch.tensor(hands[f],device='cuda');al=torch.tensor(alpha[f],device='cuda')
                restore_neg=inside&elig[y,x]&(ha[cy,cx]==0)&(al[cy,cx]>0)&(pc[:,2]<bd[cy,cx]-.01)
            else:restore_neg=inside&elig[y,x]&(pc[:,2]<bd[y,x]-.01)
            positive+=pos.float();known+=(pos|neg).float()+.5*restore_neg.float();hits|=pos;used+=1
        view_support+=hits.short();observations.append({'serial':v['serial'],'observations':used});print('voted',v['serial'],flush=True)
    ratio=positive/known.clamp_min(1);occupancy=(ratio>=options.ratio)&(positive>=12)&(view_support>=3)
    occ=occupancy.reshape(n,n,n).cpu().numpy();occ[[0,-1]]=False;occ[:,[0,-1]]=False;occ[:,:,[0,-1]]=False
    components,count=label(occ);sizes=np.bincount(components.ravel());sizes[0]=0
    diagnostics={'grid':n,'ratio':options.ratio,'occupied_voxels':int(occ.sum()),'positive_max':float(positive.max()),'known_max':float(known.max()),'ratio_quantiles_supported':np.quantile(ratio[(positive>=12)&(view_support>=3)].cpu().numpy(),[0,.25,.5,.75,.9,.95,.99,1]).tolist()}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    assert count>0 and sizes.max()>20,'No supported shape at fixed vote threshold; diagnose pose/mask disagreement, do not fabricate geometry'
    occupancy=binary_fill_holes(components==sizes.argmax())
    verts,faces,_,_=marching_cubes(occupancy.astype('float32'),.5,spacing=spacing.numpy());verts+=origin.numpy()
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(verts.astype('float64'),faces.astype('int32')))
    if len(faces)>16000:ms.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=16000,preserveboundary=True,preservetopology=True)
    mesh=ms.current_mesh();tm=trimesh.Trimesh(mesh.vertex_matrix(),mesh.face_matrix(),process=False);tm.export(out/'local_candidate.obj')
    result={**motion,'vertices_local':torch.tensor(np.array(tm.vertices),dtype=torch.float32),'faces':torch.tensor(np.array(tm.faces),dtype=torch.int32)}
    torch.save(result,out/'sequence.pt')
    torch.save({'vertices_local':result['vertices_local'],'faces':result['faces'],'colors':torch.full_like(result['vertices_local'],.65)},out/'mesh.pt')
    report={'success':True,'observations':observations,'total_observations':sum(r['observations'] for r in observations),'seconds':time.time()-start,'diagnostics':diagnostics,'vertices':len(tm.vertices),'faces':len(tm.faces),'watertight':bool(tm.is_watertight),'source_motion':options.motion,'scope':'All300-time/all8-view candidate silhouette voting in one object coordinate system, unknowns excluded and conservative half-weight body-depth negatives restored away from hands/boundaries. Fixed.85 vote ratio,≥12 positive observations and≥3 contributing cameras. Bounded grid prior/largest3D component and cavity filling produce a CLOSED preliminary hull, not a recovered hollow helmet. Pose errors/mask bias can shrink/distort it. Must alternate pose/shape refinement and inspect before cavity editing/UV/texture; this is not accepted geometry or held-out evidence.'}
    report['search_bounds_source']=options.bounds
    report['sources']=vars(options);report['observation_resolution']=[width,height]
    if options.native_observations:report['scope']+=' Native2048x1536 labels sampled directly at projected voxels; no full-frame256 silhouette downsampling. Coarse body cache remains256x192 and is sampled with its explicit pixel-center mapping. This improves observation sampling only, not mask correctness or recovered cavity truth.'
    if options.frame_range!=[0,300]:report['scope']=report['scope'].replace('All300-time/all8-view','Selected-time-window/all8-view')+' Frame window is explicit in sources; not full300 all-time fusion.'
    (out/'report.json').write_text(json.dumps(report,indent=2));print('all-time shape vote complete',flush=True)
