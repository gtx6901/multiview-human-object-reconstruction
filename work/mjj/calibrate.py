"""Isolated MJJ camera audit/recalibration. Never writes baseline assets.

Coordinates: undistorted full-resolution pixels from capture manifests;
OpenCV world-to-camera matrices. No SMPL/body/motion fitting.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
import inspect

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/mjj/calibration'
JOINTS = np.arange(5, 17)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    caller=Path(inspect.currentframe().f_back.f_code.co_filename)
    if isinstance(data,dict):
        data['_provenance']={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'script':str(caller.relative_to(ROOT)) if caller.is_file() and caller.is_relative_to(ROOT) else str(caller),
            'script_sha256':digest(caller) if caller.is_file() else None,
            'geometry_library_sha256':digest(Path(__file__))}
    previous=OUT/name
    if previous.exists():
        with (OUT/'previous_reports.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'name':name,'sha256':digest(previous),'data':json.loads(previous.read_text(encoding='utf-8'))})+'\n')
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def load(kind):
    manifest = json.loads((ROOT / f'assets/mjj/{kind}/manifest.json').read_text())
    z = np.load(ROOT / f'assets/mjj/keypoints/{kind}.npz', allow_pickle=False)
    cams = [v['serial'] for v in manifest['views']]
    assert cams == list(z['cams']) and len(cams) == 8 and len(set(cams)) == 8
    assert list(z['frames']) == manifest['frames']
    K = np.array([v['K'] for v in manifest['views']])
    W = np.array([v['W'] for v in manifest['views']])
    xy = z['xy'][:, :, JOINTS].astype(float)
    valid = (z['score'][:, :, JOINTS] >= .6) & z['in_image'][:, :, JOINTS]
    valid &= np.isfinite(xy).all(-1)
    return manifest, cams, K, W, xy, valid


def triangulate(K, W, xy, valid):
    """DLT uses each observation only with its own camera (no broadcasting)."""
    P = K @ W[:, :3]
    shape = xy.shape[1:-1]
    xx, vv = xy.reshape(8, -1, 2), valid.reshape(8, -1)
    X = np.full((xx.shape[1], 3), np.nan)
    for i in range(len(X)):
        ids = np.flatnonzero(vv[:, i])
        if len(ids) < 2:
            continue
        A = np.concatenate([np.stack([xx[c, i, 0]*P[c, 2]-P[c, 0],
                                     xx[c, i, 1]*P[c, 2]-P[c, 1]]) for c in ids])
        _, _, vt = np.linalg.svd(A)
        if abs(vt[-1, 3]) > 1e-12:
            X[i] = vt[-1, :3] / vt[-1, 3]
    return X.reshape(*shape, 3)


def project(K, W, X):
    pc = np.einsum('cij,...j->c...i', W[:, :3, :3], X)
    pc += W[:, :3, 3].reshape(8, *([1]*(X.ndim-1)), 3)
    uv = np.einsum('cij,c...j->c...i', K, pc)
    return uv[..., :2] / np.where(abs(uv[..., 2:]) < 1e-10, 1e-10, uv[..., 2:]), pc[..., 2]


def stats(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return {'n':len(a), 'median_px':float(np.median(a)) if len(a) else None,
            'p95_px':float(np.percentile(a,95)) if len(a) else None,
            'mean_px':float(np.mean(a)) if len(a) else None,
            'fraction_le_5px':float(np.mean(a <= 5)) if len(a) else None}


def evaluate(K, W, xy, valid, cams):
    X = triangulate(K, W, xy, valid)
    uv, depth = project(K, W, X)
    e = np.linalg.norm(uv-xy, axis=-1)
    allcam = {c:stats(e[i][valid[i]]) for i,c in enumerate(cams)}
    # Leave one camera's pixels out of 3D reconstruction; camera poses stay fixed.
    loo = {}
    for c in range(8):
        mask = valid.copy(); mask[c] = False
        Xc = triangulate(K,W,xy,mask)
        uvc, dc = project(K,W,Xc)
        good = valid[c] & np.isfinite(Xc).all(-1) & (dc[c] > 0)
        loo[cams[c]] = stats(np.linalg.norm(uvc[c]-xy[c],axis=-1)[good])
    return {'all_observations':stats(e[valid]), 'by_camera':allcam,
            'positive_depth_fraction':float(np.mean(depth[valid]>0)),
            'leave_camera_out':loo}


def audit():
    result = {'created':time.strftime('%Y-%m-%d %H:%M:%S'), 'coordinate_system':'undistorted 2048x1536; OpenCV w2c',
              'joint_ids':JOINTS.tolist(), 'minimum_score':.6, 'datasets':{},
              'inputs':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'data/20260827_ex.txt',
                  ROOT/'assets/mjj/static/manifest.json', ROOT/'assets/mjj/dynamic/manifest.json',
                  ROOT/'assets/mjj/keypoints/static.npz', ROOT/'assets/mjj/keypoints/dynamic.npz']}}
    for kind in ['static','dynamic']:
        man,cams,K,W,xy,valid = load(kind)
        result['datasets'][kind] = {}
        for name,T in [('as_w2c',W),('inverse_as_w2c',np.linalg.inv(W))]:
            r = evaluate(K,T,xy,valid,cams)
            result['datasets'][kind][name] = r
            print(kind,name,json.dumps(r['all_observations']), 'positive',r['positive_depth_fraction'],flush=True)
            print('camera medians',[round(r['by_camera'][c]['median_px'],2) for c in cams],flush=True)
        # Intrinsics must reproduce exactly the preprocessing camera matrix.
        for v in man['views']:
            Ku,_=cv2.getOptimalNewCameraMatrix(np.array(v['K_native']),np.array(v['distortion_opencv5']),
                                             (2048,1536),0)
            assert np.allclose(Ku,v['K'],rtol=0,atol=1e-8), v['serial']
    write_json('audit.json',result)


def skew(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def sampson(Ki,Kj,T,pi,pj):
    F=np.linalg.inv(Kj).T @ skew(T[:3,3]) @ T[:3,:3] @ np.linalg.inv(Ki)
    a=np.c_[pi,np.ones(len(pi))]; b=np.c_[pj,np.ones(len(pj))]
    Fa=a@F.T; Fb=b@F
    return np.abs(np.sum(b*Fa,axis=1))/np.sqrt(np.sum(Fa[:,:2]**2+Fb[:,:2]**2,axis=1)+1e-30)


def features():
    cv2.setNumThreads(4); cv2.setRNGSeed(20261027)
    man,cams,K,W,xy,valid=load('static')
    detector=cv2.SIFT_create(nfeatures=12000,contrastThreshold=.025)
    feats=[]
    for c,serial in enumerate(cams):
        path=ROOT/f'assets/mjj/static/{serial}/0050.png'
        im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        mask=np.full(im.shape,255,np.uint8)
        z=np.load(ROOT/'assets/mjj/keypoints/static.npz')
        box=z['boxes'][c,1].copy(); box[[0,1]]-=70; box[[2,3]]+=70
        x0,y0,x1,y1=box.astype(int)
        mask[max(y0,0):min(y1,1536),max(x0,0):min(x1,2048)]=0
        mask[:20]=0;mask[-20:]=0;mask[:,:20]=0;mask[:,-20:]=0
        kp,desc=detector.detectAndCompute(im,mask)
        pts=np.array([p.pt for p in kp],np.float64)
        feats.append((pts,desc))
        print('SIFT',serial,len(pts),flush=True)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    report=[]; cache={}
    for i,j in itertools.combinations(range(8),2):
        pi,di=feats[i];pj,dj=feats[j]
        m=matcher.knnMatch(di,dj,k=2); rev=matcher.knnMatch(dj,di,k=2)
        a={p.queryIdx:p.trainIdx for p,q in m if p.distance<.72*q.distance}
        b={p.queryIdx:p.trainIdx for p,q in rev if p.distance<.72*q.distance}
        ids=np.array([(p,q) for p,q in a.items() if b.get(q)==p],int).reshape(-1,2)
        cache[f'raw_{i}_{j}']=ids.copy()
        r={'i':i,'j':j,'cameras':[cams[i],cams[j]],'mutual_ratio_matches':len(ids)}
        if len(ids)>=12:
            p=pi[ids[:,0]];q=pj[ids[:,1]]
            # Unequal intrinsics: normalize each side, use normalized threshold.
            pn=cv2.undistortPoints(p[:,None],K[i],None).reshape(-1,2)
            qn=cv2.undistortPoints(q[:,None],K[j],None).reshape(-1,2)
            E,inl=cv2.findEssentialMat(pn,qn,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1.5/1900)
            if E is not None and E.shape==(3,3):
                nin,R,t,front=cv2.recoverPose(E,pn,qn,np.eye(3),mask=inl.copy())
                ids=ids[front.ravel()>0];p=pi[ids[:,0]];q=pj[ids[:,1]]
                T=W[j]@np.linalg.inv(W[i])
                angle=float(Rotation.from_matrix(R@T[:3,:3].T).magnitude()*180/np.pi)
                r.update({'essential_inliers':int(inl.sum()),'cheirality_inliers':int(nin),
                          'rotation_difference_deg':angle,'relative_R':R.tolist(),'relative_t_direction':t.ravel().tolist(),
                          'old_w2c_sampson':stats(sampson(K[i],K[j],T,p,q))})
                cache[f'match_{i}_{j}']=ids
        report.append(r)
        print('pair',cams[i],cams[j],r.get('mutual_ratio_matches'),r.get('cheirality_inliers'),r.get('rotation_difference_deg'),flush=True)
    for c,(p,d) in enumerate(feats):cache[f'points_{c}']=p
    np.savez_compressed(OUT/'features.npz',**cache)
    write_json('features.json',{'pairs':report,'source':'static frame 50; SIFT mutual .72 ratio; padded person exclusion',
               'essential_threshold_px':1.5,'seed':20261027})


def bundle(K, W, xy, valid, max_nfev=250, min_views=3, f_scale=3.):
    """Fixed K, camera 0 gauge, baseline norm gauge; unconstrained 3D correspondences."""
    shape=xy.shape[1:-1]
    xx=xy.reshape(8,-1,2); vv=valid.reshape(8,-1)
    X=triangulate(K,W,xy,valid).reshape(-1,3)
    good=np.isfinite(X).all(-1)&(vv.sum(0)>=min_views)
    xx=xx[:,good]; vv=vv[:,good]; X=X[good]
    # Normalize optimization scale; exported transforms retain the original gauge.
    scale=np.linalg.norm(W[1,:3,3]); T=W.copy(); T[:,:3,3]/=scale; X/=scale
    rv=Rotation.from_matrix(T[1:,:3,:3]).as_rotvec()
    c0=np.c_[rv,T[1:,:3,3]].ravel()
    start=np.r_[c0,X.ravel()]
    ci,pi=np.nonzero(vv); observed=xx[ci,pi]
    nc=42; nobs=len(ci)
    sparsity=lil_matrix((2*nobs+1,nc+3*len(X)),dtype=int)
    for k in range(nobs):
        if ci[k]>0:sparsity[2*k:2*k+2,6*(ci[k]-1):6*ci[k]]=1
        sparsity[2*k:2*k+2,nc+3*pi[k]:nc+3*pi[k]+3]=1
    sparsity[-1,3:6]=1
    def unpack(v):
        pars=v[:nc].reshape(7,6)
        R=np.concatenate([T[0:1,:3,:3],Rotation.from_rotvec(pars[:,:3]).as_matrix()])
        t=np.concatenate([T[0:1,:3,3],pars[:,3:]])
        return R,t,v[nc:].reshape(-1,3)
    def residual(v):
        R,t,pts=unpack(v)
        pc=np.einsum('nij,nj->ni',R[ci],pts[pi])+t[ci]
        uv=np.einsum('nij,nj->ni',K[ci],pc)
        z=uv[:,2:];z=np.where(abs(z)<1e-8,1e-8,z)
        err=uv[:,:2]/z-observed
        return np.r_[err.ravel(),1000*(np.linalg.norm(t[1])-np.linalg.norm(T[1,:3,3]))]
    print('bundle start',nobs,'observations',len(X),'3D points',flush=True)
    opt=least_squares(residual,start,jac_sparsity=sparsity.tocsr(),x_scale='jac',
                      loss='soft_l1',f_scale=f_scale,max_nfev=max_nfev,ftol=1e-7,verbose=1)
    R,t,pts=unpack(opt.x)
    output=np.tile(np.eye(4),(8,1,1));output[:,:3,:3]=R;output[:,:3,3]=t*scale
    print('bundle complete',opt.nfev,'cost',opt.cost,'median',np.median(np.linalg.norm(residual(opt.x)[:-1].reshape(-1,2),axis=1)),flush=True)
    return output,{'cost':float(opt.cost),'optimality':float(opt.optimality),'nfev':opt.nfev,
                   'status':int(opt.status),'message':str(opt.message),'points':len(X),'observations':nobs}


def keypoint_bundle():
    man,cams,K,W,xy,valid=load('dynamic')
    train=np.arange(0,100,5); test=np.setdiff1d(np.arange(100),train)
    # This estimates cameras from corresponding image points; no body model or temporal fit.
    candidate,optim=bundle(K,W,xy[:,train],valid[:,train],max_nfev=3000)
    np.savez_compressed(OUT/'candidate_keypoints.npz',W=candidate,K=K,cams=np.array(cams),
                        train_frames=train,test_frames=test)
    result={'optimization':optim,'source':'dynamic 20 frames, 12 joint correspondences; camera-only bundle with free triangulated points',
            'limitations':['Not a body or motion asset','Unknown metric scale','RTMW correspondences can have occlusion bias',
                            'Requires independent static-background and held-out-camera verification'],
            'train_frames':train.tolist(),'test_frames':test.tolist(),
            'train':evaluate(K,candidate,xy[:,train],valid[:,train],cams),
            'heldout_dynamic':evaluate(K,candidate,xy[:,test],valid[:,test],cams)}
    ms,cs,ks,ws,xs,vs=load('static')
    result['independent_static']=evaluate(ks,candidate,xs,vs,cs)
    result['accepted']=False
    result['status']='candidate_only_pending_independent_geometric_verification'
    write_json('candidate_keypoints.json',result)
    for group in ['train','heldout_dynamic','independent_static']:
        print(group,json.dumps(result[group]['all_observations']),flush=True)
        print('camera medians',[round(result[group]['by_camera'][c]['median_px'],2) for c in cams],flush=True)


def check_features():
    man,cams,K,W,xy,valid=load('static')
    candidate=np.load(OUT/'candidate_keypoints.npz')['W']
    cache=np.load(OUT/'features.npz')
    pairs=json.loads((OUT/'features.json').read_text())['pairs']
    report=[]
    for r in pairs:
        i,j=r['i'],r['j']
        key=f'match_{i}_{j}'
        if key not in cache:continue
        ids=cache[key];pi=cache[f'points_{i}'][ids[:,0]];pj=cache[f'points_{j}'][ids[:,1]]
        e=sampson(K[i],K[j],candidate[j]@np.linalg.inv(candidate[i]),pi,pj)
        item={'cameras':[cams[i],cams[j]],'sampson_px':stats(e)}
        report.append(item)
        print(item,flush=True)
    write_json('background_check.json',{'pairs':report,'independent_of_keypoint_optimization':True,
                 'warning':'Essential-RANSAC inliers can be wrong with few matches or planar geometry; inspect match overlays.'})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['audit','features','keypoint-bundle','check-features'])
    args=ap.parse_args()
    {'audit':audit,'features':features,'keypoint-bundle':keypoint_bundle,'check-features':check_features}[args.mode]()
