"""Camera acceptance diagnostics from robust image-point triangulation only.

No body, shape, root, joint rotation, skeleton, or motion is optimized here.
"""
import argparse
import itertools
import numpy as np
from scipy.optimize import least_squares
import calibrate as c


def dlt(P,xy):
    A=np.stack([row for p,(x,y) in zip(P,xy) for row in [x*p[2]-p[0],y*p[2]-p[1]]])
    _,_,vt=np.linalg.svd(A)
    return vt[-1,:3]/vt[-1,3] if abs(vt[-1,3])>1e-12 else np.full(3,np.nan)


def solve(K,W,xy,valid,gate=5,minimum=3):
    P=K@W[:,:3];shape=xy.shape[1:-1]
    xx=xy.reshape(8,-1,2);vv=valid.reshape(8,-1)
    xyz=np.full((xx.shape[1],3),np.nan);support=np.zeros(xx.shape[1],int)
    inlier=np.zeros(vv.shape,bool);errors=np.full(vv.shape,np.nan)
    centers=-np.einsum('cji,cj->ci',W[:,:3,:3],W[:,:3,3])
    for p in range(xx.shape[1]):
        ids=np.flatnonzero(vv[:,p]);best=None
        if len(ids)<minimum:continue
        def project(X):
            h=P[ids]@np.r_[X,1]
            uv=h[:,:2]/h[:,2:]
            return uv,h[:,2]
        for pair in itertools.combinations(ids,2):
            pair=list(pair);X=dlt(P[pair],xx[pair,p])
            if not np.isfinite(X).all():continue
            uv,z=project(X);e=np.linalg.norm(uv-xx[ids,p],axis=1)
            keep=(e<=gate)&(z>0);n=keep.sum()
            if n<minimum:continue
            rank=(n,-float(np.median(e[keep])))
            if best is None or rank>best[0]:best=(rank,keep)
        if best is None:continue
        keep=best[1]
        for _ in range(3):
            selected=ids[keep];X=dlt(P[selected],xx[selected,p])
            def fun(v):
                h=P[selected]@np.r_[v,1]
                return (h[:,:2]/h[:,2:]-xx[selected,p]).ravel()
            X=least_squares(fun,X,max_nfev=30).x
            uv,z=project(X);e=np.linalg.norm(uv-xx[ids,p],axis=1)
            new=(e<=gate)&(z>0)
            if np.array_equal(new,keep):break
            keep=new
            if keep.sum()<minimum:break
        if keep.sum()<minimum:continue
        rays=centers[ids[keep]]-X;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        angle=np.rad2deg(np.arccos(np.clip((rays@rays.T).min(),-1,1)))
        if angle<1.5:continue
        # Retain every residual for coverage, including rejected detections.
        xyz[p]=X;support[p]=keep.sum();inlier[ids[keep],p]=True;errors[ids,p]=e
    return xyz.reshape(*shape,3),support.reshape(shape),inlier.reshape(8,*shape),errors.reshape(8,*shape)


def run(candidate):
    W=np.load(c.OUT/f'{candidate}.npz')['W'];report={'candidate':candidate,'gate_px':5,'minimum_views':3,'datasets':{}}
    for kind in ['static','dynamic']:
        man,cams,K,old,xy,valid=c.load(kind)
        kp=np.load(c.OUT/f'keypoints_crop_{kind}.npz')
        xy=kp['xy'][:,:,c.JOINTS].astype(float);valid=(kp['score'][:,:,c.JOINTS]>=.6)&kp['in_image'][:,:,c.JOINTS]
        X,sup,inl,err=solve(K,W,xy,valid)
        good=sup>=3
        r={'joint_frame_coverage':float(good.mean()),'frames_all_12_joints':int(good.all(1).sum()),
           'median_support':float(np.median(sup[good])) if good.any() else None,
           'inlier_residual':c.stats(err[inl]),'all_detected_residual_where_3d_valid':c.stats(err[valid]),
           'per_joint_coverage':good.mean(0).tolist(),'per_camera':{}}
        for ci,cam in enumerate(cams):
            r['per_camera'][cam]={'candidate_observations':int(valid[ci].sum()),'inlier_observations':int(inl[ci].sum()),
                'inlier_fraction_of_candidates':float(inl[ci].sum()/max(valid[ci].sum(),1)),
                'residual':c.stats(err[ci][inl[ci]])}
        np.savez_compressed(c.OUT/f'{candidate}_observations_{kind}.npz',xyz=X,support=sup,inlier=inl,
                            error_px=err,frames=kp['frames'],cams=kp['cams'],joints=c.JOINTS)
        report['datasets'][kind]=r
        print(kind,'coverage',r['joint_frame_coverage'],'support',r['median_support'],'all12frames',r['frames_all_12_joints'],flush=True)
        print('per camera inlier coverage',[round(v['inlier_fraction_of_candidates'],3) for v in r['per_camera'].values()],flush=True)
    c.write_json(f'{candidate}_screening.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',default='candidate_near');run(p.parse_args().candidate)
