"""Read-only exhaustive camera-label permutation diagnostic, never recalibrate."""
import os
from pathlib import Path
import sys,json,itertools,time,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
out=ROOT/'backpack_camera_assignment_audit_v1';out.mkdir(exist_ok=False);start=time.time()
manifest=ROOT/'backpack_native000_098_v2/manifest.json';original=hashlib.sha256(manifest.read_bytes()).hexdigest();m=json.loads(manifest.read_text());d=np.load(ROOT/'backpack_wholebody_all99_v2/detections.npz');xy=d['xy'][:,:,5:17];valid=d['in_image'][:,:,5:17]&(d['score'][:,:,5:17]>2)
K=np.array([v['K'] for v in m['views']]);W=np.array([v['W'] for v in m['views']]);invW=np.linalg.inv(W);N=8;T=xy.shape[1];a=np.concatenate([xy,np.ones_like(xy[...,:1])],-1);permutations=np.array(list(itertools.permutations(range(8))),dtype=np.int16);pairs=list(itertools.combinations(range(8),2));cost=np.full((2,2,8,8,8,8),np.nan)
splits=[np.arange(T)%2==0,np.arange(T)%2==1];rows=[]
for pi,(u,v) in enumerate(pairs):
 for i in range(N):
  for j in range(N):
   if i==j:continue
   Q=W[j]@invW[i];tx,ty,tz=Q[:3,3];S=np.array([[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]])
   for mode in range(2):
    Ku,Kv=(K[u],K[v]) if mode==0 else (K[i],K[j]);F=np.linalg.inv(Kv).T@S@Q[:3,:3]@np.linalg.inv(Ku);l1=a[u]@F.T;l2=a[v]@F
    e=abs((a[v]*l1).sum(-1))*.5*(1/np.maximum(np.linalg.norm(l1[...,:2],axis=-1),1e-10)+1/np.maximum(np.linalg.norm(l2[...,:2],axis=-1),1e-10));good=valid[u]&valid[v]&np.isfinite(e)
    for split,frames in enumerate(splits):
     values=e[good&frames[:,None]];value=float(np.median(values)) if len(values)>=30 else np.inf;cost[mode,split,u,v,i,j]=value
 print('camera pair evaluated',u,v,flush=True)
for mode,label in enumerate(['extrinsics_assignment_only','whole_camera_K_and_W_assignment']):
 train=np.stack([cost[mode,0,u,v,permutations[:,u],permutations[:,v]] for u,v in pairs],1);test=np.stack([cost[mode,1,u,v,permutations[:,u],permutations[:,v]] for u,v in pairs],1)
 # Trim the worst20percent pairs so a few heavily occluded/detection pairs do
 # not decide the mapping. Fixed before searching; untrimmed values retained.
 objective=np.sort(train,axis=1)[:,:23].mean(1);order=np.argsort(objective);identity=int(np.where((permutations==np.arange(8)).all(1))[0]);top=[]
 for k in [identity,*order[:5].tolist()]:
  mapping=permutations[k];top.append({'is_identity':k==identity,'image_serial_to_candidate_calibration_serial':{m['views'][u]['serial']:m['views'][int(mapping[u])]['serial'] for u in range(8)},'train_trimmed_pair_mean_native_px':float(objective[k]),'test_trimmed_pair_mean_native_px':float(np.sort(test[k])[:23].mean()),'train_pair_median_native_px':float(np.median(train[k])),'test_pair_median_native_px':float(np.median(test[k])),'pairs':[{'image_cameras':[m['views'][u]['serial'],m['views'][v]['serial']],'train_native_px':float(train[k,pi]),'test_native_px':float(test[k,pi])} for pi,(u,v) in enumerate(pairs)]})
 rows.append({'hypothesis':label,'permutations_evaluated':len(permutations),'identity_and_top5':top})
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==original
report={'success':True,'hypotheses':rows,'seconds':time.time()-start,'source_manifest_sha256':original,'sources_unchanged':True,'scope':'Exhaustive8! assignment search using same-frame12 torso/limb detector correspondences, rawscore>2 and in_image. Even frames rank assignments, odd frames evaluate the same fixed ranking; temporal correlation remains. Pairwise symmetric epipolar medians then trimmed23-of28 mean, thresholds fixed before search. Tests extrinsics-only and whole K/W serial misassignment hypotheses. Detector semantics, synchronization, original calibration quality and degenerate camera geometry remain possible confounds. No calibration/serial mapping file changed, no candidate accepted automatically; requires multiview3D and static-background confirmation.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps([{'hypothesis':r['hypothesis'],'identity':r['identity_and_top5'][0]['test_trimmed_pair_mean_native_px'],'best':r['identity_and_top5'][1]['test_trimmed_pair_mean_native_px'],'mapping':r['identity_and_top5'][1]['image_serial_to_candidate_calibration_serial']} for r in rows]),flush=True)
