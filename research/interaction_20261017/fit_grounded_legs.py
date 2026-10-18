"""One lower-body block: six observed feet, temporal rotations and contact."""
import os
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import axis_angle_to_matrix
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--avatar',default='body_head_faces_all300_v1');p.add_argument('--constraints',default='feet_constraints_all300_v1');p.add_argument('--steps',type=int,default=600);o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
assert json.loads((ROOT/'foot_surface_verification_v1/report.json').read_text())['success'];b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);base=b['poses'];aa0=torch.cat([pose_axis_angles(base,t,'cpu') for t in range(300)]).cuda();scale=base['scale'].cuda();trans=base['translation_world'].cuda();beta=base['betas'].cuda();model=SMPLXSurfaceLandmarks().cuda();legids=[1,2,4,5,7,8,10,11]
# Symmetric matrix filtering and SO3 projection provide a zero-phase start.
R0=axis_angle_to_matrix(aa0[:,legids]).cpu().numpy();filtered=savgol_filter(R0,11,2,axis=0);u,_,vh=np.linalg.svd(filtered);z=np.tile(np.eye(3),(300,len(legids),1,1));z[...,-1,-1]=np.linalg.det(u@vh);smoothR=u@z@vh;initial=torch.tensor(Rotation.from_matrix(smoothR.reshape(-1,3,3)).as_rotvec().reshape(300,len(legids),3),device='cuda',dtype=torch.float32);legs=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([legs],lr=.003)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());K=torch.tensor(np.array([v['K'] for v in meta['views']]),device='cuda',dtype=torch.float32);W=torch.tensor(np.array([v['W'] for v in meta['views']]),device='cuda',dtype=torch.float32)
d=np.load(ROOT/o.constraints/'constraints.npz');xy=torch.tensor(d['xy'],device='cuda',dtype=torch.float32);valid=torch.tensor(d['inlier'],device='cuda');fw=torch.tensor(np.minimum(d['raw_score']/5,1)*d['inlier'],device='cuda',dtype=torch.float32);contact=torch.tensor(np.repeat(d['contact'],3,axis=1),device='cuda');anchors=torch.tensor(d['contact_anchors'],device='cuda',dtype=torch.float32)
body=np.load(ROOT/'observations_all300_v1/body.npz');bxy=torch.tensor(body['xy'][:,:,11:17]*2+.5,device='cuda',dtype=torch.float32);bv=torch.tensor(body['score'][:,:,11:17]>3,device='cuda')
anchorrows=anchors[contact].cpu().numpy();assert len(anchorrows)>30;center=anchorrows.mean(0);_,_,vv=np.linalg.svd(anchorrows-center);normal=vv[-1];up=axis_angle_to_matrix(aa0[:,0]).cpu().numpy()[:,:,1].mean(0);up/=np.linalg.norm(up)
if normal@up<0:normal=-normal
if normal@up<np.cos(np.deg2rad(20)):normal=up.copy()
normal=torch.tensor(normal,device='cuda',dtype=torch.float32);center=torch.tensor(center,device='cuda',dtype=torch.float32)
def pose():
 aa=aa0.clone();aa[:,legids]=legs;return aa
def project(points):
 pc=torch.einsum('vij,tkj->vtki',W[:,:3,:3],points)+W[:,None,None,:3,3];q=torch.einsum('vij,vtkj->vtki',K,pc);return q[...,:2]/q[...,2:].clamp_min(.01)
def evaluate(aa):
 r=model(aa,scale,trans,betas=beta);feet=r['feet6'];fe=(project(feet)-xy).norm(dim=-1);be=(project(r['coco17'][:,11:17])-bxy).norm(dim=-1);robust=lambda x:2*(torch.sqrt(1+x*x)-1)
 data=(robust(fe/8)*fw).sum()/fw.sum()+.5*(robust(be/8)*bv).sum()/bv.sum()
 points=torch.cat([feet,r['joints55'][:,[1,2,4,5,7,8]]],1)/scale;acc=(points[2:]-2*points[1:-1]+points[:-2])*900;rot=axis_angle_to_matrix(aa[:,legids]);racc=(rot[2:]-2*rot[1:-1]+rot[:-2])*900
 temporal=.5*robust(acc/1.).mean()+.25*robust(racc/8.).mean();position=robust((feet-anchors)/scale/.01)[contact].mean();velocity=(feet[1:]-feet[:-1])/scale*30;both=contact[1:]&contact[:-1];stance=robust(velocity/.03)[both].mean();floor=((feet-center)@normal)/scale;floorloss=robust(floor[contact]/.02).mean();prior=.02*((aa[:,legids]-aa0[:,legids])/.3).square().mean()
 loss=data+temporal+position+.5*stance+.1*floorloss+prior
 return loss,r,{'data':data,'temporal':temporal,'contact_position':position,'contact_velocity':stance,'floor':floorloss,'foot_native_px':fe[valid].mean(),'body_native_px':be[bv].mean()}
def metrics(aa):
 _,r,terms=evaluate(aa);feet=r['feet6'];d2=(feet[47:]-2*feet[46:-1]+feet[45:-2])/scale;uv=project(feet);imagecenter=uv.reshape(8,300,2,3,2).mean(3);jitter=(imagecenter[:,47:]-2*imagecenter[:,46:-1]+imagecenter[:,45:-2]).norm(dim=-1);v=(feet[1:]-feet[:-1])/scale*30;keep=contact[1:]&contact[:-1]
 return {**{k:float(x) for k,x in terms.items()},'foot_d2_model_units_median':float(d2.norm(dim=-1).median()),'foot_d2_model_units_p95':float(torch.quantile(d2.norm(dim=-1),.95)),'projected_foot_d2_native_px_median':float(jitter.median()),'contact_speed_model_units_per_sec_median':float(v.norm(dim=-1)[keep].median()),'contact_speed_model_units_per_sec_p95':float(torch.quantile(v.norm(dim=-1)[keep],.95))}
history=[];best=None
with torch.no_grad():before=metrics(aa0)
try:
 for step in range(o.steps+1):
  opt.zero_grad();aa=pose();loss,r,terms=evaluate(aa);value=float(loss.detach())
  if best is None or value<best[0]:best=(value,legs.detach().clone())
  if step%100==0 or step==o.steps:
   row={'step':step,'loss':value,**{k:float(x.detach()) for k,x in terms.items()}};history.append(row);torch.save({'legs':legs.detach().cpu(),'best_legs':best[1].cpu(),'optimizer':opt.state_dict(),'step':step,'source_avatar':o.avatar},out/'checkpoint.pt');print(json.dumps(row),flush=True)
  if step<o.steps:loss.backward();opt.step()
except KeyboardInterrupt:
 torch.save({'legs':legs.detach().cpu(),'best_legs':best[1].cpu(),'optimizer':opt.state_dict(),'history':history,'source_avatar':o.avatar},out/'interrupted_checkpoint.pt');raise
with torch.no_grad():
 legs.copy_(best[1]);aa=pose();_,r,_=evaluate(aa);after=metrics(aa);poses={**base,'body_pose':aa[:,1:22].cpu(),'joints55_world':r['joints55'].cpu(),'joints_world':r['coco17'][:,5:17].cpu(),'projected':((project(r['coco17'][:,5:17])-.5)/2).cpu()};unchanged=[j for j in range(1,22) if j not in legids];assert torch.equal(aa[:,unchanged],aa0[:,unchanged]);assert torch.equal(aa[:,0],aa0[:,0]);torch.save(poses,out/'pose_sequence.pt');torch.save({**b,'poses':poses},out/'avatar.pt')
report={'success':True,'source_avatar':o.avatar,'constraints':o.constraints,'steps':o.steps,'lower_joint_ids':legids,'upper_root_translation_canonical_frozen':True,'before':before,'after':after,'history':history,'seconds':time.time()-start,'scope':'Single lower-body block on original accepted canonical human; eight views/all300 source observations. Foot6 data, hip/knee/ankle body data, model-normalized positional acceleration and SO3 matrix rotation acceleration at30fps, fixed observed contact anchors/low stance speed and fitted approximate contact plane. Root/upper body/fingers/beta/scale/appearance exactly unchanged. Foot-plane is an anatomical contact approximation, not measured shoe sole. Actual saved LHM render and integrated helmet sequence comparison required before adoption.'};(out/'report.json').write_text(json.dumps(report,indent=2));print('grounded legs candidate ready for render',flush=True)
