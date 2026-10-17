"""Input-only multiview checks of actual raw model predictions, never GT scores."""
import os
import json,time
from cari_runtime import configure,ROOT,ASSETS,make_smplh
configure()
import torch,numpy as np
from scipy.spatial.transform import Rotation
from lib_smpl import pose72to156
from lib_smpl.body_landmark import BodyLandmarks
from tools import geometry_utils as geom_utils
out=ROOT/'coconet_all8_audit180_211_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
manifest=json.loads((ROOT/'coconet_all8_180_211_v1/report.json').read_text());meta=json.loads((ROOT/'helmet_contiguous180_211_v1/manifest.json').read_text());det=np.load(ROOT/'detections_contiguous180_211_v2/detections.npz')
alpha=json.loads((ROOT/'nlf_depth_world_alignment180_211_v1/report.json').read_text())['common_model_to_world_scale'];body=make_smplh().eval();landmark=BodyLandmarks(str(ASSETS));ids=[5,2,6,3,7,4,12,9,13,10,14,11]
world_in=[];world_out=[];contacts=[];metrics=[];object_centers=[];object_rotations=[];perview=[]
with torch.inference_mode():
 for vi,v in enumerate(meta['views']):
    serial=v['serial'];K=np.array(v['K']);W=np.array(v['W']);initial=[];predicted=[];view_contacts=[];oc=[];orr=[]
    for offset in [0,16]:
        window=ROOT/'coconet_all8_180_211_v1'/f'{serial}_{offset:02d}';raw=torch.load(window/'raw_predictions.pt',weights_only=True);batch=torch.load(window/'input_batch.pt',weights_only=True)
        frames=batch['frames'];nlf=[torch.load(ROOT/'nlf_depthaligned180_211_v5'/f'{serial}_{f:04d}.pt',weights_only=True) for f in frames]
        rotations=torch.tensor(Rotation.from_rotvec(np.concatenate([p['pose_rotvecs'].numpy() for p in nlf]).reshape(-1,3)).as_matrix(),dtype=torch.float32).reshape(16,52,3,3)[:,:24]
        delta=geom_utils.rot6d_to_rotmat(raw['hum_pose'].reshape(-1,6)).reshape(16,24,3,3);pr=delta@rotations
        pose=geom_utils.rotation_matrix_to_angle_axis(pr.reshape(-1,3,3)).reshape(16,72)
        translations=torch.cat([p['trans'] for p in nlf])+raw['hum_trans'];mean_betas=torch.cat([p['shape_betas'] for p in nlf]).mean(0,keepdim=True);betas=mean_betas+raw['hum_shape']
        vertices=body(pose72to156(pose),betas,translations)[0].numpy();initial.extend([p['vertices'][0].numpy() for p in nlf]);predicted.extend(vertices);view_contacts.extend(raw['contact'].sigmoid().numpy())
        object_t=batch['batch']['pose_perturbed'][0,:,:3,3]+raw['trans']*batch['batch']['mesh_diameter'][0,:,None]/2
        object_r=geom_utils.rot6d_to_rotmat(raw['rot_abs']).numpy()
        oc.extend((object_t.numpy()*alpha-W[:3,3])@W[:3,:3]);orr.extend(np.einsum('ij,tjk->tik',W[:3,:3].T,object_r))
    ji=landmark.get_body_kpts_batch(np.array(initial))[:,ids];jo=landmark.get_body_kpts_batch(np.array(predicted))[:,ids]
    wi=(ji*alpha-W[:3,3])@W[:3,:3];wo=(jo*alpha-W[:3,3])@W[:3,:3];world_in.append(wi);world_out.append(wo);contacts.append(view_contacts);object_centers.append(oc);object_rotations.append(orr)
    observed=det['xy'][vi,:,5:];valid=det['score'][vi,:,5:]>3
    errors=[]
    for joints in [ji,jo]:
        pixels=joints@K.T;pixels=pixels[...,:2]/pixels[...,2:];errors.append(float(np.linalg.norm(pixels-observed,axis=-1)[valid].mean()))
    metrics.append({'serial':serial,'initial_body_mean_reprojection_px':errors[0],'predicted_body_mean_reprojection_px':errors[1],'contact_probability_mean':np.array(view_contacts).mean(0).tolist()})
    perview.append({'serial':serial,'world_body_joints_initial':torch.tensor(wi),'world_body_joints_predicted':torch.tensor(wo),'contact_probability':torch.tensor(np.array(view_contacts)),'object_center_world':torch.tensor(np.array(oc)),'object_rotation_world':torch.tensor(np.array(orr))})
def dispersion(points):
    x=np.array(points);return float(np.linalg.norm(x-x.mean(0,keepdims=True),axis=-1).mean())
torch.save({'views':perview,'frames':torch.tensor(meta['sample_frames'])},out/'predictions_world.pt')
report={'success':True,'per_view':metrics,'initial_body_mean_cross_view_dispersion_world':dispersion(world_in),'predicted_body_mean_cross_view_dispersion_world':dispersion(world_out),'predicted_object_center_mean_cross_view_dispersion_world':dispersion(object_centers),'contact_probability_std_across_views_mean':np.array(contacts).std(0).mean(0).tolist(),'seconds':time.time()-start,'scope':'Actual raw model predictions decoded by upstream rotation/SMPL conversion rules and original SMPL-H. Diagnostics use input keypoints, fixed calibrated cameras and one common scale. No GT, no independent novel-view or contact validation. No averaging into accepted shared geometry. Object rotation uses absolute head and translation uses relative head according to checkpoint loss_type. Hand contact sigmoid values are model probabilities, not physical contact evidence.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
