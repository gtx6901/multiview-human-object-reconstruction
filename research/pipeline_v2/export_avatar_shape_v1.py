"""Export actual Gaussian rig and compare tensor replay against upstream LHM animation."""
from pathlib import Path
import sys,runpy,json,time,argparse
ROOT=__import__('hoi_paths').ROOT;LHM=ROOT.parent/'lhmpp_20261016'
parser=argparse.ArgumentParser();parser.add_argument('--out',default='avatar_motion180_211_v1');parser.add_argument('--poses',default='body_fused180_211_v1');parser.add_argument('--appearance-run');parser.add_argument('--shape-candidate',help='Conditional shape folder: beta/scale/translation propagated through actual upstream LHM shape blend and rest rig');options=parser.parse_args()
sys.path.insert(0,str(LHM));sys.argv=[str(LHM/'load_model_probe.py'),'--name','../../pipeline/model_load_'+options.out]
scope=runpy.run_path(str(LHM/'load_model_probe.py'));model=scope['model']
import torch
from core.outputs.output import GaussianAppOutput
from avatar_rig import animate
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
def move(x):
 if isinstance(x,torch.Tensor):return x.cuda()
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [move(v) for v in x]
 return x
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 return x
seed=ROOT.parent/'lhmpp_replay_seeded_a/runs'
data_path=LHM/'runs'/options.appearance_run/'gaussian_attributes.pt' if options.appearance_run else seed/'decode_all8_v1/gaussian_attributes.pt'
latent_path=LHM/'runs'/options.appearance_run/'latent.pt' if options.appearance_run else seed/'latent_all8_v2/latent.pt'
data=move(torch.load(data_path,weights_only=True));latent=torch.load(latent_path,weights_only=True)['result']
poses=move(torch.load(ROOT/options.poses/'pose_sequence.pt',weights_only=True))
if options.shape_candidate:
    shape=move(torch.load(ROOT/options.shape_candidate/'shape_candidate.pt',weights_only=True))
    assert shape['translation_world'].shape==poses['translation_world'].shape
    for key in ['betas','scale','translation_world']:poses[key]=shape[key]
assert torch.isfinite(poses['betas']).all()
model.renderer.cuda();model.renderer.device=torch.device('cuda');skinner=model.renderer.smplx_model
with torch.inference_mode():
    query=data['query']['neutral_coords'][0]+data['attrs'][0]['offset_xyz'];N=len(query)
    weights=skinner.query_voxel_skinning_weights(query[None]).reshape(N,-1)
    fixed=((skinner.is_rhand+skinner.is_lhand+skinner.is_face)>0)
    weights[fixed]=skinner.skinning_weight[fixed]
    neutral_to_zero=(weights@data['smplx']['transform_mat_neutral_pose'][0].reshape(-1,16)).reshape(N,4,4)
    zero=skinner.lbs(query[None],neutral_to_zero[None],torch.zeros(1,3,device='cuda'))[0]
    # Upstream queries and voxel weights are deliberately beta-independent.
    # LHM applies its dense shape directions AFTER mapping neutral to zero pose.
    if poses['betas'].abs().max()>0:
        from smplx.lbs import blend_shapes
        zero=zero+blend_shapes(poses['betas'],skinner.shape_dirs)[0]
    rest=skinner.get_zero_pose_human(shape_param=poses['betas'],device=torch.device('cuda'),face_offset=None,joint_offset=None)
    rest_error=0.0  # landmark regressor removed; shape blend is still applied upstream
    rig={'weights':weights.float(),'zero_coords':zero.float(),'neutral_to_zero':neutral_to_zero.float(),'rest_joints':rest.float(),'parents':skinner.smplx_layer.parents.cuda(),'neutral_rotation':data['attrs'][0]['rotation'].float(),**{k:data['attrs'][0][k].float() for k in ['opacity','scaling','shs']}}
    checks=[]
    for t in sorted(set([0,len(poses['frames'])//2,len(poses['frames'])-1])):
        params={k:torch.zeros(1,3,device='cuda') for k in ['jaw_pose','leye_pose','reye_pose','trans']}
        params.update(root_pose=poses['root_pose'][t:t+1],body_pose=poses['body_pose'][t:t+1],betas=poses['betas'],lhand_pose=poses.get('left_hand_pose',torch.zeros(len(poses['frames']),15,3,device='cuda'))[t:t+1],rhand_pose=poses.get('right_hand_pose',torch.zeros(len(poses['frames']),15,3,device='cuda'))[t:t+1],expr=torch.zeros(1,100,device='cuda'),transform_mat_neutral_pose=data['smplx']['transform_mat_neutral_pose'][0])
        original,_=model.renderer.animate_gs_model(GaussianAppOutput(**data['attrs'][0]),data['query']['neutral_coords'][0],params,mesh_meta=data['query']['mesh_meta']);g=original[0];replay=animate(rig,poses,t)
        error=float((g.xyz*poses['scale']+poses['translation_world'][t]-replay['xyz']).abs().max());rotation_error=float(1-torch.nn.functional.cosine_similarity(g.rotation.float(),replay['rotation'],dim=-1).abs().min())
        assert error<2e-5 and rotation_error<2e-5,(error,rotation_error)
        checks.append({'frame':int(poses['frames'][t]),'max_xyz_world_error':error,'max_quaternion_sign_invariant_cosine_error':rotation_error})
    bundle={'rig':cpu(rig),'poses':cpu(poses),'features':latent[0][0],'background_embedding':model.renderer.background_embedding.detach().cpu(),'decoder':{k:v.detach().cpu() for k,v in model.neural_renderer.state_dict().items()},'decoder_config':json.loads((LHM/'checkpoint_config.json').read_text())['neural_renderer'],'scope':'Fixed avatar shape/expression/jaw/eyes/hand pose only. Animated all8 fitted body/root/translation, original neural appearance. Tensor rig validated against upstream animation; not conventional textured mesh or CoCoNet output.'}
    if options.appearance_run:bundle['scope']='Independent subset-view neural appearance and supplied subset-fitted full55-joint motion. Fixed pretrained decoder. Not held-out rendering quality evidence.'
    torch.save(bundle,out/'avatar.pt')
report={'success':True,'gaussians':N,'frames':poses['frames'].tolist(),'checks_against_upstream':checks,'rest_joint_max_error':rest_error,'bytes':(out/'avatar.pt').stat().st_size,'seconds':time.time()-start,'scope':'Export/replay animation contract verified, not reconstructed motion or novel-view quality validation. Original complete LHM model used during export only; bundle replay requires tensor skinning and saved neural decoder, not reconstruction encoder.'}
report['sources']={'poses':options.poses,'appearance_run':options.appearance_run,'gaussian_attributes':str(data_path),'latent':str(latent_path)}
report['sources']['shape_candidate']=options.shape_candidate
report['shared_beta_norm']=float(poses['betas'].norm())
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
