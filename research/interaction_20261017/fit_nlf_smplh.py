"""Official CARI4D NLF -> SMPL-H fitting step, using actual cached predictions."""
import os
import argparse,json,time
from cari_runtime import configure,make_smplh,SMPLH,ROOT
configure()
import torch,numpy as np
from smplfitter.pt import BodyModel,BodyFitter
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('out');options=p.parse_args()
torch.set_num_threads(4)
source=ROOT/options.source;out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time()
# smplfitter 0.6 checks the shared kid template even with enable_kid=False.
# Use the verified official demo asset; never synthesize a placeholder template.
kid_source=ROOT/'cari_assets/official_demo/smpl/kid_template.npy';kid_target=SMPLH/'kid_template.npy'
if kid_target.exists():assert kid_target.read_bytes()==kid_source.read_bytes()
else:kid_target.write_bytes(kid_source.read_bytes())
model=BodyModel('smplh','male',model_root=str(SMPLH),num_betas=10);fitter=BodyFitter(model).eval();official=make_smplh().eval()
records=[]
with torch.inference_mode():
 for path in sorted(source.glob('*.pt')):
    pred=torch.load(path,weights_only=True);vertices=pred['vertices3d'][0]/1000
    result=fitter.fit(vertices,num_iter=3,beta_regularizer=1,requested_keys=['shape_betas','trans','vertices','pose_rotvecs'])
    assert all(torch.isfinite(x).all() for x in result.values() if isinstance(x,torch.Tensor))
    assert result['pose_rotvecs'].shape==(1,156) and result['shape_betas'].shape==(1,10)
    # Current smplfitter returns parameters rather than vertices from fit().
    # Execute its actual BodyModel forward with those parameters for comparison.
    result['vertices']=model(pose_rotvecs=result['pose_rotvecs'],shape_betas=result['shape_betas'],trans=result['trans'])['vertices']
    v=official(result['pose_rotvecs'],result['shape_betas'],result['trans'])[0]
    mismatch=torch.linalg.vector_norm(v-result['vertices'],dim=-1)
    target_error=torch.linalg.vector_norm(v-vertices,dim=-1)
    torch.save({k:x.cpu() for k,x in result.items()},out/path.name)
    record={'input':path.name,'official_vs_fitter_mean_m':float(mismatch.mean()),'official_vs_fitter_max_m':float(mismatch.max()),'target_vertex_mean_m':float(target_error.mean()),'pose_shape':list(result['pose_rotvecs'].shape),'betas_shape':list(result['shape_betas'].shape)};records.append(record);print(json.dumps(record),flush=True)
assert records
(out/'report.json').write_text(json.dumps({'success':True,'source':options.source,'records':records,'seconds':time.time()-start,'scope':'Actual NLF cached predictions converted with the smplfitter BodyFitter algorithm called by official CARI4D, 3 iterations and beta_regularizer=1. CPU execution, 10 shape coefficients explicit. Cross-checked against official CARI4D SMPLLayer. NLF-implied metric scale is not independent metric ground truth; no CoCoNet or multiview fusion yet.'},indent=2))
