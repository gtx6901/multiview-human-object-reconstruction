"""Compare nonzero-shape subset/gradients to official full SMPL-X LBS."""
import os
from pathlib import Path
import sys,json,inspect
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import lbs
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
out=ROOT/'shaped_surface_verification_v1';out.mkdir(exist_ok=False);torch.manual_seed(20261018);torch.set_num_threads(4)
model=SMPLXSurfaceLandmarks();d=np.load(model.model_path,allow_pickle=True);tensor=lambda key:torch.tensor(d[key],dtype=torch.float32)
args={'v_template':tensor('v_template'),'shapedirs':tensor('shapedirs')[:,:,:10],'posedirs':tensor('posedirs').reshape(10475*3,486).T.contiguous(),'J_regressor':tensor('J_regressor'),'parents':model.parents,'lbs_weights':tensor('weights')}
for key in ['joint_offset','locator_offset']:
 if key in inspect.signature(lbs).parameters:args[key]=None
aa=torch.randn(3,55,3)*.2;beta=(torch.randn(3,10)*.7).requires_grad_();full,joints=lbs(betas=beta,pose=aa.reshape(3,-1),**args);subset=model(aa,betas=beta)
error=float((full[:,model.vertex_ids]-subset['vertices_subset']).abs().max());jerror=float((joints[:,:55]-subset['joints55']).abs().max());assert error<2e-6 and jerror<2e-6
g1=torch.autograd.grad(full[:,model.vertex_ids].square().sum(),beta,retain_graph=True)[0];g2=torch.autograd.grad(subset['vertices_subset'].square().sum(),beta)[0];gerror=float((g1-g2).abs().max());assert gerror<2e-5
plain=model(aa);zero=model(aa,betas=torch.zeros(1,10));zero_error=float((plain['vertices_subset']-zero['vertices_subset']).abs().max());assert zero_error==0
(out/'report.json').write_text(json.dumps({'success':True,'nonzero_shapes_tested':3,'max_subset_vertex_error':error,'max_joint_error':jerror,'max_beta_gradient_error':gerror,'zero_shape_path_vertex_difference':zero_error,'scope':'Numerical agreement with full official SMPL-X LBS at3random shapes/poses, gradients and exact zero-shape path. No evidence of correct person-specific shape estimation; accepted avatar unchanged.'},indent=2));print('shaped surface verified',error,jerror,gerror,flush=True)
