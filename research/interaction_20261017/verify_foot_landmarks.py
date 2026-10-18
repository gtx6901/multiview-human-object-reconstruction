"""Validate foot order, shaped forward and gradients against full SMPL-X LBS."""
import os
from pathlib import Path
import sys,json,inspect
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import lbs
from smplx.vertex_ids import vertex_ids
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
torch.manual_seed(445);out=ROOT/'foot_surface_verification_v1';out.mkdir(exist_ok=False);model=SMPLXSurfaceLandmarks().cuda();d=np.load(model.model_path,allow_pickle=True)
def tensor(k):return torch.tensor(d[k],dtype=torch.float32,device='cuda')
aa=((torch.rand(3,55,3,device='cuda')-.5)*.5).requires_grad_();beta=torch.randn(3,10,device='cuda')*.4
args={'betas':beta,'pose':aa.reshape(3,-1),'v_template':tensor('v_template'),'shapedirs':tensor('shapedirs')[:,:,:10],'posedirs':tensor('posedirs').reshape(10475*3,486).T.contiguous(),'J_regressor':tensor('J_regressor'),'parents':model.parents,'lbs_weights':tensor('weights')}
for k in ['joint_offset','locator_offset']:
 if k in inspect.signature(lbs).parameters:args[k]=None
full,joints=lbs(**args);ids=[vertex_ids['smplx'][x] for x in ['LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']];r=model(aa,betas=beta);error=float((full[:,ids]-r['feet6']).abs().max());assert error<2e-6
g1=torch.autograd.grad(full[:,ids].square().sum(),aa,retain_graph=True)[0];g2=torch.autograd.grad(r['feet6'].square().sum(),aa)[0];graderror=float((g1-g2).abs().max());assert graderror<2e-5
(out/'report.json').write_text(json.dumps({'success':True,'random_nonzero_beta_samples':3,'foot_order':['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel'],'max_forward_error':error,'max_gradient_error':graderror,'scope':'Actual full official LBS comparison for six-foot vertex subset and gradients, with nonzero shape. Validates implementation/indexing, not detection or reconstruction accuracy.'},indent=2));print('foot forward and gradients verified',flush=True)
