"""Compare subset forward and gradients against full official SMPL-X LBS."""
import os
from pathlib import Path
import sys,json,inspect,hashlib,argparse
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import lbs
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
p=argparse.ArgumentParser();p.add_argument('--out',default='surface_landmark_verification_v1');o=p.parse_args()
torch.manual_seed(41017);out=ROOT/o.out;out.mkdir(exist_ok=False)
model=SMPLXSurfaceLandmarks().cuda();d=np.load(model.model_path,allow_pickle=True)
def tensor(key):return torch.tensor(d[key],dtype=torch.float32,device='cuda')
aa=(torch.rand(3,55,3,device='cuda')-.5)*.5;aa.requires_grad_()
args={'betas':torch.zeros(3,10,device='cuda'),'pose':aa.reshape(3,-1),'v_template':tensor('v_template'),'shapedirs':tensor('shapedirs')[:,:,:10],'posedirs':tensor('posedirs').reshape(10475*3,486).T.contiguous(),'J_regressor':tensor('J_regressor'),'parents':model.parents,'lbs_weights':tensor('weights')}
for key in ['joint_offset','locator_offset']:
    if key in inspect.signature(lbs).parameters:args[key]=None
full,joints=lbs(**args);subset=model(aa)
err=float((full[:,model.vertex_ids]-subset['vertices_subset']).abs().max())
jerr=float((joints-subset['joints55']).abs().max());assert err<2e-6 and jerr<2e-6
mapping=torch.load(ROOT/'smplx_coco_regressor_v1/smplx2smpl_sparse.pt',weights_only=True).cuda().to_dense()
reg=torch.load(ROOT/'smplx_coco_regressor_v1/smpl_coco17_J_regressor.pt',weights_only=True).cuda()@mapping
coco=reg@full;cerror=float((coco-subset['coco17']).abs().max());assert cerror<2e-6
g1=torch.autograd.grad(coco.square().sum(),aa,retain_graph=True)[0]
g2=torch.autograd.grad(subset['coco17'].square().sum(),aa,retain_graph=True)[0]
gerror=float((g1-g2).abs().max());assert gerror<2e-5 and torch.isfinite(g2).all()
# Verify named hand indexing, not only subset vertices.
ids=[20,37,38,39,5361,25,26,27,4933,28,29,30,5058,34,35,36,5169,31,32,33,5286]
expected=torch.stack([joints[:,i] if k in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19] else full[:,i] for k,i in enumerate(ids)],1)
handerr=float((expected-subset['hands21'][:,0]).abs().max());assert handerr<2e-6
report={'success':True,'samples':3,'subset_vertices':len(model.vertex_ids),'model_sha256':hashlib.sha256(model.model_path.read_bytes()).hexdigest(),'max_vertex_error':err,'max_joint_error':jerr,'max_coco17_error':cerror,'max_axis_angle_gradient_error':gerror,'max_left_hand_order_error':handerr,'shape':'fixed neutral beta0, matching frozen avatar skeleton','scope':'Forward and autograd equivalence to full official LBS on random poses. Validates implementation and topology mapping, not detector accuracy, calibration, motion quality or fitted hands.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
