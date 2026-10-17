"""Actual posed SMPL-X mesh depths for conservative helmet occlusion constraints."""
import os
from pathlib import Path
import sys,json,time,inspect
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from smplx.lbs import lbs
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from mesh_renderer import MeshRenderer
out=ROOT/'body_depth_all300_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
poses=torch.load(ROOT/'body_hands_all300_v1/pose_sequence.pt',weights_only=True);T=len(poses['frames']);assert T==300
model=SMPLXSurfaceLandmarks().cuda();d=np.load(model.model_path,allow_pickle=True)
def tensor(key):return torch.tensor(d[key],dtype=torch.float32,device='cuda')
args={'v_template':tensor('v_template'),'shapedirs':tensor('shapedirs')[:,:,:10],'posedirs':tensor('posedirs').reshape(10475*3,486).T.contiguous(),'J_regressor':tensor('J_regressor'),'parents':model.parents,'lbs_weights':tensor('weights')}
for key in ['joint_offset','locator_offset']:
    if key in inspect.signature(lbs).parameters:args[key]=None
verts=[];hands=[];jointlist=[]
with torch.inference_mode():
    for first in range(0,T,16):
        stop=min(first+16,T);aa=torch.cat([pose_axis_angles(poses,t,'cpu') for t in range(first,stop)]).cuda()
        v,j=lbs(betas=poses['betas'].cuda().expand(stop-first,-1),pose=aa.reshape(stop-first,-1),**args)
        subset=model(aa);assert torch.allclose(v[:,model.vertex_ids],subset['vertices_subset'],atol=2e-6)
        v=v*poses['scale'].cuda()+poses['translation_world'][first:stop].cuda()[:,None]
        h=subset['hands21']*poses['scale'].cuda()+poses['translation_world'][first:stop].cuda()[:,None,None]
        verts.append(v.cpu());hands.append(h.cpu());jointlist.append(j.cpu()*poses['scale']+poses['translation_world'][first:stop,None])
vertices=torch.cat(verts);hands=torch.cat(hands).numpy();joints=torch.cat(jointlist)
faces=torch.tensor(d['f'].astype('int32'),device='cuda');renderer=MeshRenderer();meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());width,height=256,192
affine=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
viewreports=[]
with torch.inference_mode():
    for v in meta['views']:
        K=torch.tensor(affine@np.array(v['K']),device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
        depths=[];alphas=[];unknowns=[]
        for t in range(T):
            mesh=renderer(vertices[t].cuda(),faces,K,W,width,height)
            depth=mesh['depth'].cpu().numpy();alpha=mesh['hard_alpha'].cpu().numpy().astype('uint8')
            assert np.isfinite(depth).all() and (depth[alpha>0]>0).all()
            points=hands[t].reshape(42,3);pc=points@np.array(v['W'])[:3,:3].T+np.array(v['W'])[:3,3];q=pc@K.cpu().numpy().T;uv=q[:,:2]/q[:,2:]
            unknown=np.zeros((height,width),np.uint8)
            for side in [0,1]:
                p=uv[side*21:side*21+21];good=np.isfinite(p).all(1)&(pc[side*21:side*21+21,2]>0)
                if good.sum()>=3:
                    polygon=cv2.convexHull(np.rint(p[good]).clip(-4096,4096).astype('int32'));cv2.fillConvexPoly(unknown,polygon,1)
            unknown=cv2.dilate(unknown,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
            depths.append(depth);alphas.append(alpha);unknowns.append(unknown)
        np.savez_compressed(out/f"{v['serial']}.npz",depth=depths,alpha=alphas,hand_unknown=unknowns,frames=poses['frames'].numpy(),K=K.cpu().numpy(),W=W.cpu().numpy())
        viewreports.append({'serial':v['serial'],'depth_shape':[T,height,width]});print('depth cached',v['serial'],flush=True)
torch.save({'vertices_world':vertices,'faces':faces.cpu(),'joints_world':joints,'frames':poses['frames']},out/'body_mesh.pt')
report={'success':True,'frames':T,'views':viewreports,'seconds':time.time()-start,'source_poses':'body_hands_all300_v1','coordinate_convention':'Positive OpenCV camera-space Z, top-left pixel centers; native K mapped to256×192 by explicit affine. Float32 depths, hard alpha.','hand_unknown':'Convex hull of21 projected points per hand, dilated3 cache pixels (24native pixels).','scope':'Actual full10475-vertex SMPL-X LBS from fitted motion, checked against verified subset at each batch; fixed beta0 body, not clothed Gaussian surface. These depths support conservative occlusion only, not calibrated human shape/contact truth. Use positive depth margin and retain hand/boundary uncertainty. Private licensed mesh tensors are not public assets.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('body depth cache complete',flush=True)
