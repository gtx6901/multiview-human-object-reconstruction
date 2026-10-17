"""Differentiable SMPL-X surface COCO17 and MANO-style hands.

COCO17 = upstream GVHMR SMPL-COCO regressor @ its SMPL-X-to-SMPL map.
Only needed vertices are skinned; verify against full official LBS before use.
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix

class SMPLXSurfaceLandmarks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_path=ROOT.parent/'lhmpp_20261016/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz'
        d=np.load(self.model_path,allow_pickle=True)
        mapping=torch.load(ROOT/'smplx_coco_regressor_v1/smplx2smpl_sparse.pt',weights_only=True).to_dense().float()
        coco=torch.load(ROOT/'smplx_coco_regressor_v1/smpl_coco17_J_regressor.pt',weights_only=True).float()@mapping
        assert coco.shape==(17,10475) and torch.isfinite(coco).all()
        assert torch.allclose(coco.sum(1),torch.ones(17),atol=1e-4)
        tips=torch.tensor([5361,4933,5058,5169,5286,8079,7669,7794,7905,8022])
        ids=torch.unique(torch.cat([torch.where(coco.abs().sum(0)>0)[0],tips]),sorted=True)
        self.register_buffer('vertex_ids',ids)
        self.register_buffer('coco_regressor',coco[:,ids])
        self.register_buffer('tip_subset_ids',torch.searchsorted(ids,tips))
        template=torch.tensor(d['v_template'],dtype=torch.float32)
        jreg=torch.tensor(d['J_regressor'],dtype=torch.float32)
        rest=jreg@template
        expected=torch.load(ROOT/'body_pose_v1/skeleton.pt',weights_only=True)
        assert torch.allclose(rest,expected['rest_joints'][0],atol=1e-6)
        parents=torch.tensor(d['kintree_table'][0].astype(np.int64));parents[0]=-1
        assert torch.equal(parents,expected['parents'])
        self.register_buffer('rest',rest)
        self.register_buffer('parents',parents)
        self.register_buffer('template',template[ids])
        shapedirs=torch.tensor(d['shapedirs'][:,:,:10],dtype=torch.float32)
        self.register_buffer('shapedirs',shapedirs[ids])
        self.register_buffer('joint_shapedirs',torch.einsum('jv,vcl->jcl',jreg,shapedirs))
        self.register_buffer('weights',torch.tensor(d['weights'],dtype=torch.float32)[ids])
        self.register_buffer('posedirs',torch.tensor(d['posedirs'],dtype=torch.float32)[ids].reshape(len(ids)*3,486).T.contiguous())
        self.register_buffer('left_pca',torch.tensor(d['hands_componentsl'],dtype=torch.float32))
        self.register_buffer('right_pca',torch.tensor(d['hands_componentsr'],dtype=torch.float32))
        # Deliberately flat mean: the frozen Gaussian rig's zero fingers are flat.
        self.hand_joint_ids=[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],
                             [21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48]]

    def forward(self,axis_angles,scale=None,translation=None,betas=None):
        assert axis_angles.ndim==3 and axis_angles.shape[1:]==(55,3)
        B=len(axis_angles);R=axis_angle_to_matrix(axis_angles)
        rest=self.rest[None].expand(B,-1,-1);template=self.template[None]
        if betas is not None:
            assert betas.ndim==2 and betas.shape[1]==10 and len(betas) in [1,B] and torch.isfinite(betas).all()
            betas=betas.expand(B,-1)
            rest=rest+torch.einsum('bl,jcl->bjc',betas,self.joint_shapedirs)
            template=template+torch.einsum('bl,vcl->bvc',betas,self.shapedirs)
        joints,A=batch_rigid_transform(R,rest,self.parents)
        feature=(R[:,1:]-torch.eye(3,device=R.device)).reshape(B,486)
        posed=template+(feature@self.posedirs).reshape(B,-1,3)
        transform=(self.weights@A.reshape(B,55,16)).reshape(B,-1,4,4)
        vertices=(transform[:,:,:3,:3]@posed[...,None])[...,0]+transform[:,:,:3,3]
        coco=self.coco_regressor@vertices
        hands=[]
        for side,ids in enumerate(self.hand_joint_ids):
            joint=joints[:,ids];tip=vertices[:,self.tip_subset_ids[side*5:side*5+5]]
            hands.append(torch.cat([joint[:,:1],torch.stack([torch.cat([joint[:,1+i*3:4+i*3],tip[:,i:i+1]],1) for i in range(5)],1).reshape(B,20,3)],1))
        hands=torch.stack(hands,1)
        if scale is not None:coco=coco*scale;hands=hands*scale;joints=joints*scale
        if translation is not None:coco=coco+translation[:,None];hands=hands+translation[:,None,None];joints=joints+translation[:,None]
        return {'coco17':coco,'hands21':hands,'joints55':joints,'vertices_subset':vertices}
