"""Tensor-only replay of the validated fixed-shape LHM Gaussian skinning contract."""
import os
import torch
from smplx.lbs import batch_rigid_transform
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_quaternion,quaternion_multiply
def animate(rig,poses,t):
    aa=torch.cat([poses['root_pose'][t:t+1,None],poses['body_pose'][t:t+1],torch.zeros(1,33,3,device=rig['weights'].device)],1)
    _,A=batch_rigid_transform(axis_angle_to_matrix(aa),rig['rest_joints'],rig['parents'])
    transform=(rig['weights']@A[0].reshape(-1,16)).reshape(-1,4,4)
    hom=torch.cat([rig['zero_coords'],torch.ones_like(rig['zero_coords'][:,:1])],1)
    xyz=(transform@hom[...,None])[:,:3,0]*poses['scale']+poses['translation_world'][t]
    rotation_matrix=(transform@rig['neutral_to_zero'])[:,:3,:3]
    quaternion=torch.nn.functional.normalize(matrix_to_quaternion(rotation_matrix),dim=-1)
    rotation=quaternion_multiply(quaternion,rig['neutral_rotation'])
    return {'xyz':xyz.float(),'rotation':rotation.float(),'scaling':(rig['scaling']*poses['scale']).float(),'opacity':rig['opacity'].float(),'shs':rig['shs'].float()}
