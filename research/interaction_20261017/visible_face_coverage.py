"""A helmet projection behind the visible human face is not an occlusion."""
import os
import torch
def face_coverage(alpha,mesh_depth,body_depth,xy,valid,margin=.01):
 batch=torch.arange(len(alpha),device=alpha.device)[:,None];x=xy[...,0];y=xy[...,1];a=alpha[batch,y,x];z=mesh_depth[batch,y,x];human=body_depth[batch,y,x]
 known=valid&(human>0);front=(z>0)&(z<human-margin)
 return (a*known*front).sum(),known.sum(),(a*valid).sum()
