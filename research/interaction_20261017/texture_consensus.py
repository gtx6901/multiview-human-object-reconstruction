"""Camera-balanced color consistency; observations never vote by darkness."""
import os
import torch,math
def consistent_texel_weights(weights,colors,min_cameras=2,tolerance=.08,min_fraction=.6):
 """V,K,M weights, V,K,M,3 linear RGB; no gradients required for baking."""
 assert weights.ndim==3 and colors.shape==(*weights.shape,3)
 best,index=weights.max(1);representative=torch.gather(colors,1,index[:,None,:,None].expand(-1,1,-1,3))[:,0]
 known=best>.03;reference=torch.nanmedian(representative.masked_fill(~known[...,None],float('nan')),dim=0).values.nan_to_num(0)
 residual=(colors-reference[None,None]).abs().mean(-1);inlier=(residual<=tolerance)&(weights>.03)
 camera_inliers=inlier.any(1);camera_count=known.sum(0);agreeing=camera_inliers.sum(0)
 required=torch.maximum(torch.full_like(camera_count,min_cameras),torch.ceil(camera_count.float()*min_fraction).long());eligible=(camera_count>=min_cameras)&(agreeing>=required)
 filtered=weights*inlier*eligible[None,None]
 report={'minimum_independent_cameras':min_cameras,'maximum_mean_absolute_linear_rgb_difference':tolerance,'minimum_camera_agreement_fraction':min_fraction,'candidate_texels':int((camera_count>0).sum()),'single_camera_texels':int((camera_count==1).sum()),'eligible_texels':int(eligible.sum()),'conflicting_multicamera_texels':int(((camera_count>=min_cameras)&~eligible).sum()),'rejected_weight_fraction':float(1-filtered.sum()/weights.sum().clamp_min(1e-8)),'scope':'Each camera contributes at most one best-angle representative to an unweighted median reference, so one high-angle view cannot dominate the consistency vote. Retain samples close to that reference only with enough distinct cameras agreeing; rejected/single-view texels remain unknown and any later fill is inference. No stage restriction, white-shell or dark-lining color gate; common multi-camera registration bias can survive. Fixed lighting/exposure/pose ambiguity remains.'}
 return filtered,eligible,report
