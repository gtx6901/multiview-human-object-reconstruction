"""Explicit foreground/background/unknown masks for silhouette constraints."""
import os
import numpy as np
import cv2

def preserve_components(mask):
    """An occluder can split one object into several visible components.

    Component size alone cannot identify the occluded part of a helmet.
    Keep all predicted foreground pixels; reject contamination using observation
    QA rather than silently converting a disconnected piece to background.
    """
    mask=np.asarray(mask)
    if mask.ndim!=2:raise ValueError('Expected a 2D binary mask')
    return mask.astype(bool)

def silhouette_labels(mask,valid=None,roi=None,boundary_radius=3,occluder=None):
    """Return uint8 labels: 0 known background, 1 foreground, 255 unknown.

    Occluder is conservative unknown; measured object foreground takes priority.
    Pixels outside the valid image/ROI are always unknown, including positives.
    Boundary bands on both sides are unknown, not negative evidence.
    """
    if not isinstance(boundary_radius,int) or boundary_radius<0:raise ValueError('Invalid boundary radius')
    mask=preserve_components(mask);domain=np.ones_like(mask) if valid is None else np.asarray(valid,dtype=bool).copy()
    if domain.shape!=mask.shape:raise ValueError('Valid-mask shape mismatch')
    if roi is not None:
        if len(roi)!=4:raise ValueError('Expected xyxy ROI')
        x0,y0,x1,y1=map(int,roi);region=np.zeros_like(mask);h,w=mask.shape
        region[max(0,y0):min(h,max(0,y1)),max(0,x0):min(w,max(0,x1))]=True;domain&=region
    labels=np.zeros_like(mask,dtype=np.uint8)
    if occluder is not None:
        occ=np.asarray(occluder,dtype=bool)
        if occ.shape!=mask.shape:raise ValueError('Occluder shape mismatch')
        labels[occ]=255
    labels[mask]=1
    if boundary_radius:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*boundary_radius+1,2*boundary_radius+1))
        dilation=cv2.dilate(mask.astype('uint8'),kernel).astype(bool)
        erosion=cv2.erode(mask.astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        labels[dilation&~erosion]=255
    labels[~domain]=255
    return labels
