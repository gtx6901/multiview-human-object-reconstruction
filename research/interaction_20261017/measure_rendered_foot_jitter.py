"""Measure actual decoded video silhouettes with a shared fixed shoe region."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,imageio.v2 as imageio
from PIL import Image
folder=ROOT/'grounded_legs_integrated_review_v1';out=ROOT/'grounded_legs_rendered_jitter_v1';out.mkdir(exist_ok=False);report=json.loads((folder/'report.json').read_text());assert report['success'];rows=[]
for vi in range(2):
 reader=imageio.get_reader(str(folder/f'novel_{vi}_before_after_45_95.mp4'));frames=[frame for frame in reader];reader.close();assert len(frames)==51;a=np.stack(frames);width=a.shape[2]//2;panels=np.stack([a[:,:,:width],a[:,:,width:]],axis=0);mask=(panels.min(-1)<230);mask[:,:,:28]=False;union=mask.any((0,1));ys,xs=np.where(union);lo,hi=int(ys.min()),int(ys.max());height=hi-lo+1;y0=int(lo+.91*height);x0,x1=int(xs.min()),int(xs.max())+1;yy,xx=np.indices(mask.shape[-2:]);roi=(yy>=y0)&(yy<=hi)&(xx>=x0)&(xx<x1);centers=[]
 for k in range(2):
  m=mask[k]&roi;count=m.sum((1,2));assert (count>30).all();x=(m*xx).sum((1,2))/count;d2=np.diff(x,n=2);centers.append(x);rows.append({'camera':vi,'version':['before','grounded_legs'][k],'frames':51,'fixed_roi':[x0,y0,x1,hi+1],'body_height_pixels':height,'foot_region_centroid_d2_abs_median_pixels':float(np.median(abs(d2))),'foot_region_centroid_d2_abs_p95_pixels':float(np.percentile(abs(d2),95)),'normalized_median_per_thousand_height':float(np.median(abs(d2))/height*1000),'normalized_p95_per_thousand_height':float(np.percentile(abs(d2),95)/height*1000)})
 np.savez_compressed(out/f'camera{vi}_centroids.npz',before=centers[0],after=centers[1],frame_ids=np.arange(45,96))
 # Foot-only generated animation; displays normal speed and a repeated loop.
 cropframes=[Image.fromarray(frame).crop((0,max(y0-40,28),2*width,hi+10)) for frame in frames];cropframes[0].save(out/f'camera{vi}_feet_before_after.gif',save_all=True,append_images=cropframes[1:],duration=33,loop=0)
(out/'report.json').write_text(json.dumps({'success':True,'records':rows,'scope':'Independent measurement from actually decoded generated before/after videos,51continuous frames45..95. RGB min-channel<230 foreground; SAME fixed bottom9percent of combined body-height bounding box per camera and both versions. Horizontal centroid of combined foot region, absolute second differences normalized by body height. Includes neural decoding/video encoding. Region can include ankle pixels; this is rendered temporal stability, not source-motion or contact accuracy. Both source-view comparison and animation require review.'},indent=2));print(json.dumps(rows),flush=True)
