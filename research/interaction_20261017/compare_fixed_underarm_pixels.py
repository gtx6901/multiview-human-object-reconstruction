"""Same fixed source/baseline underarm pixel set for a candidate comparison."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image
from avatar_rig import pose_axis_angles
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--after',required=True);p.add_argument('--before',default='grounded_shared_shape_review20_v2');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);torch.set_num_threads(4)
b=torch.load(ROOT/'avatar_grounded_shared_shape_v2/avatar.pt',weights_only=True);ps=b['poses'];frames=[145,165,200,225];cams=json.loads((ROOT/o.before/'cameras.json').read_text());assert cams==json.loads((ROOT/o.after/'cameras.json').read_text());native={v['serial']:v for v in json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text())['views']};w,h=cams['width'],cams['height'];rows=[]
with torch.no_grad():world=SMPLXSurfaceLandmarks()(torch.cat([pose_axis_angles(ps,f,'cpu') for f in frames]),ps['scale'],ps['translation_world'][frames],ps['betas'])['coco17'].numpy()
for c in cams['views'][:8]:
 serial=c['label'];K=np.array(c['K']);W=np.array(c['W']);H=K@np.linalg.inv(np.array(native[serial]['K']))
 for i,f in enumerate(frames):
  src=cv2.warpAffine(np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB')),H[:2],(w,h),borderValue=(255,255,255));a=np.array(Image.open(ROOT/o.before/f'{f:04d}_{serial}.png').convert('RGB'));z=np.array(Image.open(ROOT/o.after/f'{f:04d}_{serial}.png').convert('RGB'))
  person=cv2.warpAffine(np.array(Image.open(ROOT/'sam2_all300_v1'/serial/'person'/f'{f:04d}.png')),H[:2],(w,h),flags=cv2.INTER_NEAREST);core=cv2.erode((person>127).astype('uint8'),np.ones((3,3),np.uint8))>0
  pc=world[i]@W[:3,:3].T+W[:3,3];q=pc@K.T;uv=q[:,:2]/q[:,2:];torso=np.linalg.norm((uv[5]+uv[6]-uv[11]-uv[12])/2);roi=np.zeros((h,w),np.uint8)
  for sh,hip in [(5,11),(6,12)]:cv2.circle(roi,tuple(np.round(.75*uv[sh]+.25*uv[hip]).astype(int)),max(2,int(.18*torso)),1,-1)
  mask=(roi>0)&core&(src.mean(-1)>170)&(a.mean(-1)<120)&(src.max(-1)<250);n=int(mask.sum());ae=np.abs(src.astype('float32')-a)/255;ze=np.abs(src.astype('float32')-z)/255
  rows.append({'serial':serial,'frame':f,'pixels':n,'before_abs_sum':float(ae[mask].sum()),'after_abs_sum':float(ze[mask].sum())})
n=sum(r['pixels'] for r in rows);assert n>100;r={'success':True,'before':o.before,'after':o.after,'fixed_pixels':n,'before_mae_srgb':sum(r['before_abs_sum'] for r in rows)/(3*n),'after_mae_srgb':sum(r['after_abs_sum'] for r in rows)/(3*n),'records':rows,'scope':'Same baseline-defined bright-source/dark-rendered underarm pixels, original calibrated projection, source segmentation core and unclipped source, all8x4times. No after-dependent reselection or exposure recalibration. Input-image diagnostic; actual whole-candidate review determines adoption.'};(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True)
