"""Measure whether visibly wrong underarm pixels actually received supervision."""
import os
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
p_arg=argparse.ArgumentParser();p_arg.add_argument('--cache',default='overhead_body_aligned_observations_all300_v2');p_arg.add_argument('--out',default='underarm_support_audit_v2');p_arg.add_argument('--after',help='Optional same-camera render for fixed initial underarm-error pixel comparison');o=p_arg.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
cache=ROOT/o.cache;report=json.loads((cache/'report.json').read_text());crops={v['serial']:v['native_crop_xyxy'] for v in report['camera_crop_records']};frames=[145,165,200,225]
render=ROOT/'overhead_body_appearance_after_review20_v2';cameras=json.loads((render/'cameras.json').read_text());cams={v['label']:v for v in cameras['views'][:8]};p=torch.load(ROOT/'grounded_legs_all300_v1/avatar.pt',weights_only=True)['poses'];gains=json.loads((ROOT/'overhead_body_aligned_features_v1/report.json').read_text())['fixed_camera_rgb_gains'];rows=[]
from smplx_surface_landmarks import SMPLXSurfaceLandmarks
from avatar_rig import pose_axis_angles
with torch.no_grad():worlds=SMPLXSurfaceLandmarks()(torch.cat([pose_axis_angles(p,f,'cpu') for f in frames]),p['scale'],p['translation_world'][frames],p['betas'])['coco17'].numpy()
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
for path in sorted(cache.glob('*.npz')):
 serial=path.stem
 with np.load(path) as d:
  ids=[d['frames'].tolist().index(f) for f in frames];targets=d['target_rgb_u8'][ids];supports=d['fixed_initial_support'][ids];masks=d['initial_decoded_alpha'][ids];K=d['K'];W=d['W'];reasons=d['selection_exclusion_bits'][ids] if 'selection_exclusion_bits' in d.files else None
 H=K@np.linalg.inv(np.array(cams[serial]['K']));w,h=report['resolution']
 for i,f in enumerate(frames):
  source=targets[i];prediction=cv2.warpAffine(np.array(Image.open(render/f'{f:04d}_{serial}.png').convert('RGB')),H[:2],(w,h),borderValue=(255,255,255))
  world=worlds[i];pc=world@W[:3,:3].T+W[:3,3];q=pc@K.T;xy=q[:,:2]/q[:,2:];torso=float(np.linalg.norm((xy[5]+xy[6]-xy[11]-xy[12])/2));roi=np.zeros((h,w),np.uint8)
  for sh,hip in [(5,11),(6,12)]:cv2.circle(roi,tuple(np.round(.75*xy[sh]+.25*xy[hip]).astype(int)),max(2,int(.18*torso)),1,-1)
  person=np.asarray(Image.open(ROOT/'sam2_all300_v1'/serial/'person'/f'{f:04d}.png').crop(crops[serial]).resize((w,h),Image.Resampling.NEAREST))>0;core=cv2.erode(person.astype('uint8'),np.ones((3,3),np.uint8))>0
  a=linear(source.astype('float32')/255)/np.array(gains[serial]);b=linear(prediction.astype('float32')/255)
  mismatch=(roi>0)&core&(masks[i]>.8)&(a.mean(-1)>.5)&(b.mean(-1)<.2);unclipped=source.max(-1)<250;bad=mismatch&unclipped;good=supports[i]
  rows.append({'serial':serial,'frame':f,'bright_source_dark_model_pixels':int(mismatch.sum()),'unclipped_mismatch_pixels':int(bad.sum()),'selected_unclipped_mismatch_pixels':int((bad&good).sum()),'selected_fraction':float((bad&good).sum()/max(int(bad.sum()),1)),'clipped_fraction':float((mismatch&~unclipped).sum()/max(int(mismatch.sum()),1))})
  if reasons is not None:rows[-1]['exclusion_reasons']={name:int((bad&((reasons[i]&bit)>0)).sum()) for bit,name in [(1,'person_core'),(2,'object_union'),(4,'decoded_alpha'),(8,'gaussian_accumulated_alpha'),(16,'anatomical_region'),(32,'boundary_band'),(64,'clipping')]}
  if o.after:
   after=cv2.warpAffine(np.array(Image.open(ROOT/o.after/f'{f:04d}_{serial}.png').convert('RGB')),H[:2],(w,h),borderValue=(255,255,255));af=linear(after.astype('float32')/255)
   rows[-1]['fixed_pixel_absolute_error_before_sum']=float(np.abs(a-b)[bad].sum());rows[-1]['fixed_pixel_absolute_error_after_sum']=float(np.abs(a-af)[bad].sum())
  overlay=source.copy();overlay[bad&~good]=[255,50,50];overlay[bad&good]=[30,230,80];sheet=Image.new('RGB',(w*3,h+24),'white')
  for j,(arr,label) in enumerate([(source,'PRIVATE source'),(prediction,'initial model'),(overlay,'red excluded / green selected')]):sheet.paste(Image.fromarray(arr),(j*w,24));ImageDraw.Draw(sheet).text((j*w+3,4),f'{serial} f{f} {label}',fill='black')
  sheet.save(out/f'{serial}_{f:04d}_private.jpg',quality=94)
 print('underarm support audited',serial,flush=True)
total=sum(x['unclipped_mismatch_pixels'] for x in rows);kept=sum(x['selected_unclipped_mismatch_pixels'] for x in rows)
r={'success':True,'frames':frames,'cameras':8,'records':rows,'unclipped_mismatch_pixels':total,'selected_unclipped_mismatch_pixels':kept,'pooled_selected_fraction':kept/max(total,1),'seconds':time.time()-start,'scope':'32 actual source/model/support comparisons, same projection and fixed calibrated response. Underarm ROIs from surface COCO shoulder/hip, conservative strong bright-source/dark-model thresholds. Tests whether such pixels entered the training loss, not ground-truth garment labels. Source-bearing panels PRIVATE; scalar counts may be shared.'}
if o.after:
 r['after_render']=o.after;r['fixed_initial_pixel_mae_before']=sum(x['fixed_pixel_absolute_error_before_sum'] for x in rows)/max(total*3,1);r['fixed_initial_pixel_mae_after']=sum(x['fixed_pixel_absolute_error_after_sum'] for x in rows)/max(total*3,1);r['scope']+=' After comparison uses the EXACT SAME initial error pixels and fixed old response, not a new after-dependent pixel set.'
(out/'report.json').write_text(json.dumps(r,indent=2))
