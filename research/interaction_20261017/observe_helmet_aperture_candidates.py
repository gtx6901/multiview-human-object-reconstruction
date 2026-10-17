"""Image-derived dark-aperture proposals during inspection; no automatic geometry."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps
out=ROOT/'helmet_aperture_candidates_v1';out.mkdir(exist_ok=False)
records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());proposals=[]
for v in meta['views']:
 serial=v['serial'];sheet=Image.new('RGB',(4*384,2*408),'white');draw=ImageDraw.Draw(sheet)
 for i,f in enumerate([50,60,70,80]):
  r=records[serial,f]
  if not r['mask']:continue
  rgb=np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB'));obj=np.array(Image.open(ROOT/'sam2_all300_v1'/r['mask']))>127
  valid=cv2.erode(obj.astype('uint8'),np.ones((5,5),np.uint8))>0;x=rgb.astype('float32')/255;lum=x@np.array([.2126,.7152,.0722]);white=max(.3,float(np.quantile(lum[obj],.95)));dark=(lum<.22*white)&valid
  count,labels,stats,_=cv2.connectedComponentsWithStats(dark.astype('uint8'))
  if count<=1:continue
  index=1+int(stats[1:,cv2.CC_STAT_AREA].argmax());region=(labels==index).astype('uint8');area=int(region.sum());fraction=area/max(int(obj.sum()),1)
  contours,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);contour=max(contours,key=cv2.contourArea).reshape(-1,2);perimeter=float(cv2.arcLength(contour[:,None],True));solidity=area/max(float(cv2.contourArea(cv2.convexHull(contour))),1)
  proposed=.15<=fraction<=.9 and area>=80 and solidity>.65
  overlay=rgb.copy();overlay[region>0]=(rgb[region>0]*.4+np.array([255,0,255])*.6).astype('uint8')
  for row,image in enumerate([rgb,overlay]):sheet.paste(ImageOps.pad(Image.fromarray(image).crop(tuple(r['roi_xyxy'])),(384,384),color='white'),(i*384,row*408+24));draw.text((i*384+4,row*408+4),f'f{f} dark/object={fraction:.2f} proposal={proposed}',fill='black')
  proposals.append({'serial':serial,'frame':f,'candidate':proposed,'area_native_px':area,'fraction_of_object':fraction,'convex_solidity':solidity,'contour_native_xy':contour.tolist(),'perimeter_native_px':perimeter})
 sheet.save(out/f'{serial}.jpg',quality=94)
(out/'report.json').write_text(json.dumps({'success':True,'proposals':proposals,'scope':'Dark connected regions inside eroded native candidate helmet masks at inspection50/60/70/80. Threshold.22 of object95percentile luminance, area/fraction/solidity gates. These are image proposals, not certified cavity masks: visor, painted stripes, shadow or segmentation mistakes can pass. No head-based opening plane used; no3D plane or geometry adopted. Source overlays must be checked before fitting opening.'},indent=2));print('inspection aperture proposals ready',flush=True)
