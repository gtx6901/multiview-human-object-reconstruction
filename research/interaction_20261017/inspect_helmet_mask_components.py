"""Real native helmet crops and mask overlays for held/worn inspection."""
import os
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,cv2
from PIL import Image,ImageDraw
out=ROOT/'helmet_mask_held_worn_review_v1';out.mkdir(exist_ok=False)
records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};stats=[]
for serial in ['21305717','FCM21121414','FCM22020159']:
 sheet=Image.new('RGB',(768,4*408),'white');draw=ImageDraw.Draw(sheet)
 for row,f in enumerate([42,90,180,299]):
  rec=records[serial,f]
  if not rec['mask']:continue
  rgb=np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB'));mask=np.array(Image.open(ROOT/'sam2_all300_v1'/rec['mask']))>127;lab=np.array(Image.open(ROOT/'sam2_all300_v1'/rec['labels']))
  count,_,components,_=cv2.connectedComponentsWithStats(mask.astype('uint8'));parts=sorted(components[1:,cv2.CC_STAT_AREA].tolist(),reverse=True)
  x0,y0,x1,y1=map(int,rec['roi_xyxy']);x0=max(0,x0);y0=max(0,y0);x1=min(2048,x1);y1=min(1536,y1)
  overlay=rgb.copy();overlay[mask]=(rgb[mask]*.45+np.array([0,255,0])*.55).astype('uint8');unknown=(lab==255)&~mask;overlay[unknown]=(rgb[unknown]*.7+np.array([255,180,0])*.3).astype('uint8')
  from PIL import ImageOps
  for col,im in enumerate([rgb,overlay]):sheet.paste(ImageOps.pad(Image.fromarray(im[y0:y1,x0:x1]),(384,384),color='white'),(384*col,row*408+24))
  draw.text((5,row*408+5),f'{serial} frame{f} | photo / green helmet; amber unknown',fill='black');stats.append({'serial':serial,'frame':f,'components':count-1,'component_areas_native_px':parts})
 sheet.save(out/f'{serial}.jpg',quality=93)
(out/'report.json').write_text(json.dumps({'success':True,'records':stats,'scope':'Native candidate mask components and real-image overlays only. Does not certify headstrap inclusion or opening correctness. No largest-component filtering applied.'},indent=2));print('mask review ready',flush=True)
