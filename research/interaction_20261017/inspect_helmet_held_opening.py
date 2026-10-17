"""Native held-phase photo contact sheet for observable opening selection."""
import os
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parent;out=ROOT/'helmet_held_opening_photos_v1';out.mkdir(exist_ok=False)
records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']}
for serial in ['21305717','FCM21121415','FCM22020159']:
 sheet=Image.new('RGB',(5*280,3*304),'white');draw=ImageDraw.Draw(sheet)
 for i,f in enumerate(range(0,136,10)):
  r=records[serial,f]
  if not r['mask']:continue
  photo=Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB');box=r['roi_xyxy'];x,y=i%5*280,i//5*304;sheet.paste(ImageOps.pad(photo.crop(tuple(box)),(280,280),color='white'),(x,y+24));draw.text((x+5,y+5),f'{serial} f{f}',fill='black')
 sheet.save(out/f'{serial}.jpg',quality=93)
print('native held opening contact sheets ready',flush=True)
