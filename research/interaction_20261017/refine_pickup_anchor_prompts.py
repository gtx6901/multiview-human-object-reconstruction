"""Correct/reject anchors after actual overlay QA; preserve rejected v2 results."""
import os
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;out=ROOT/'pickup_anchor_prompts_v2';out.mkdir(exist_ok=False)
cfg=json.loads((ROOT/'pickup_anchor_prompts_v1/prompts.json').read_text());crops=json.loads((ROOT/'pickup_anchor_inspection_v1/crops.json').read_text())
reject={('FCM21121414',30),('FCM21121414',60),('FCM21121414',120),('FCM21121414',135),('FCM21121416',90),('FCM22020159',60),('FCM22020159',135),('FCM22020160',0),('FCM22020160',90),('FCM22020160',120),('FCM22030167',90)}
fix={
 ('21305711',120):([60,60,205,225],[[150,135],[145,185]],[[120,20],[205,220],[50,40],[100,240],[200,70]]),
 ('FCM21121415',60):([90,85,195,210],[[143,138],[115,125]],[[80,120],[120,185],[210,180]]),
 ('FCM21121415',120):([90,85,200,195],[[140,155],[135,115]],[[205,140],[200,65],[100,70],[110,210]]),
 ('FCM22030167',60):([90,85,180,185],[[130,125],[155,110]],[[150,155],[200,100],[80,100]])}
for p in cfg['records']:
 key=(p['serial'],p['frame'])
 if key in reject:
    p.update(status='unknown',reason='Actual v2 overlay QA rejected background/body/strap-dominated segmentation; no negative mask.')
 elif key in fix:
    c=next(x for x in crops if (x['serial'],x['frame'])==key);ox,oy=c['crop_xyxy'][:2];box,pos,neg=fix[key]
    p.update(box=[box[0]+ox,box[1]+oy,box[2]+ox,box[3]+oy],pos=[[x+ox,y+oy] for x,y in pos],neg=[[x+ox,y+oy] for x,y in neg],status='prompted',prompt_source='Corrected using actual 32px-grid source inspection and failed-overlay QA.')
(out/'prompts.json').write_text(json.dumps(cfg,indent=2))
