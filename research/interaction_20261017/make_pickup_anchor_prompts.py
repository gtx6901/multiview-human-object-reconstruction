"""Manual annotations from the inspected six-time/eight-view crop sheet."""
import os
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;out=ROOT/'pickup_anchor_prompts_v1';out.mkdir(exist_ok=False)
crops=json.loads((ROOT/'pickup_anchor_inspection_v1/crops.json').read_text());rows={}
# Coordinates below are LOCAL to actual saved 256px source-image crops.
# None denotes unobserved/outside/occluded, never a negative object mask.
rows['21305711']=[None,None,None,None,None,([140,0,252,90],[[200,30]],[[170,90],[210,90]])]
rows['21305717']=[None,None,None,None,None,([40,20,135,115],[[80,60]],[[100,106],[180,120]])]
rows['FCM21121414']=[None,([70,90,210,205],[[135,148]],[[198,135],[150,205]]),([60,55,170,190],[[110,120],[145,90]],[[165,138],[232,100]]),([65,100,190,210],[[133,170]],[[126,118],[185,157]]),([25,130,110,210],[[60,180]],[[95,160],[140,100]]),([25,60,150,160],[[85,100]],[[145,125],[175,155]])]
rows['FCM21121415']=[([45,70,160,175],[[100,125]],[[60,85],[145,88]]),None,([135,50,240,190],[[193,100],[175,65]],[[105,70],[175,145]]),([105,90,205,195],[[163,140]],[[140,95],[205,170]]),([90,110,205,195],[[150,157]],[[210,110],[120,120]]),([75,5,195,110],[[130,45]],[[180,85],[100,150]])]
rows['FCM21121416']=[None,None,([35,100,135,200],[[80,135]],[[125,145],[127,70]]),([50,100,140,195],[[85,145]],[[125,120],[145,175]]),None,([125,25,200,110],[[164,55]],[[130,105],[120,130]])]
rows['FCM22020159']=[None,None,([55,85,155,200],[[110,120],[100,165]],[[147,155],[170,85]]),None,None,([50,50,140,140],[[100,80]],[[135,135],[165,120]])]
rows['FCM22020160']=[([45,80,165,165],[[100,120]],[[65,65],[165,110]]),None,None,([0,100,80,190],[[25,140]],[[58,153],[100,130]]),([10,60,115,165],[[70,90]],[[112,110],[120,160]]),([90,15,215,135],[[155,45]],[[140,105],[100,140]])]
rows['FCM22030167']=[None,None,([120,75,235,205],[[170,112]],[[190,165],[110,75]]),([60,95,165,210],[[112,175]],[[110,120],[165,100]]),([95,80,225,205],[[165,157]],[[150,110],[110,100]]),([105,15,250,145],[[205,65]],[[170,105],[130,120]])]
existing711={p['frame']:p for p in json.loads((ROOT/'helmet_temporal_prompts_v1/711.json').read_text())['records']}
existing717={p['frame']:p['prompt'] for p in json.loads((ROOT/'helmet_timeline717_v1/report.json').read_text())['records']}
f30={p['serial']:p['prompt'] for p in json.loads((ROOT/'helmet_masks_frame30_v2/report.json').read_text())['observed_views']}
records=[]
for c in crops:
    serial=c['serial'];f=c['frame'];i=[0,30,60,90,120,135].index(f);annotation=rows[serial][i]
    if serial=='21305711' and f in existing711 and existing711[f]['status']=='prompted':
        p=existing711[f];record={'box':p['box'],'pos':p['pos'],'neg':p.get('neg',[])}
    elif serial=='21305717' and f in existing717:
        p=existing717[f];record={k:p[k] for k in ['box','pos','neg']}
    elif f==30 and serial in f30:
        p=f30[serial];record={'box':p['box'],'pos':[p['positive']]+p.get('additional_positive',[]),'neg':p['negative']}
    elif annotation is not None:
        box,pos,neg=annotation;ox,oy=c['crop_xyxy'][:2]
        record={'box':[max(0,min(1023,box[0]+ox)),max(0,min(767,box[1]+oy)),max(0,min(1023,box[2]+ox)),max(0,min(767,box[3]+oy))],'pos':[[x+ox,y+oy] for x,y in pos],'neg':[[x+ox,y+oy] for x,y in neg if 0<=x+ox<1024 and 0<=y+oy<768]}
    else:
        records.append({'serial':serial,'frame':f,'status':'unknown','reason':'Occluded, outside image, or insufficiently clear in inspected source crop.'});continue
    record.update(serial=serial,frame=f,status='prompted',prompt_source='Actual all8 keyframe source-crop inspection; inherited previously inspected prompts where available; mask still requires visual QA.')
    records.append(record)
(out/'prompts.json').write_text(json.dumps({'records':records},indent=2))
