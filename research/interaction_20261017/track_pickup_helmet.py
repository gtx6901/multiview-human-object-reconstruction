"""Anchor-bounded bidirectional flow/SAM for pickup; missing views stay unknown."""
import os
from pathlib import Path
import sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT.parent/'lhmpp_20261016/deps'),str(ROOT/'vendor/MobileSAM')]
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from mobile_sam import sam_model_registry,SamPredictor
p=argparse.ArgumentParser();p.add_argument('--out',default='helmet_track_pickup000_135_v1');p.add_argument('--extra-anchors');options=p.parse_args()
torch.set_num_threads(4);start=time.time();out=ROOT/options.out;out.mkdir(exist_ok=False)
data=ROOT/'helmet_pickup_inspect000_135_v1';meta=json.loads((data/'manifest.json').read_text());anchors=json.loads((ROOT/'pickup_anchor_masks_v3/report.json').read_text())['records']
for a in anchors:
 if a.get('mask'):a['mask_path']='pickup_anchor_masks_v3/'+a['mask']
 if a['serial']=='21305711' and a['frame']==120:a.update(mask=None,status='unknown; post-QA rejects shirt leakage')
if options.extra_anchors:
    for a in json.loads((ROOT/options.extra_anchors/'report.json').read_text())['records']:
        if a.get('mask'):a['mask_path']=options.extra_anchors+'/'+a['mask']
        anchors.append(a)
(out/'anchor_registry.json').write_text(json.dumps(anchors,indent=2))
model=sam_model_registry['vit_t'](checkpoint=str(ROOT/'vendor/MobileSAM/weights/mobile_sam.pt')).cuda().eval();predictor=SamPredictor(model)
flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);records=[]
def core(m):
    m=cv2.morphologyEx(m.astype('uint8'),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    n,lab,stats,_=cv2.connectedComponentsWithStats(m)
    return lab==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA])) if n>1 else m.astype(bool)
def box(m):
    y,x=np.where(m);return np.array([x.min(),y.min(),x.max()+1,y.max()+1],dtype='float32')
def gray(arr):return cv2.resize(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),(512,384))
for v in meta['views']:
    serial=v['serial'];folder=out/serial;folder.mkdir();seeds={a['frame']:a for a in anchors if a['serial']==serial and a.get('mask')};results={};images={}
    def rgb(f):
        if f not in images:images[f]=np.asarray(Image.open(data/serial/f'{f:04d}.png').convert('RGB'))
        return images[f]
    nearest={f:min(seeds,key=lambda a:abs(a-f)) for f in range(136)} if seeds else {}
    for anchor,a in seeds.items():
        initial=core(np.asarray(Image.open(ROOT/a['mask_path']))>127)
        seed_area=initial.sum();seed_box=box(initial)
        for direction in [1,-1]:
            previous=gray(rgb(anchor));mask=initial.copy();neg=np.array(a.get('neg',[]),dtype='float32').reshape(-1,2)
            targets=[anchor]+[f for f in range(anchor+direction,136 if direction>0 else -1,direction) if nearest[f]==anchor and abs(f-anchor)<=15]
            # Nearest-anchor regions are contiguous; never leap over missing frames.
            for ti,f in enumerate(targets):
                arr=rgb(f);current=gray(arr);diagnostics=[];checks=[];accepted=True
                if ti:
                    backward=flow.calc(current,previous,None);yy,xx=np.indices((384,512),dtype='float32');small=cv2.resize(mask.astype('uint8'),(512,384),interpolation=cv2.INTER_NEAREST)
                    warped=cv2.remap(small,xx+backward[...,0],yy+backward[...,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0;prior=cv2.resize(warped.astype('uint8'),(1024,768),interpolation=cv2.INTER_NEAREST)>0
                    if not prior.any():results[f]={'mask':None,'anchor':anchor,'diagnostics':['flow prior outside image']};break
                    forward=flow.calc(previous,current,None)
                    if len(neg):
                        loc=np.rint(neg/2).astype(int);loc[:,0]=loc[:,0].clip(0,511);loc[:,1]=loc[:,1].clip(0,383);neg+=forward[loc[:,1],loc[:,0]]*2;neg[:,0]=neg[:,0].clip(0,1023);neg[:,1]=neg[:,1].clip(0,767)
                    d=cv2.distanceTransform(prior.astype('uint8'),cv2.DIST_L2,3);y,x=np.unravel_index(d.argmax(),d.shape);positive=np.array([[x,y]],dtype='float32')
                    b=box(prior)+np.array([-12,-12,12,12]);b[[0,2]]=b[[0,2]].clip(0,1023);b[[1,3]]=b[[1,3]].clip(0,767)
                    predictor.set_image(arr);candidates,scores,_=predictor.predict(point_coords=np.concatenate([positive,neg]),point_labels=np.r_[1,np.zeros(len(neg),dtype=int)],box=b,multimask_output=True)
                    choices=[]
                    for j,candidate in enumerate(candidates):
                        candidate=core(candidate);area=candidate.sum();iou=float((candidate&prior).sum()/max(1,(candidate|prior).sum()));good=False
                        if area>100:
                            cb=box(candidate);relative=float(area/max(1,seed_area));shift=float(np.linalg.norm((cb[:2]+cb[2:]-b[:2]-b[2:])/2));good=bool(.2<relative<3 and iou>.15 and shift<max(30,float(np.max(b[2:]-b[:2])*.5)))
                        checks.append({'index':j,'sam_score':float(scores[j]),'warp_iou':iou,'pixels':int(area),'accepted':good})
                        if good:choices.append((.55*float(scores[j])+.45*iou,candidate))
                    if choices:mask=max(choices,key=lambda x:x[0])[1]
                    else:accepted=False;mask=prior;diagnostics.append('All candidates rejected; warped mask retained as proposal only')
                b=box(mask);roi=(b+np.array([-30,-30,30,30])).astype(int);roi[[0,2]]=roi[[0,2]].clip(0,1024);roi[[1,3]]=roi[[1,3]].clip(0,768)
                if accepted:
                    name=f'{serial}/{f:04d}.png';Image.fromarray(mask.astype('uint8')*255).save(out/name)
                else:name=None
                results[f]={'mask':name,'anchor':anchor,'anchor_distance':abs(f-anchor),'pixels':int(mask.sum()) if accepted else None,'roi_xyxy':roi.tolist(),'checks':checks,'diagnostics':diagnostics};previous=current
    sheet=Image.new('RGB',(10*192,216),'white')
    for f in range(136):
        r=results.get(f,{'mask':None,'diagnostics':['No visually accepted anchor within 15 frames; observation unknown']})
        qa_rejected=(serial=='21305711' and 100<=f<=126) or (serial=='FCM22020159' and 15<=f<=23) or (serial=='FCM22030167' and 15<=f<=23) or (serial=='FCM21121416' and 120<=f<=126)
        if qa_rejected:r={**r,'mask':None,'diagnostics':r.get('diagnostics',[])+['Conservative interval excluded after temporal overlay QA identified strap, arm, shirt, or sleeve drift; no observation used.']};results[f]=r
        record={'serial':serial,'frame':f,**r};records.append(record)
    for j,f in enumerate([0,15,30,45,60,75,90,105,120,135]):
        r=results.get(f,{});arr=rgb(f).copy()
        if r.get('mask'):
            m=np.asarray(Image.open(out/r['mask']))>127;arr[m]=np.uint8(arr[m]*.5+np.array([255,25,25])*.5);b=r['roi_xyxy'];pic=Image.fromarray(arr).crop(b)
        else:pic=Image.fromarray(arr)
        sheet.paste(pic.resize((192,192)),(j*192,24));ImageDraw.Draw(sheet).text((j*192+3,4),f'{f} '+('candidate' if r.get('mask') else 'UNKNOWN'),fill='black')
    sheet.save(out/f'{serial}_qa.jpg');print(serial,'tracking complete',flush=True)
report={'success':True,'frames':list(range(136)),'views_retained':8,'records':records,'observed_masks':sum(r['mask'] is not None for r in records),'unknown_masks':sum(r['mask'] is None for r in records),'seconds':time.time()-start,'scope':'Actual source-image bidirectional DIS flow plus MobileSAM, nearest visually screened anchor at most15frames away. Predicted masks, not GT. 5px opening removes straps/thin parts. Ambiguous/outside/occluded views and rejected candidates unknown; no constant worn-phase size assumption, no geometry or pose claim. 414 has no accepted anchor after exposure/background failure; all8 still constrain human motion.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('complete',report['observed_masks'],flush=True)
