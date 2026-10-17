"""Full-action native SAM2 candidate propagation with explicit unknown observations."""
import os
from pathlib import Path
import sys,json,time,gc,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps'),str(ROOT/'vendor/sam2')]
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2_video_predictor
from mask_observations import silhouette_labels
out=ROOT/'sam2_all300_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());anchors=[]
for source in ['pickup_anchor_masks_v3','pickup_extra_masks_v1']:
    for a in json.loads((ROOT/source/'report.json').read_text())['records']:
        if a.get('mask') and not (a['serial']=='21305711' and a['frame']==120):anchors.append({**a,'source':source,'path':source+'/'+a['mask']})
worn=json.loads((ROOT/'helmet_track180_211_v2/report.json').read_text())['records']
for a in worn:
    if a['frame'] in [180,211] and a.get('mask'):anchors.append({**a,'source':'helmet_track180_211_v2','path':'helmet_track180_211_v2/'+a['mask']})
(out/'anchor_registry.json').write_text(json.dumps(anchors,indent=2))
predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'sam2_runtime_v1/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False,hydra_overrides_extra=['++model.fill_hole_area=0'])
samples=[0,30,60,90,120,135,150,165,180,210,240,270,299]
records=[];viewreports=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for v in meta['views']:
        serial=v['serial'];folder=out/serial;folder.mkdir();video=folder/'video_frames';video.mkdir()
        for name in ['helmet_proposals','person','helmet_labels']:(folder/name).mkdir()
        # Decode directly, using the exact same native camera remap as frozen PNGs.
        cap=cv2.VideoCapture(v['source']);mx,my=cv2.initUndistortRectifyMap(np.array(v['K_native']),np.array(v['distortion_opencv5']),None,np.array(v['K']),(2048,1536),cv2.CV_32FC1)
        for f in range(300):
            ok,bgr=cap.read();assert ok
            rgb=cv2.cvtColor(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB);Image.fromarray(rgb).save(video/f'{f:04d}.jpg',quality=95)
        cap.release();assert hashlib.sha256(Path(v['source']).read_bytes()).hexdigest()==v['source_sha256']
        valid=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/serial/'valid.png'))>127
        seeds=[a for a in anchors if a['serial']==serial];assert seeds
        torch.cuda.reset_peak_memory_stats();state=predictor.init_state(str(video),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for a in seeds:
            mask=np.asarray(Image.open(ROOT/a['path']).resize((2048,1536),Image.Resampling.NEAREST))>127
            assert mask.any();predictor.add_new_mask(state,frame_idx=a['frame'],obj_id=1,mask=mask)
        for f,source in [(30,'people_pickup000_135_v1'),(180,'people_contiguous180_211_v1')]:
            path=ROOT/source/serial/f'{f:04d}_raw.png';assert path.exists()
            mask=np.asarray(Image.open(path).resize((2048,1536),Image.Resampling.NEAREST))>127
            predictor.add_new_mask(state,frame_idx=f,obj_id=2,mask=mask)
        first=min([a['frame'] for a in seeds]+[30]);current={};sheet=Image.new('RGB',(len(samples)*320,260),'white')
        for reverse in [False,True]:
            for f,ids,logits in predictor.propagate_in_video(state,start_frame_idx=first,reverse=reverse):
                m={i:(logits[j,0]>0).cpu().numpy() for j,i in enumerate(ids)};helmet=m[1];person=m[2];assert helmet.shape==person.shape==(1536,2048)
                # Preserve every disconnected component; constraints remain candidate evidence.
                hpath=f'{serial}/helmet_proposals/{f:04d}.png';ppath=f'{serial}/person/{f:04d}.png'
                Image.fromarray(helmet.astype('uint8')*255).save(out/hpath);Image.fromarray(person.astype('uint8')*255).save(out/ppath)
                reason=[]
                if not helmet.any():reason.append('No foreground proposed')
                if serial=='FCM21121414' and f<150:reason.append('No accepted pickup/inspection anchor in this view; proposal remains unknown')
                # Gross failures only; size consistency does not certify mask quality.
                if helmet.sum()>1536*2048*.08:reason.append('Gross foreground-area failure')
                roi=None
                if not reason:
                    yy,xx=np.where(helmet);roi=[int(xx.min())-60,int(yy.min())-60,int(xx.max())+61,int(yy.max())+61]
                    labels=silhouette_labels(helmet,valid=valid,roi=roi,boundary_radius=6,occluder=person)
                else:labels=np.full(helmet.shape,255,dtype='uint8')
                lpath=f'{serial}/helmet_labels/{f:04d}.png';Image.fromarray(labels).save(out/lpath)
                current[f]={'serial':serial,'frame':f,'mask':hpath if not reason else None,'proposal':hpath,'person_mask':ppath,'labels':lpath,'roi_xyxy':roi,'helmet_pixels':int(helmet.sum()),'person_pixels':int(person.sum()),'known_positive_pixels':int((labels==1).sum()),'known_background_pixels':int((labels==0).sum()),'unknown_reason':reason,'status':'candidate requires temporal/identity QA' if not reason else 'unknown'}
                if f in samples:
                    arr=np.array(Image.open(video/f'{f:04d}.jpg'));arr[helmet]=(arr[helmet]*.5+np.array([0,255,0])*.5).astype('uint8');arr[person&~helmet]=(arr[person&~helmet]*.8+np.array([255,0,0])*.2).astype('uint8')
                    pic=Image.fromarray(arr).resize((320,240));col=samples.index(f);sheet.paste(pic,(col*320,20));ImageDraw.Draw(sheet).text((col*320+3,3),f'{serial} f{f} '+('unknown' if reason else 'candidate'),fill='black')
        assert sorted(current)==list(range(300));records.extend(current[f] for f in range(300));sheet.save(folder/'timeline.jpg')
        vr={'serial':serial,'frames':300,'unknown':sum(r['mask'] is None for r in current.values()),'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'anchors':[{'frame':a['frame'],'path':a['path']} for a in seeds]};viewreports.append(vr)
        (folder/'report.json').write_text(json.dumps({'success':True,**vr,'records':[current[f] for f in range(300)]},indent=2))
        (out/'status.json').write_text(json.dumps({'completed_views':len(viewreports),'observations':len(records),'seconds':time.time()-start},indent=2));print('completed',serial,flush=True)
        del state,current;gc.collect();torch.cuda.empty_cache()
report={'success':True,'frames':list(range(300)),'views':viewreports,'records':records,'seconds':time.time()-start,'unknown':sum(r['mask'] is None for r in records),'scope':'Real SAM2.1 tiny full300 native video propagation in all8 cameras using frozen pseudo-mask anchors, CPU video/state offload. JPEG95 encodings of calibrated native video frames retained. No largest-component pruning. Tri-state0/1/255 labels separate foreground/background/unknown and retain conservative person occlusion/boundary uncertainty; rendered body-depth refinement not implemented here. Candidates require visual QA and are not ground truth. No acceptance/geometry/motion/texture or final-video completion claim.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('SAM2 all8 all300 completed',flush=True)
