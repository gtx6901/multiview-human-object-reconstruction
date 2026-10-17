"""Freeze all eight native backpack streams with their OWN calibration."""
import os
from pathlib import Path
import json,hashlib,time,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent;out=ROOT/'backpack_native000_098_v2'
assert shutil.disk_usage(ROOT).free>12*1024**3,'Insufficient workspace disk headroom'
out.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
calibration=WORK/'dataAndVedio/20260827_ex.txt';blocks=[b.splitlines() for b in calibration.read_text().strip().split('\n\n') if b.strip()];cameras={}
for block in blocks:
    lines=[line.strip() for line in block if line.strip()];assert len(lines)==8
    serial=lines[0];K=np.array([[float(x) for x in row.split()] for row in lines[1:4]]);dist=np.array([float(x) for x in lines[4].split()]);W=np.eye(4);W[:3]=np.array([[float(x) for x in row.split()] for row in lines[5:8]])
    assert abs(np.linalg.det(W[:3,:3])-1)<1e-4 and np.max(abs(W[:3,:3].T@W[:3,:3]-np.eye(3)))<1e-4
    assert serial not in cameras;cameras[serial]=(K,dist,W)
paths=sorted((WORK/'dataAndVedio/vedio').glob('*/mjj_2026827pm/seuvcl_202608271840_002/*.mp4'));assert len(paths)==8 and {p.stem for p in paths}==set(cameras)
audit=json.loads((ROOT/'backpack_decode_audit_v1/report.json').read_text());counts={v['serial']:v['opencv_decoded_frames'] for v in audit['views']};assert all(v['opencv_decoded_frames']==v['ffmpeg_decoded_frames']==99 for v in audit['views'])
rows=[];samples=[0,15,30,50,75,98];sheet=Image.new('RGB',(6*256,8*212),'white')
for i,path in enumerate(paths):
    cap=cv2.VideoCapture(str(path));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));header_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=counts[path.stem];fps=cap.get(cv2.CAP_PROP_FPS);assert (w,h,header_count,n)==(2048,1536,100,99)
    K,dist,Wraw=cameras[path.stem];mx,my=cv2.initUndistortRectifyMap(K,dist,None,K,(w,h),cv2.CV_32FC1);valid=cv2.remap(np.full((h,w),255,np.uint8),mx,my,cv2.INTER_NEAREST)
    directory=out/path.stem;directory.mkdir();Image.fromarray(valid).save(directory/'valid.png');timeline=[]
    for f in range(n):
        ok,bgr=cap.read();assert ok;timeline.append({'frame':f,'media_pts_ms':float(cap.get(cv2.CAP_PROP_POS_MSEC))});rgb=cv2.cvtColor(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB);Image.fromarray(rgb).save(directory/f'{f:04d}.png')
        if f in samples:
            col=samples.index(f);sheet.paste(Image.fromarray(rgb).resize((256,192)),(col*256,i*212+20));ImageDraw.Draw(sheet).text((col*256+2,i*212+2),f'{path.stem} f{f}',fill='black')
    assert not cap.read()[0];cap.release();W=Wraw.copy();W[:3,3]/=1000
    effective_fps=(n-1)*1000/(timeline[-1]['media_pts_ms']-timeline[0]['media_pts_ms'])
    rows.append({'id':i,'serial':path.stem,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':n,'metadata_frames':header_count,'fps_reported':fps,'fps_from_decoded_pts':effective_fps,'timeline':timeline,'width':w,'height':h,'native_width':w,'native_height':h,'K':K.tolist(),'K_native':K.tolist(),'K_undistorted_native':K.tolist(),'distortion_opencv5':dist.tolist(),'native_to_output':np.eye(3).tolist(),'W':W.tolist(),'W_raw':Wraw.tolist()});print('backpack native complete',path.stem,flush=True)
sheet.save(out/'all8_timeline.jpg')
(out/'manifest.json').write_text(json.dumps({'success':True,'sequence':'seuvcl_202608271840_002','views':rows,'sample_frames':list(range(99)),'calibration':str(calibration),'calibration_sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),'world_normalization':{'center_raw':[0,0,0],'raw_units_per_numerical_unit':1000},'seconds':time.time()-start,'scope':'All792 native2048x1536 undistorted PNGs,99 ACTUALLY decoded frames per camera confirmed independently byOpenCV/FFmpeg. Header100 is inaccurate; actual PTS retained rather than inventing frame99. Uses backpack20260827 extrinsics, not helmet calibration. World coordinates are raw calibration units divided1000 for numerical conditioning ONLY, not measured metres; no actor/object normalization or shape estimated. Same-index/PTS correspondence is provisional, not hardware synchronization acceptance. Separate actor appearance required.'},indent=2));print('backpack dataset prepared',flush=True)
