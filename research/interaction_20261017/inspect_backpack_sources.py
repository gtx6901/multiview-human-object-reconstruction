"""Read-only native backpack source inventory and temporal contact sheet."""
import os
from pathlib import Path
import json,hashlib,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent.parent;out=ROOT/'backpack_source_review_v1';out.mkdir(exist_ok=False)
paths=sorted((WORK/'dataAndVedio/vedio').glob('*/mjj_2026827pm/seuvcl_202608271840_002/*.mp4'));assert len(paths)==8
frames=[0,15,30,50,75,99];sheet=Image.new('RGB',(6*320,8*260),'white');views=[]
for i,path in enumerate(paths):
    cap=cv2.VideoCapture(str(path));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);assert n==100
    for j,f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pic=Image.fromarray(rgb);pic.thumbnail((320,240));sheet.paste(pic,(j*320,i*260+20));ImageDraw.Draw(sheet).text((j*320+2,i*260+2),f'{path.stem} f{f}',fill='black')
    cap.release();views.append({'serial':path.stem,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':w,'height':h,'frames':n,'fps_reported':fps})
sheet.save(out/'all8_timeline.jpg');calibration=[]
for name in ['20260827_ex.txt','20260827noon_in.txt']:
    path=WORK/'dataAndVedio'/name;calibration.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(out/'report.json').write_text(json.dumps({'success':True,'views':views,'sample_frames':frames,'calibration_candidates':calibration,'scope':'Read-only original backpack inventory and sampled raw-frame contact sheet. Native videos preserved; no calibrated alignment, identity reconstruction, bag segmentation, static-window acceptance or asset completion claimed. Images include original footage and remain private.'},indent=2));print('backpack source inventory ready',flush=True)
