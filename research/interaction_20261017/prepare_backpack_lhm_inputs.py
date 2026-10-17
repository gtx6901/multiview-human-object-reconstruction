"""Independent multitime backpack actor appearance inputs, without calibration use."""
import os
from pathlib import Path
import sys,json,time,hashlib,importlib.util,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np,cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='backpack_lhm_inputs16_v2');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
data=ROOT/'backpack_native000_098_v2';meta=json.loads((data/'manifest.json').read_text());det=np.load(ROOT/'backpack_wholebody_all99_v2/detections.npz')
weight=Path.home()/'.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth';assert weight.exists();model=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None).eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);model.transform.min_size=(640,);model.transform.max_size=1024;model.roi_heads.score_thresh=.3;model.roi_heads.detections_per_img=30
spec=importlib.util.spec_from_file_location('lhm_data_utils',LHM/'vendor/core/datasets/data_utils.py');util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
images=[];records=[]
with torch.inference_mode():
 for frame in [15,75]:
  for vi,v in enumerate(meta['views']):
   serial=v['serial'];rgb=np.asarray(Image.open(data/serial/f'{frame:04d}.png').convert('RGB'));x0,y0,x1,y1=np.r_[np.floor(det['boxes_native'][vi,frame,:2]),np.ceil(det['boxes_native'][vi,frame,2:])].astype(int);crop=rgb[y0:y1+1,x0:x1+1].copy();pred=model([torch.from_numpy(crop).permute(2,0,1).float()/255])[0]
   ids=torch.where((pred['labels']==1)&(pred['scores']>.6))[0];assert len(ids),f'Missing backpack actor {serial} {frame}';area=(pred['masks'][ids,0]>.5).sum((1,2));j=ids[(area.float()*pred['scores'][ids]).argmax()];person=(pred['masks'][j,0]>.5).numpy();bag=np.zeros_like(person);objects=[]
   for k in torch.where((pred['scores']>.6)&torch.isin(pred['labels'],torch.tensor([27,31,33])))[0]:
    mask=(pred['masks'][k,0]>.5).numpy()
    if mask.sum()>.4*person.sum():continue
    bag|=mask;objects.append({'coco_label':int(pred['labels'][k]),'confidence':float(pred['scores'][k]),'mask_pixels':int(mask.sum())})
   bag=cv2.dilate(bag.astype('uint8'),np.ones((5,5),np.uint8))>0;alpha=np.zeros(rgb.shape[:2],np.uint8);alpha[y0:y1+1,x0:x1+1]=np.uint8(person&~bag)*255
   name=f'{serial}_{frame:04d}';Image.fromarray(alpha).save(out/f'{name}_alpha.png');fullbag=np.zeros_like(alpha);fullbag[y0:y1+1,x0:x1+1]=bag*255;Image.fromarray(fullbag).save(out/f'{name}_bag_unknown.png');Image.fromarray(np.dstack([rgb,alpha])).save(out/f'{name}_rgba.png')
   a=alpha.astype('float32')/255;composite=rgb.astype('float32')*a[...,None]+255*(1-a[...,None]);tight,_,left,top,_=util.src_center_crop_according_to_mask(composite,a,5/3,[1.,1.],None);images.append(tight/255)
   records.append({'serial':serial,'frame':frame,'source_sha256':hashlib.sha256((data/serial/f'{frame:04d}.png').read_bytes()).hexdigest(),'person_score':float(pred['scores'][j]),'person_pixels':int(person.sum()),'retained_pixels':int((alpha>0).sum()),'object_exclusions':objects,'crop_origin':[int(left),int(top)]});print('backpack appearance crop',name,flush=True)
padded=util.PadRatioWithScale(target_ratio=5/3,tgt_max_size_list=[840],val=True)(images);inputs=torch.from_numpy(np.stack(padded)).permute(0,3,1,2).float();assert inputs.shape==(16,3,840,504);torch.save(inputs,out/'inputs.pt')
sheet=Image.new('RGB',(8*168,2*304),'white');draw=ImageDraw.Draw(sheet)
for i,x in enumerate(inputs):
 sheet.paste(Image.fromarray(np.uint8(x.permute(1,2,0).numpy().clip(0,1)*255)).resize((168,280)),((i%8)*168,(i//8)*304+24));draw.text(((i%8)*168+2,(i//8)*304+4),f"{records[i]['serial']} f{records[i]['frame']}",fill='black')
sheet.save(out/'inputs.jpg',quality=95)
(out/'report.json').write_text(json.dumps({'success':True,'input_shape':list(inputs.shape),'records':records,'mask_model_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'seconds':time.time()-start,'scope':'Independent backpack actor;16real native images,8cameras x times15/75. CPU COCO MaskRCNN person masks with confident limited-size backpack/handbag/suitcase candidate exclusions; mask/identity/occlusion QA required before inference. Exact upstream mask-center crop and840x504 padding. No calibration used, no helmet-actor feature reuse, no LHM avatar inferred yet. Missing/occluded cloth is unknown for prior completion, not recovered appearance. RGB/masks/input tensors contain original data and remain private.'},indent=2));print('backpack16 appearance inputs prepared; inspect before inference',flush=True)
