"""Separate 16-image appearance candidate, preserving all eight static inputs."""
import os
from pathlib import Path
import sys,json,importlib.util,hashlib,argparse
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',default='lhm_multitime_inputs16_v2');o=p.parse_args()
out=ROOT/o.out;out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('lhm_data_utils',LHM/'vendor/core/datasets/data_utils.py');util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());records={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']}
baseline=torch.load(LHM/'runs/encode_all8_v2/inputs.pt',weights_only=True);assert baseline.shape[0]==8
images=[];provenance=[]
for i,v in enumerate(meta['views']):
 f=42 if i%2==0 else 90;r=records[v['serial'],f];source=ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png';rgb=np.array(Image.open(source).convert('RGB'));alpha=np.array(Image.open(ROOT/'sam2_all300_v1'/r['person_mask'])).astype('float32')/255
 valid=np.array(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/'valid.png'))>127;alpha*=valid
 if r['mask']:alpha[np.array(Image.open(ROOT/'sam2_all300_v1'/r['mask']))>127]=0
 composited=rgb.astype('float32')*alpha[...,None]+255*(1-alpha[...,None]);crop,_,left,top,_=util.src_center_crop_according_to_mask(composited,alpha,5/3,[1.,1.],None);images.append(crop/255)
 provenance.append({'serial':v['serial'],'dynamic_frame':f,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_resolution':[2048,1536],'crop_shape':list(crop.shape),'crop_origin':[int(left),int(top)]})
padded=util.PadRatioWithScale(target_ratio=5/3,tgt_max_size_list=[840],val=True)(images);extra=torch.from_numpy(np.stack(padded)).permute(0,3,1,2).float();assert extra.shape==baseline.shape
inputs=torch.cat([baseline,extra]);assert torch.equal(inputs[:8],baseline);torch.save(inputs,out/'inputs.pt')
sheet=Image.new('RGB',(168*8,280*2+40),'white');draw=ImageDraw.Draw(sheet)
for i,x in enumerate(inputs):sheet.paste(Image.fromarray(np.uint8(x.permute(1,2,0).numpy().clip(0,1)*255)).resize((168,280)),((i%8)*168,(i//8)*300+20))
draw.text((5,3),'TOP: original static all8 | BOTTOM: native held42/90, all8 cameras',fill='black');sheet.save(out/'inputs.jpg',quality=94)
(out/'report.json').write_text(json.dumps({'success':True,'input_shape':list(inputs.shape),'original_static_inputs_bit_identical':True,'extra_images':provenance,'scope':'Prepared candidate inputs ONLY, no new encoder/fusion/avatar inferred or adopted. Same actor identity based on existing FCS capture organization and visual review must be checked; original eight static image tensors are unchanged. Eight dynamic held42/90 native person masks with known helmet pixels removed, exact upstream mask crop and840x504 padding. Different times complement poses, not synchronized camera geometry. Selection is an initial candidate, not proof of best visibility or quality.'},indent=2));print('separate multitime16 candidate prepared',flush=True)
