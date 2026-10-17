"""Reject large interior object-occlusion holes instead of feeding white cloth."""
import os
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np
from PIL import Image,ImageDraw
source=ROOT/'backpack_lhm_inputs16_v2';r=json.loads((source/'report.json').read_text());assert r['success'];x=torch.load(source/'inputs.pt',weights_only=True);keep=[i for i,v in enumerate(r['records']) if 1-v['retained_pixels']/v['person_pixels']<.03];assert len(keep)>=8 and set(range(8))<=set(keep)
out=ROOT/'backpack_lhm_inputs_selected_v1';out.mkdir(exist_ok=False);torch.save(x[keep],out/'inputs.pt');sheet=Image.new('RGB',(5*168,2*304),'white');draw=ImageDraw.Draw(sheet)
for j,i in enumerate(keep):
 v=r['records'][i];sheet.paste(Image.fromarray(np.uint8(x[i].permute(1,2,0).numpy().clip(0,1)*255)).resize((168,280)),((j%5)*168,(j//5)*304+24));draw.text(((j%5)*168+2,(j//5)*304+4),f"{v['serial']} f{v['frame']}",fill='black')
sheet.save(out/'inputs.jpg',quality=95);records=[r['records'][i] for i in keep]
(out/'report.json').write_text(json.dumps({'success':True,'input_shape':list(x[keep].shape),'records':records,'selected_source_indices':keep,'source_inputs_sha256':hashlib.sha256((source/'inputs.pt').read_bytes()).hexdigest(),'scope':'All8 first-time images plus two arm-raised front views survive<=3percent person-mask removal by object exclusions. Large white holes at worn-backpack locations are rejected rather than conditioning white cloth into the shared person model. This10image set is an initialization selection only, not all792data fitting or evidence of complete masks/appearance; private original-derived inputs remain local.'},indent=2));print('selected clean initialization images',len(keep),keep,flush=True)
