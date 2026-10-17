"""Same-camera/pose/helmet comparison, no appearance adoption by metric proxy."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch
from PIL import Image,ImageDraw,ImageOps
out=ROOT/'lhm_multitime_visual_comparison_v1';out.mkdir(exist_ok=False)
a=torch.load(ROOT/'body_head_faces_all300_v1/avatar.pt',weights_only=True);b=torch.load(ROOT/'avatar_multitime16_all300_v1/avatar.pt',weights_only=True)
assert a['poses'].keys()==b['poses'].keys()
assert all(torch.equal(a['poses'][k],b['poses'][k]) for k in a['poses'] if isinstance(a['poses'][k],torch.Tensor))
assert a['decoder_config']==b['decoder_config'] and a['decoder'].keys()==b['decoder'].keys()
assert all(torch.equal(a['decoder'][k],b['decoder'][k]) for k in a['decoder'])
old=ROOT/'lhm_tight_consistent_portrait_probe_v1';new=ROOT/'lhm_multitime16_portrait_probe_v1'
for folder in [old,new]:assert json.loads((folder/'report.json').read_text())['success']
old_report=json.loads((old/'report.json').read_text());new_report=json.loads((new/'report.json').read_text())
for key in ['object_motion_source','object_mesh_source','occlusion_method','frames']:assert old_report[key]==new_report[key]
assert json.loads((old/'cameras.json').read_text())==json.loads((new/'cameras.json').read_text())
sheet=Image.new('RGB',(1134,2*720),'white');draw=ImageDraw.Draw(sheet)
for row,f in enumerate([42,299]):
 for col,folder in enumerate([old,new]):
  im=Image.open(folder/f'{f:04d}_orbit_032.png');im=im.crop((250,140,850,900));im=ImageOps.contain(im,(567,680));sheet.paste(im,(col*567,row*720+32));draw.text((col*567+8,row*720+8),f'frame{f} | '+('accepted static8 appearance' if col==0 else 'experimental static8 + dynamic8'),fill='black')
sheet.save(out/'upper_body_comparison.jpg',quality=95)
meta=json.loads((old/'cameras.json').read_text())
for v in meta['views'][:8]:
 serial=v['label'];f=299;sheet=Image.new('RGB',(1134*3,1552),'white');draw=ImageDraw.Draw(sheet)
 images=[ImageOps.pad(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png'),(1134,1512),color='white'),Image.open(old/f'{f:04d}_{serial}.png'),Image.open(new/f'{f:04d}_{serial}.png')]
 for col,(im,label) in enumerate(zip(images,['real source (letterboxed)','accepted static8 appearance','experimental static8 + dynamic8'])):sheet.paste(im,(col*1134,40));draw.text((col*1134+10,10),serial+' frame299 | '+label,fill='black')
 sheet.save(out/f'{serial}_frame299.jpg',quality=93)
(out/'report.json').write_text(json.dumps({'success':True,'pose_tensor_fields_identical':list(a['poses'].keys()),'decoder_and_config_identical':True,'camera_files_equal':True,'helmet_motion_and_texture_equal':True,'views_with_real_source_comparison':8,'scope':'Frozen accepted full300 motion and pretrained decoder, same camera K/W and helmet in two independently replayed Gaussian appearance branches. Selected42/180/299 rendered, not full300 new video. Real frame299 all8 sheets are input-view visual comparisons, no strict heldout claim. Changes include decoded Gaussian geometry/skin weights/features from new image conditioning, not just color. Original body never overwritten. Visual inspection only, not quality acceptance.'},indent=2));print('same-camera comparison ready',flush=True)
