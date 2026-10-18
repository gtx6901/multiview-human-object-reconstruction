"""Readable20-frame review pages/videos; source-bearing pages stay PRIVATE."""
import os
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--avatar',default='grounded_legs_worn_head_v2');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
old=ROOT/o.before;new=ROOT/o.after;r0=json.loads((old/'report.json').read_text());r1=json.loads((new/'report.json').read_text());assert r0['success'] and r1['success'] and r0['frames']==r1['frames'];cams=json.loads((new/'cameras.json').read_text());assert cams==json.loads((old/'cameras.json').read_text());frames=r1['frames'];w,h=cams['width'],cams['height'];meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());native={v['serial']:v for v in meta['views']};b=torch.load(ROOT/o.avatar/'avatar.pt',weights_only=True);poses=b['poses'];human_scale=float(poses['scale']);novel=cams['views'][8:];writers=[imageio.get_writer(str(out/f'novel{i}_keyframe_review.mp4'),format='FFMPEG',fps=30,codec='libx264',macro_block_size=2) for i in range(2)]
try:
 for f in frames:
  private=Image.new('RGB',(3*284,8*402),'white');generated=Image.new('RGB',(2*284,8*402),'white');head_private=Image.new('RGB',(3*320,8*344),'white');head_generated=Image.new('RGB',(2*320,8*344),'white')
  for vi,c in enumerate(cams['views'][:8]):
   serial=c['label'];K=np.array(c['K']);W=np.array(c['W']);H=K@np.linalg.inv(np.array(native[serial]['K']));assert np.allclose(H[2],[0,0,1]);raw=np.array(Image.open(ROOT/'helmet_native000_299_v1'/serial/f'{f:04d}.png').convert('RGB'));raw=cv2.warpAffine(raw,H[:2],(w,h),borderValue=(255,255,255));pics=[Image.fromarray(raw),Image.open(old/f'{f:04d}_{serial}.png'),Image.open(new/f'{f:04d}_{serial}.png')]
   for k,pic in enumerate(pics):private.paste(pic.resize((284,378)),(k*284,vi*402+24))
   for k,pic in enumerate(pics[1:]):generated.paste(pic.resize((284,378)),(k*284,vi*402+24))
   ImageDraw.Draw(private).text((3,vi*402+4),f'{serial} f{f} | PRIVATE source / before / update',fill='black');ImageDraw.Draw(generated).text((3,vi*402+4),f'{serial} f{f} | before / integrated update',fill='black')
   if f>=180:
    pc=W[:3,:3]@poses['joints55_world'][f,15].numpy()+W[:3,3];q=K@pc;center=q[:2]/q[2];extent=float(K[0,0]*human_scale/pc[2]*.36);box=(int(center[0]-extent),int(center[1]-extent),int(center[0]+extent),int(center[1]+extent))
    for k,pic in enumerate(pics):head_private.paste(pic.crop(box).resize((320,320)),(k*320,vi*344+24))
    for k,pic in enumerate(pics[1:]):head_generated.paste(pic.crop(box).resize((320,320)),(k*320,vi*344+24))
    ImageDraw.Draw(head_private).text((3,vi*344+4),f'{serial} f{f} | PRIVATE source / before / update',fill='black');ImageDraw.Draw(head_generated).text((3,vi*344+4),f'{serial} f{f} | before / integrated update',fill='black')
  private.save(out/f'all8_{f:04d}_private.jpg',quality=94);generated.save(out/f'all8_{f:04d}_generated.jpg',quality=94)
  if f>=180:head_private.save(out/f'head8_{f:04d}_private.jpg',quality=95);head_generated.save(out/f'head8_{f:04d}_generated.jpg',quality=95)
  for vi,c in enumerate(novel):
   pair=Image.new('RGB',(1134,780),'white')
   for k,folder in enumerate([old,new]):pair.paste(Image.open(folder/f'{f:04d}_{c["label"]}.png').resize((567,756)),(k*567,24))
   ImageDraw.Draw(pair).text((4,5),f'KEYFRAME {f}: before / integrated update (one-second hold; not continuous motion)',fill='black');arr=np.asarray(pair)
   for _ in range(30):writers[vi].append_data(arr)
  print('storyboard original frame',f,flush=True)
finally:
 for writer in writers:writer.close()
checks=[]
for vi in range(2):
 reader=imageio.get_reader(str(out/f'novel{vi}_keyframe_review.mp4'));n=sum(1 for _ in reader);reader.close();assert n==len(frames)*30;checks.append({'camera':vi,'decoded_frames':n,'original_keyframes':len(frames)})
(out/'report.json').write_text(json.dumps({'success':True,'before':o.before,'after':o.after,'avatar':o.avatar,'original_frame_ids':frames,'real_cameras':8,'video_checks':checks,'seconds':time.time()-start,'scope':'Matched-camera whole-candidate review. Each of20 original keyframes held1second for inspection, explicitly NOT continuous action video. Original RGB panels only in *_private.jpg; generated-only pages/videos may be published. Model-projected head crops use SAME rectangle for source/before/after. Camera matrices/raw source pixels are not in the public report.'},indent=2));print('integrated source/generated storyboards ready',flush=True)
