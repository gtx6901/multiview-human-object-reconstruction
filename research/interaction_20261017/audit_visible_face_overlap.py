"""Inspect source face observations against mask uncertainty and helmet overlap."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from joint_head_constraints import JointHeadConstraints
from mesh_renderer import MeshRenderer
p=argparse.ArgumentParser();p.add_argument('--out',default='visible_face_overlap_audit_v1');o=p.parse_args();out=ROOT/o.out;out.mkdir(exist_ok=False);torch.set_num_threads(4);meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());guard=JointHeadConstraints(ROOT,'shared_shape_body_face_motion_v1','shared_shape_body_face_occlusion_all300_v1',meta,torch.ones(300,device='cuda',dtype=torch.bool),'shared_shape_head_face_anchors_v2',True)
records=json.loads((ROOT/'shared_shape_body_face_object_contacts_v1/report.json').read_text())['records'];lookup={(r['serial'],r['frame']):r for r in records};masks={(r['serial'],r['frame']):r for r in json.loads((ROOT/'sam2_all300_v1/report.json').read_text())['records']};motion=torch.load(ROOT/'shared_shape_body_face_object_motion_v1/sequence.pt',weights_only=True);mesh=torch.load(ROOT/'helmet_joint_pose_texture_all300_v2/mesh.pt',weights_only=True);renderer=MeshRenderer();rows=[]
with torch.no_grad():
 for vi,v in enumerate(meta['views']):
  sheet=Image.new('RGB',(1008,3*528),'white');cache=np.load(ROOT/'shared_shape_body_face_occlusion_all300_v1'/f"{v['serial']}.npz");cached_depth=cache['depth'];cached_alpha=cache['alpha']
  for row,t in enumerate([42,180,299]):
   r=lookup[(v['serial'],t)];rec=masks[(v['serial'],t)]
   if rec['mask'] is None:rows.append({'serial':v['serial'],'frame':t,'points':[]});continue
   K=torch.tensor(r['K_crop'],device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32);world=mesh['vertices_local'].cuda()@motion['R'][t].cuda().T*motion['logscale'].cuda().exp()+motion['translation_world'][t].cuda();render=renderer.depth_alpha_batch(world[None],mesh['faces'].cuda(),K[None],W[None],504,504);C=np.array(r['K_crop'])@np.linalg.inv(v['K']);xy=guard.xy[vi,t,:3].cpu().numpy();q=np.c_[xy,np.ones(3)]@C.T;uv=q[:,:2]/q[:,2:];pix=np.rint(uv).astype(int);source=Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{t:04d}.png').convert('RGB');origin=np.array(r['native_crop_origin']);factor=504/r['native_crop_extent'];source=source.transform((504,504),Image.Transform.AFFINE,(1/factor,0,origin[0],0,1/factor,origin[1]),resample=Image.Resampling.BICUBIC);generated=Image.open(ROOT/'shared_shape_body_face_object_contacts_v1'/f"{v['serial']}_{t:04d}_gaussian-center.png").convert('RGB')
   label=np.array(Image.open(ROOT/'sam2_all300_v1'/rec['labels']));obj=np.array(Image.open(ROOT/'sam2_all300_v1'/rec['mask']))>127;outside=cv2.distanceTransform((~obj).astype('uint8'),cv2.DIST_L2,5);native=np.rint(xy).astype(int);stats=[]
   for j,(x,y) in enumerate(pix):
    nx,ny=native[j];inside=0<=x<504 and 0<=y<504 and 0<=nx<2048 and 0<=ny<1536
    if not inside:continue
    valid=bool(guard.valid[vi,t,j]);strong=valid and outside[ny,nx]>=4 and float(guard.weights[vi,t,j])>.6;alpha=float(render['alpha'][0,y,x]);color='lime' if strong else ('orange' if valid else 'grey')
    for im in [source,generated]:draw=ImageDraw.Draw(im);draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);draw.text((x+5,y+3),str(j),fill=color)
    cx,cy=np.floor((xy[j]+.5)*.125).astype(int);cx=int(np.clip(cx,0,255));cy=int(np.clip(cy,0,191));bd=float(cached_depth[t,cy,cx]);bz=float(render['depth'][0,y,x]);reliable=bool(cached_alpha[t,cy,cx]>0 and bd>0);in_front=bool(reliable and bz>0 and bz<bd-.01)
    stats.append({'face_index':j,'initial_valid':valid,'source_object_label':int(label[ny,nx]),'inside_source_object_mask':bool(obj[ny,nx]),'outside_object_distance_native_px':float(outside[ny,nx]),'proposed_strong_visible':strong,'rendered_helmet_alpha':alpha,'helmet_depth':bz,'reliable_body_depth':bd,'body_depth_known':reliable,'helmet_unambiguously_in_front':in_front})
   for col,im in enumerate([source,generated]):sheet.paste(im,(col*504,row*528+24));ImageDraw.Draw(sheet).text((col*504+3,row*528+3),f'{v["serial"]} f{t} '+['SOURCE','GENERATED'][col]+' green=strong orange=uncertain',fill='black')
   rows.append({'serial':v['serial'],'frame':t,'points':stats})
  sheet.save(out/f"{v['serial']}_private.jpg",quality=95)
allpoints=[p for r in rows for p in r['points']];valid=[p for p in allpoints if p['initial_valid']];strong=[p for p in allpoints if p['proposed_strong_visible']];supported=[p for p in strong if p['body_depth_known']];summary={'initial_valid_points':len(valid),'initial_valid_overlap_fraction':float(np.mean([p['rendered_helmet_alpha']>.5 for p in valid])),'strong_visible_points':len(strong),'strong_visible_overlap_fraction':float(np.mean([p['rendered_helmet_alpha']>.5 for p in strong])) if strong else None,'initial_valid_inside_object_mask':sum(p['inside_source_object_mask'] for p in valid),'strong_body_depth_supported_points':len(supported),'depth_ordered_visible_overlap_points':sum(p['rendered_helmet_alpha']>.5 and p['helmet_unambiguously_in_front'] for p in supported)}
(out/'report.json').write_text(json.dumps({'success':True,'summary':summary,'records':rows,'scope':'Private source-image visual diagnostic. COCO nose/eyes from existing3view screen, initial actual-head directional and hand screening. Proposed stronger visible-face evidence additionally requires outside actual SAM object mask by>=4native pixels and rawscore>3, rather than merely label!=1 inside an uncertainty band. Point validity remains pseudo-observation; inspect source overlays before using as object negative evidence. No pose/model modified, no quality or true-contact claim.'},indent=2));print(json.dumps(summary),flush=True)
