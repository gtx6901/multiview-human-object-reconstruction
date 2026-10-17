"""Test silhouette pose ambiguity against observed white-shell/dark-lining evidence.

This audit does not modify any pose or claim that a two-material prior is truth.
"""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from mesh_renderer import MeshRenderer
out=ROOT/'helmet_material_orientation_audit_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
motion=torch.load(ROOT/'helmet_shell_pose_all300_v3/sequence.pt',weights_only=True);mesh=torch.load(ROOT/'helmet_shell_all300_v5/mesh.pt',weights_only=True)
meta=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text());obs=json.loads((ROOT/'sam2_all300_v1/report.json').read_text());records={(r['serial'],r['frame']):r for r in obs['records']};maskroot=ROOT/'sam2_all300_v1'
vertices=mesh['vertices_local'].cuda();faces=mesh['faces'].cuda();scale=motion['logscale'].exp().cuda();pivot=(vertices.min(0).values+vertices.max(0).values)/2
colors=torch.full_like(vertices,.05);colors[:mesh['outer_vertex_count']]=.9;renderer=MeshRenderer();width,height=256,192;A=np.array([[.125,0,-.4375],[0,.125,-.4375],[0,0,1.]])
group=Rotation.create_group('O');candidates=np.concatenate([np.eye(3)[None],group.as_matrix()]);frames=[0,30,42,60,90,120,150,180,240,299];rows=[];seeds=[];sheet=Image.new('RGB',(len(frames)*256,3*212),'white')
with torch.inference_mode():
    for col,f in enumerate(frames):
        targets=[]
        for vi,v in enumerate(meta['views']):
            r=records[(v['serial'],f)]
            if r['mask'] is None:continue
            labels=cv2.resize(np.asarray(Image.open(maskroot/r['labels'])),(width,height),interpolation=cv2.INTER_NEAREST)
            positive=labels==1
            if positive.sum()<15:continue
            raw=np.asarray(Image.open(ROOT/'helmet_native000_299_v1'/v['serial']/f'{f:04d}.png').convert('RGB'));rgb=cv2.resize(raw,(width,height),interpolation=cv2.INTER_AREA).astype('float32')/255
            luma=rgb@np.array([.2126,.7152,.0722]);white=max(.6,float(np.quantile(luma[positive],.95)));normalized=luma/white
            confidence=positive&((normalized>.7)|(normalized<.2));bright=normalized>.7
            cache=np.load(ROOT/'body_depth_all300_v1'/f"{v['serial']}.npz");confidence&=cache['hand_unknown'][f]==0
            targets.append((vi,torch.tensor(positive,device='cuda'),torch.tensor(labels!=255,device='cuda'),torch.tensor(confidence,device='cuda'),torch.tensor(bright,device='cuda'),torch.tensor(A@np.array(v['K']),device='cuda',dtype=torch.float32),torch.tensor(v['W'],device='cuda',dtype=torch.float32)))
        R0=motion['R'][f].cuda();t0=motion['translation_world'][f].cuda();scores=[];pictures={}
        for qi,q in enumerate(candidates):
            R=R0@torch.tensor(q,device='cuda',dtype=torch.float32);t=t0+(R0-R)@pivot*scale;world=vertices@R.T*scale+t;ious=[];photos=[]
            for vi,target,known,confidence,bright,K,W in targets:
                render=renderer(world,faces,K,W,width,height,colors)
                pred=render['alpha']*known;intersection=(pred*target).sum();union=(pred+target-pred*target).sum().clamp_min(1);ious.append(float(intersection/union))
                material=((render['premultiplied_rgb'][...,0]/render['alpha'].clamp_min(1e-5)-.05)/.85).clamp(0,1)
                photo=((material-bright.float()).abs()*confidence*render['alpha']).sum()/confidence.sum().clamp_min(1);photos.append(float(photo))
                if vi==1:
                    picture=np.uint8((render['premultiplied_rgb']+1-render['alpha'][...,None]).cpu().numpy().clip(0,1)*255);pictures[qi]=Image.fromarray(picture)
            angle=float(Rotation.from_matrix(q).magnitude());score=1-float(np.mean(ious))+.5*float(np.mean(photos))+.02*(angle/np.pi)**2
            scores.append({'candidate':qi,'joint_cost':score,'soft_iou':float(np.mean(ious)),'material_error':float(np.mean(photos)),'delta_rotation_deg':float(np.rad2deg(angle))})
        best=min(scores,key=lambda r:r['joint_cost']);q=torch.tensor(candidates[best['candidate']],dtype=torch.float32);newR=motion['R'][f]@q;newt=motion['translation_world'][f]+(motion['R'][f]-newR)@pivot.cpu()*scale.cpu()
        seeds.append({'frame':f,'R':newR,'translation_world':newt});rows.append({'frame':f,'current':scores[0],'best_discrete_candidate':best,'candidates':scores})
        rgb=Image.open(ROOT/'helmet_native000_299_v1/21305717'/f'{f:04d}.png').resize((256,192));sheet.paste(rgb,(col*256,20))
        for row,index in [(1,0),(2,best['candidate'])]:
            if index in pictures:sheet.paste(pictures[index],(col*256,row*212+20))
        ImageDraw.Draw(sheet).text((col*256+2,2),f'f{f} source / current / best prior',fill='black');print('material orientation audited',f,flush=True)
torch.save({'frames':torch.tensor([s['frame'] for s in seeds]),'R':torch.stack([s['R'] for s in seeds]),'translation_world':torch.stack([s['translation_world'] for s in seeds])},out/'seed_candidates.pt');sheet.save(out/'source_current_material_candidate.jpg')
(out/'report.json').write_text(json.dumps({'success':True,'seconds':time.time()-start,'records':rows,'pose_changes_applied':False,'scope':'Audit only. Explicit observed-design two-material prior (white outer/dark inner),25 discrete rotations including duplicate identity, local bbox center preserved, all8 candidate masks/native colors, hand exclusions, uncertain middle brightness ignored. Photometric term is a coarse material-class proxy, not exact calibrated color/pose truth. Selective-frame candidates must be reviewed and jointly refitted/temporally checked before any adoption; no held-out evidence.'},indent=2));print('material orientation audit complete',flush=True)
