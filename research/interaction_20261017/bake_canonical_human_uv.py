"""Bake72 neural-avatar targets into the original SMPL-X UV layout."""
import os
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')];sys.path.append(os.environ.get("HOI_SHARED_SITE", ""))
import torch,numpy as np,trimesh,nvdiffrast.torch as dr
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from mesh_renderer import MeshRenderer
from helmet_texture_sampling import srgb_to_linear,linear_to_srgb
out=ROOT/'human_mesh_textured_v1';out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4)
source=ROOT/'human_mesh_canonical_v2';mesh=torch.load(source/'mesh.pt',weights_only=True);quality=json.loads((source/'report.json').read_text());assert quality['canonical_inverted_triangles']==0
views=ROOT/'avatar_canonical_views_v2';meta=json.loads((views/'cameras.json').read_text());faces=mesh['faces'].long().numpy();ft=mesh['official_uv_faces'].long().numpy();uv=mesh['official_uv_vertices'].float().cuda()
mapping=np.full(len(uv),-1,np.int64)
for tri,uvtri in zip(faces,ft):
    for v,t in zip(tri,uvtri):
        assert mapping[t] in [-1,v],'Official UV index maps to incompatible vertices'
        mapping[t]=v
used=mapping>=0;safe=np.maximum(mapping,0);vertices=mesh['vertices_Apose_world'].float().cuda();indices=torch.tensor(ft.astype('int32'),device='cuda')
tm=trimesh.Trimesh(mesh['vertices_Apose_world'].numpy(),faces,process=False);normals=torch.tensor(np.array(tm.vertex_normals)[safe],device='cuda',dtype=torch.float32)
size=1024;context=dr.RasterizeCudaContext();clip=torch.cat([uv*2-1,torch.zeros(len(uv),1,device='cuda'),torch.ones(len(uv),1,device='cuda')],1)[None]
rast,_=dr.rasterize(context,clip.contiguous(),indices,resolution=[size,size]);rast=rast.flip(1).contiguous();fid=rast[0,...,3].long()-1;mask=fid>=0;y,x=torch.where(mask)
xyz,_=dr.interpolate(vertices[torch.tensor(safe,device='cuda')][None].contiguous(),rast,indices);ns,_=dr.interpolate(normals[None].contiguous(),rast,indices)
points=xyz[0][mask];normal=F.normalize(ns[0][mask],dim=1);M=len(points);renderer=MeshRenderer();topweights=torch.zeros(8,M,device='cuda');topcolors=torch.zeros(8,M,3,device='cuda');topview=torch.full((8,M),-1,device='cuda',dtype=torch.int16)
with torch.inference_mode():
    for i,v in enumerate(meta['views']):
        K=torch.tensor(v['K'],device='cuda',dtype=torch.float32);W=torch.tensor(v['W'],device='cuda',dtype=torch.float32)
        render=renderer.depth_alpha_batch(vertices[None],mesh['faces'].cuda(),K[None],W[None],meta['width'],meta['height'])
        pc=points@W[:3,:3].T+W[:3,3];q=pc@K.T;uvimage=q[:,:2]/q[:,2:].clamp_min(.001);grid=((uvimage+.5)/torch.tensor([meta['width'],meta['height']],device='cuda')*2-1).reshape(1,-1,1,2)
        def sample(tensor,mode='nearest'):
            if tensor.ndim==2:tensor=tensor[None,None]
            return F.grid_sample(tensor.float(),grid,mode=mode,padding_mode='zeros',align_corners=False)[0,:,:,0].T
        depth=sample(render['depth'][0])[:,0];hard=sample(render['hard_alpha'][0])[:,0]>.5
        targetalpha=torch.tensor(np.array(Image.open(views/v['alpha'])),device='cuda').float()/255
        camera=-W[:3,:3].T@W[:3,3];cos=(normal*F.normalize(camera-points,dim=1)).sum(1).clamp(0,1)
        good=(pc[:,2]>.01)&hard&((pc[:,2]-depth).abs()<.003)&(sample(targetalpha)[:,0]>.9)&(cos>.2)
        rgb=torch.tensor(np.array(Image.open(views/v['rgb']).convert('RGB')),device='cuda').permute(2,0,1)[None].float()/255;color=srgb_to_linear(sample(rgb,'bilinear'));weight=cos.pow(3)*good
        values,order=torch.cat([topweights,weight[None]],0).topk(8,dim=0)
        candidates=torch.cat([topcolors,color[None]],0);ids=torch.cat([topview,torch.full((1,M),i,device='cuda',dtype=torch.int16)],0)
        topweights=values;topcolors=torch.gather(candidates,0,order[...,None].expand(-1,-1,3));topview=torch.gather(ids,0,order)
        if i%12==0:print('human UV sampled views',i,flush=True)
    total=topweights.sum(0);observed=total>.02;channels=[]
    for j in range(3):
        value,order=topcolors[:,:,j].sort(0);weights=torch.gather(topweights,0,order);pick=(weights.cumsum(0)>=total[None]*.5).long().argmax(0);channels.append(torch.gather(value,0,pick[None])[0])
    colors=linear_to_srgb(torch.stack(channels,1)).cpu().numpy();seen=observed.cpu().numpy();yy=y.cpu().numpy();xx=x.cpu().numpy();faceids=fid[mask].cpu().numpy()
    # Part labels only govern explicit repair, never mark repaired pixels observed.
    dominant=mesh['weights'][mesh['faces'].long()].mean(1).argmax(1).numpy();parts=np.full(len(dominant),2)
    parts[np.isin(dominant,[12,15,22,23,24])]=0;parts[np.isin(dominant,[18,19,20,21]+list(range(25,55)))]=1;parts[np.isin(dominant,[0,1,2,4,5])]=3;parts[np.isin(dominant,[7,8,10,11])]=4
    partimage=np.full((size,size),-1,np.int16);partimage[yy,xx]=parts[faceids];observedimage=np.zeros((size,size),bool);observedimage[yy,xx]=seen;texture=np.zeros((size,size,3),np.float32);texture[yy,xx]=colors;repairs=[]
    for part in range(5):
        region=partimage==part;known=region&observedimage;unknown=region&~observedimage;median=np.median(texture[known],axis=0) if known.any() else np.array([.5,.5,.5]);texture[unknown]=median
        if known.any():
            distance,nearest=distance_transform_edt(~known,return_indices=True);fill=unknown&(distance<32);texture[fill]=texture[nearest[0][fill],nearest[1][fill]]
        repairs.append({'part':part,'observed_texels':int(known.sum()),'repaired_texels':int(unknown.sum()),'class_median':median.tolist()})
    validimage=partimage>=0;distance,nearest=distance_transform_edt(~validimage,return_indices=True);pad=(~validimage)&(distance<=4);texture[pad]=texture[nearest[0][pad],nearest[1][pad]]
    image=np.uint8(texture.clip(0,1)*255);Image.fromarray(image).save(out/'human_basecolor.png');Image.fromarray(observedimage.astype('uint8')*255).save(out/'observed_texels.png');Image.fromarray((validimage&~observedimage).astype('uint8')*255).save(out/'inferred_texels.png')
    np.savez_compressed(out/'texture_evidence.npz',texel_y=yy,texel_x=xx,source_views=topview.cpu().numpy(),source_weights=topweights.cpu().numpy().astype('float16'),observed=seen)
    result={**mesh,'texture':torch.tensor(image.copy(),dtype=torch.float32)/255};torch.save(result,out/'mesh.pt')
    (out/'human.mtl').write_text('newmtl HumanBaked\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nmap_Kd human_basecolor.png\n',encoding='utf-8')
    with (out/'human_textured.obj').open('w',encoding='utf-8') as f:
        f.write('mtllib human.mtl\nusemtl HumanBaked\n')
        for v in mesh['vertices_canonical'].numpy():f.write('v '+' '.join(f'{a:.9g}' for a in v)+'\n')
        for t in mesh['official_uv_vertices'].numpy():f.write('vt '+' '.join(f'{a:.9g}' for a in t)+'\n')
        for tri,uvtri in zip(faces,ft):f.write('f '+' '.join(f'{int(v)+1}/{int(t)+1}' for v,t in zip(tri,uvtri))+'\n')
    report={'success':True,'seconds':time.time()-start,'source_mesh_sha256':hashlib.sha256((source/'mesh.pt').read_bytes()).hexdigest(),'synthetic_views':72,'mesh_vertices':10475,'mesh_faces':20908,'texture_resolution':[size,size],'observed_uv_fraction':float(seen.mean()),'repairs':repairs,'scope':'Actual72 frozen-neural-avatar A-pose RGB/alpha views distilled to original SMPL-X official UV. Mesh self-depth and target alpha visibility checks, angle-weighted median of8 best synthetic views, explicit unobserved-part repair masks. This is model-predicted appearance, not raw real-frame texture or relightable albedo. Canonical10475-vertex OBJ/MTL/PNG is a static asset only; skeleton animation/shape-key Blender integration and round-trip export remain unverified. Licensed derived mesh tensors/OBJ stay local.'}
    (out/'report.json').write_text(json.dumps(report,indent=2));print('canonical human official-UV texture baked',flush=True)
