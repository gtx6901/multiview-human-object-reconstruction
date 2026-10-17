"""Five observed frame30 silhouettes produce a bounded closed geometry baseline."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')];sys.path.append(os.environ.get("HOI_SHARED_SITE", ""))
import torch,numpy as np,cv2,trimesh,pymeshlab
from PIL import Image
from scipy.ndimage import label,binary_fill_holes
from skimage.measure import marching_cubes
out=ROOT/'helmet_hull_frame30_v1';out.mkdir(exist_ok=False);start=time.time()
old=torch.load(ROOT/'helmet_frame30_fit_v2/object_pose.pt',weights_only=True);R=old['R'];s=old['logscale'].exp();t=old['translation'];base=trimesh.load(ROOT/'helmet_frame30_fit_v2/local_candidate.obj',process=False)
world=torch.tensor(np.array(base.vertices),dtype=torch.float32)@R.T*s+t;center=(world.min(0).values+world.max(0).values)/2;extent=(world.max(0).values-world.min(0).values)*1.4
n=160;origin=center-extent/2;spacing=extent/(n-1);grid=torch.stack(torch.meshgrid(*[torch.arange(n,device='cuda') for _ in range(3)],indexing='ij'),-1).reshape(-1,3);points=origin.cuda()+grid*spacing.cuda()
positive=torch.zeros(len(points),device='cuda',dtype=torch.int16);negative=positive.clone();views=[];meta=json.loads((ROOT/'helmet_all8_v2/manifest.json').read_text())
for v in meta['views']:
    serial=v['serial'];path=ROOT/'helmet_masks_frame30_v2'/f'{serial}_mask.png'
    if not path.exists():continue
    mask=np.asarray(Image.open(path))>127;person=np.asarray(Image.open(ROOT/'people_frame30_v1'/f'{serial}.png'))>127;person&=~mask;valid=np.asarray(Image.open(ROOT/'helmet_all8_v2'/serial/'valid.png'))>127
    dilated=cv2.dilate(mask.astype('uint8'),np.ones((7,7),np.uint8))>0
    W=torch.tensor(v['W'],device='cuda');K=torch.tensor(v['K'],device='cuda');pc=points@W[:3,:3].T+W[:3,3];uv=pc@K.T;xy=(uv[:,:2]/uv[:,2:]).round().long();inside=(xy[:,0]>=0)&(xy[:,0]<1024)&(xy[:,1]>=0)&(xy[:,1]<768)&(pc[:,2]>.01);x=xy[:,0].clamp(0,1023);y=xy[:,1].clamp(0,767)
    good=inside&torch.tensor(valid,device='cuda')[y,x];pm=torch.tensor(dilated,device='cuda')[y,x];ph=torch.tensor(person,device='cuda')[y,x];positive+=(good&pm).short();negative+=(good&~pm&~ph).short();views.append(serial)
occupancy=((positive>=3)&(negative<=0)).reshape(n,n,n).cpu().numpy();occupancy[[0,-1],:,:]=False;occupancy[:,[0,-1],:]=False;occupancy[:,:,[0,-1]]=False
labels,count=label(occupancy);sizes=np.bincount(labels.ravel());sizes[0]=0;occupancy=binary_fill_holes(labels==sizes.argmax());vertices,faces,_,_=marching_cubes(occupancy.astype('float32'),.5,spacing=spacing.numpy());vertices+=origin.numpy();local=(torch.tensor(vertices.copy(),dtype=torch.float32)-t)@R/s
ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(local.numpy().astype('float64'),faces.astype('int32')));ms.apply_filter('meshing_decimation_quadric_edge_collapse',targetfacenum=16000,preserveboundary=True,preservetopology=True);m=ms.current_mesh();mesh=trimesh.Trimesh(m.vertex_matrix(),m.face_matrix(),process=False);mesh.export(out/'local_candidate.obj')
torch.save({'vertices_local':torch.tensor(np.array(mesh.vertices),dtype=torch.float32),'faces':torch.tensor(np.array(mesh.faces),dtype=torch.int32),'R':R,'translation_world':t,'logscale':old['logscale']},out/'mesh_and_anchor.pt')
(out/'report.json').write_text(json.dumps({'success':True,'frame':30,'observed_views':views,'unknown_views':[v['serial'] for v in meta['views'] if v['serial'] not in views],'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'grid':n,'mask_dilation_pixels':3,'required_positive_views':3,'allowed_known_background_contradictions':0,'seconds':time.time()-start,'scope':'Closed bounded silhouette occupancy at held frame30, not learned CARI shape. Original candidate supplies search bounds/local coordinates; three unobserved views and human-occluded pixels unknown. Fills internal cavity and lacks physical helmet thickness/straps. Predicted masks and calibration are not geometric truth.'},indent=2));print('closed hull complete',flush=True)
