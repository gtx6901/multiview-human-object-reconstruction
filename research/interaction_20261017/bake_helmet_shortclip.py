"""Observed vertex colors for the actual tracked candidate, no hallucinated paint."""
import os
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,trimesh
from PIL import Image
from mesh_renderer import MeshRenderer
p=argparse.ArgumentParser();p.add_argument('out',nargs='?',default='helmet_colors180_211_v1');p.add_argument('motion',nargs='?',default='helmet_pose180_211_v1');p.add_argument('--data',default='helmet_contiguous180_211_v1');p.add_argument('--objects',default='helmet_track180_211_v2');p.add_argument('--frame',type=int,default=180);options=p.parse_args()
out=ROOT/options.out;out.mkdir(exist_ok=False);motion=torch.load(ROOT/options.motion/'sequence.pt',weights_only=True);ti=motion['frames'].tolist().index(options.frame)
base=motion['vertices_local'].cuda();faces=motion['faces'].cuda();world=base@motion['R'][ti].cuda().T*motion['logscale'].cuda().exp()+motion['translation_world'][ti].cuda()
mesh=trimesh.Trimesh(world.cpu().numpy(),faces.cpu().numpy(),process=False);normal=torch.tensor(np.array(mesh.vertex_normals),device='cuda',dtype=torch.float32)
meta=json.loads((ROOT/options.data/'manifest.json').read_text());records=json.loads((ROOT/options.objects/'report.json').read_text())['records'];renderer=MeshRenderer()
best=torch.zeros(len(base),device='cuda');colors=torch.full_like(base,.8);source=torch.full((len(base),),-1,device='cuda',dtype=torch.long);counts=[]
with torch.inference_mode():
 for i,v in enumerate(meta['views']):
    if v['serial']=='FCM21121414':continue
    record=next(p for p in records if p['serial']==v['serial'] and p['frame']==options.frame)
    if record['mask'] is None:continue
    K=torch.tensor(v['K'],device='cuda');W=torch.tensor(v['W'],device='cuda');r=renderer(world,faces,K,W,1024,768)
    pc=world@W[:3,:3].T+W[:3,3];uv=pc@K.T;uv=uv[:,:2]/uv[:,2:];x=uv[:,0].round().long();y=uv[:,1].round().long();inside=(x>=0)&(x<1024)&(y>=0)&(y<768)&(pc[:,2]>.01);x=x.clamp(0,1023);y=y.clamp(0,767)
    mask=torch.tensor(np.asarray(Image.open(ROOT/options.objects/record['mask']))>127,device='cuda')
    valid=inside&mask[y,x]&(r['hard_alpha'][y,x]>.5)&((r['depth'][y,x]-pc[:,2]).abs()<.003)
    C=-W[:3,:3].T@W[:3,3];direction=torch.nn.functional.normalize(C-world,dim=1);weight=(normal*direction).sum(1).abs().square()*valid
    take=weight>best;rgb=torch.tensor(np.asarray(Image.open(ROOT/options.data/v['serial']/f'{options.frame:04d}.png').convert('RGB')).copy(),device='cuda').float()/255
    colors[take]=rgb[y[take],x[take]];source[take]=i;best[take]=weight[take];counts.append({'serial':v['serial'],'selected':int(take.sum())})
torch.save({'vertices_local':base.cpu(),'faces':faces.cpu(),'colors':colors.cpu(),'observed_vertex':best.cpu()>0,'source_view':source.cpu()},out/'mesh.pt')
(out/'report.json').write_text(json.dumps({'success':True,'frame':options.frame,'observed_vertex_fraction':float((best>0).float().mean()),'counts':counts,'scope':'Actual source RGB, self-depth and observed object-positive visibility. 414 excluded from color sampling for segmentation issues. Missing observations excluded. Abs normal incidence supports uncertain triangle winding, not unseen texture reconstruction. Unobserved vertices neutral gray; vertex fraction is not area coverage. Candidate geometry and color alignment remain unvalidated.'},indent=2))
