"""Actual official NLF inference on explicit real project images and intrinsics."""
import os
from pathlib import Path
import argparse,json,time,traceback,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',default='nlf_all8_frame30_v1');p.add_argument('--frames',nargs='+',type=int,default=[30]);p.add_argument('--data',default='helmet_all8_v2');p.add_argument('--detections',default='body_detections_v2');p.add_argument('--all-frames',action='store_true');options=p.parse_args()
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,torchvision,numpy as np
from PIL import Image,ImageDraw
out=ROOT/options.out;out.mkdir(exist_ok=False);start=time.time();report={'success':False,'records':[],'source_dataset':options.data,'detections':options.detections,'scope':'Actual official pretrained NLF inference with real RGB, known undistorted K and observed person boxes. Per-view predictions, no cross-view fusion or CoCoNet output yet.'}
try:
 # TorchScript's Windows fopen cannot read this Unicode workspace path.
 # Python's binary stream preserves the official checkpoint byte-for-byte.
 with (ROOT/'weights/nlf_l_multi_0.3.2.torchscript').open('rb') as stream:
  # NLF stores canonical points in ordinary tensor dictionaries as well as
  # registered buffers. Loading all storages onto CUDA is essential: .cuda()
  # alone does not move the dictionary entries after a CPU-mapped load.
  model=torch.jit.load(stream,map_location='cuda').eval()
 print('NLF loaded',flush=True)
 meta=json.loads((ROOT/options.data/'manifest.json').read_text());det=np.load(ROOT/options.detections/'detections.npz');frames=det['frames'].tolist()
 if options.all_frames:options.frames=frames
 assert set(options.frames).issubset(frames)
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [cpu(v) for v in x]
  return x
 def shapes(x):
  if isinstance(x,torch.Tensor):return {'shape':list(x.shape),'dtype':str(x.dtype)}
  if isinstance(x,dict):return {k:shapes(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [shapes(v) for v in x]
  return str(x)
 with torch.inference_mode():
  for vi,v in enumerate(meta['views']):
   for f in options.frames:
    source=ROOT/options.data/v['serial']/f'{f:04d}.png';image=Image.open(source).convert('RGB');arr=np.asarray(image).copy()
    b=det['boxes'][vi,frames.index(f)].copy();box=torch.tensor([b[0],b[1],b[2]-b[0],b[3]-b[1],1.01],device='cuda')[None]
    tensor=torch.from_numpy(arr.transpose(2,0,1)[None]).cuda()
    tick=time.time();pred=model.detect_smpl_batched(tensor,extra_boxes=[box],detector_threshold=1.0,suppress_implausible_poses=False,intrinsic_matrix=torch.tensor(v['K'],device='cuda',dtype=torch.float32)[None])
    result=cpu(pred);torch.save(result,out/f"{v['serial']}_{f:04d}.pt")
    vertices=pred['vertices3d'][0];assert vertices.shape[0]==1 and torch.isfinite(vertices).all()
    xyz=vertices[0].cpu().numpy();K=np.array(v['K']);points=xyz@K.T;uv=points[:,:2]/points[:,2:];draw=ImageDraw.Draw(image)
    for x,y in uv[::5]:
     if 0<=x<1024 and 0<=y<768:draw.point((float(x),float(y)),fill='cyan')
    image.save(out/f"{v['serial']}_{f:04d}_projection.jpg")
    row={'serial':v['serial'],'frame':f,'seconds':time.time()-tick,'vertices':list(vertices.shape),'output':shapes(result),'person_box_xywh':box.cpu().tolist()};report['records'].append(row)
    print(v['serial'],f,row['seconds'],row['vertices'],flush=True)
 report['success']=True;report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
except Exception:report['error']=traceback.format_exc()
report['seconds']=time.time()-start;(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}|{'predictions':len(report['records'])}),flush=True)
if not report['success']:raise SystemExit(1)
