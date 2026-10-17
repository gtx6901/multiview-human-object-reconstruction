"""Actual full all8 continuous depth predictions, preserving float output and known K."""
import os
from pathlib import Path
import sys,json,time,hashlib,traceback
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps'),str(ROOT/'vendor/UniDepth')]
sys.path.append(os.environ.get("HOI_SHARED_SITE", ""))
from unidepth_runtime import configure
configure()
import torch,numpy as np
from PIL import Image
from safetensors.torch import load_file
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
out=ROOT/'unidepth_contiguous180_211_v2';out.mkdir(exist_ok=False);start=time.time();report={'success':False,'records':[],'resolution_level':9,'scope':'Actual official UniDepthV2-L predictions for all8 real continuous source images using provided K rays. Returned intrinsics are the independent camera-head prediction, not the supplied K; geometric unprojection must retain original calibrated K. Monocular learned depth, not independently measured metric truth. No ICP, CoCoNet or final scene yet.'}
try:
 weights=ROOT/'weights/unidepth-v2-vitl14';config=json.loads((weights/'config.json').read_text());model=UniDepthV2(config)
 state=load_file(str(weights/'model.safetensors'),device='cpu');loaded=model.load_state_dict(state,strict=True);del state
 model.eval().cuda();model.resolution_level=9
 report.update(strict=True,missing=loaded.missing_keys,unexpected=loaded.unexpected_keys)
 data=ROOT/'helmet_contiguous180_211_v1';meta=json.loads((data/'manifest.json').read_text())
 people=ROOT/'people_contiguous180_211_v1';objects=ROOT/'helmet_track180_211_v2'
 human={(p['serial'],p['frame']):p for p in json.loads((people/'report.json').read_text())['records']};helmet={(p['serial'],p['frame']):p for p in json.loads((objects/'report.json').read_text())['records']}
 with torch.inference_mode():
  for vi,v in enumerate(meta['views']):
   serial=v['serial'];view_out=out/serial;view_out.mkdir();valid=np.asarray(Image.open(data/serial/'valid.png'))>127
   for f in meta['sample_frames']:
    tick=time.time();source=data/serial/f'{f:04d}.png';rgb=np.asarray(Image.open(source).convert('RGB')).copy();K=torch.tensor(v['K'],device='cuda',dtype=torch.float32)[None]
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model.infer(torch.from_numpy(rgb).permute(2,0,1).cuda(),Pinhole(K=K))
    depth=pred['depth'][0,0].float().cpu().numpy();confidence=pred['confidence'][0,0].float().cpu().numpy();actual_K=pred['intrinsics'][0].float().cpu().numpy()
    assert depth.shape==(768,1024) and np.isfinite(depth).all() and (depth>0).all() and np.isfinite(confidence).all()
    K_difference=float(np.max(np.abs(actual_K-np.array(v['K']))))
    # Upstream Decoder.run_camera uses supplied rays_gt for depth conditioning,
    # but still RETURNS its independently predicted intrinsics_matrix. Retain
    # both explicitly; never replace the calibrated K with that prediction.
    np.savez_compressed(view_out/f'{f:04d}.npz',depth=depth,confidence=confidence.astype('float16'),intrinsics_calibrated=np.array(v['K']),intrinsics_predicted=actual_K,valid=valid)
    xyz=pred['points'][0].float().cpu().numpy();yy,xx=np.indices(depth.shape);known_K=np.array(v['K'])
    u=known_K[0,0]*xyz[0]/xyz[2]+known_K[0,2];w=known_K[1,1]*xyz[1]/xyz[2]+known_K[1,2]
    ray_error=np.sqrt((u-xx)**2+(w-yy)**2)
    mm=np.rint(depth*1000);clipped=int(((mm<1)|(mm>65535)).sum());Image.fromarray(mm.clip(1,65535).astype('uint16')).save(view_out/f'{f:04d}_mm.png')
    stats={}
    for name,base,records in [('human',people,human),('helmet',objects,helmet)]:
     rec=records[(serial,f)]
     if rec['mask'] is not None:
      mask=(np.asarray(Image.open(base/rec['mask']))>127)&valid
      stats[name]={'pixels':int(mask.sum()),'depth_quantiles_m':np.quantile(depth[mask],[.1,.5,.9]).tolist()}
    record={'serial':serial,'frame':f,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seconds':time.time()-tick,'max_predicted_vs_calibrated_K_difference':K_difference,'sdk_ray_resampling_error_mean_px':float(ray_error[valid].mean()),'sdk_ray_resampling_error_p99_px':float(np.quantile(ray_error[valid],.99)),'depth_quantiles_m':np.quantile(depth[valid],[.1,.5,.9]).tolist(),'foreground_stats':stats,'uint16_clipped_pixels':clipped};report['records'].append(record)
   print(serial,'32 actual depth predictions',flush=True)
 report.update(success=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception:report['error']=traceback.format_exc()
report['seconds']=time.time()-start;(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}|{'predictions':len(report['records'])}),flush=True)
if not report['success']:raise SystemExit(1)
