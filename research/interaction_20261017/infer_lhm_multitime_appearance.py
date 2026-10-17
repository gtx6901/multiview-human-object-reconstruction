"""Experimental16-image multi-pose LHM appearance; accepted avatar untouched."""
import os
from pathlib import Path
import sys,runpy,json,time,random,dataclasses,hashlib
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path[:0]=[str(LHM),str(ROOT/'deps')]
source=ROOT/'lhm_multitime_inputs16_v2';assert json.loads((source/'report.json').read_text())['success']
sys.argv=[str(LHM/'load_model_probe.py'),'--name','multitime16_appearance_v1'];scope=runpy.run_path(str(LHM/'load_model_probe.py'));model=scope['model'];out=scope['out']
import torch,numpy as np
from chunk_attention import install,validate
torch.set_num_threads(4);torch._dynamo.config.disable=True;torch.manual_seed(0);np.random.seed(0);random.seed(0);start=time.time();validation=validate();install()
inputs=torch.load(source/'inputs.pt',weights_only=True);original=torch.load(LHM/'runs/encode_all8_v2/inputs.pt',weights_only=True);assert inputs.shape[0]==16 and torch.equal(inputs[:8],original)
cached=torch.load(LHM/'runs/encode_all8_v2/image_features.pt',weights_only=True);features=[cached];model.encoder.cuda()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for i in range(8,16):
  f=model.forward_encode_image(inputs[i:i+1].cuda());assert torch.isfinite(f).all();features.append(f.cpu());print('encoded additional image',i,flush=True)
model.encoder.cpu();torch.cuda.empty_cache();features=torch.cat(features);torch.save(features,out/'image_features.pt');features=features.cuda();model.forward_encode_image=lambda image:features
model.transformer.cuda();model.motion_embed_mlp.cuda();model.renderer.smplx_model.cuda()
def cpu(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();return x.cpu()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [cpu(v) for v in x]
 return x
with torch.inference_mode():
 query,params=model.renderer.get_query_points(None,{'betas':torch.zeros(1,10,device='cuda')},torch.device('cuda'))
 with torch.autocast('cuda',dtype=torch.bfloat16):result=model.forward_latent_points(inputs[None].cuda(),None,query_points=query,ref_imgs_bool=torch.ones(1,16,dtype=torch.bool,device='cuda'))
 torch.save({'result':cpu(result),'query':cpu(query),'smplx':cpu(params)},out/'latent.pt');print('16-image latent complete',flush=True)
 model.transformer.cpu();model.motion_embed_mlp.cpu();torch.cuda.empty_cache();model.renderer.cuda();model.renderer.hyper_step(10000000)
 with torch.autocast('cuda',dtype=torch.bfloat16):attrs,query,params=model.renderer.forward_gs(result[0],query,params)
 torch.save({'attrs':[cpu({f.name:getattr(a,f.name) for f in dataclasses.fields(a)}) for a in attrs],'query':cpu(query),'smplx':cpu(params)},out/'gaussian_attributes.pt')
(out/'multitime_appearance_report.json').write_text(json.dumps({'success':True,'input_images':16,'static_images':8,'additional_dynamic_images':8,'inputs_sha256':hashlib.sha256((source/'inputs.pt').read_bytes()).hexdigest(),'seconds':time.time()-start,'attention_validation':validation,'scope':'New experimental LHM++ appearance from all8 static cameras plus8 multi-time held images. Original8 features reused exactly, added images independently encoded then all16 jointly fused. Same pretrained model/decoder and beta0. Neither accepted avatar nor motion overwritten. Multi-pose execution success is not image-quality improvement; export and same-camera visual comparison required before adoption.'},indent=2));print('experimental multitime appearance decoded',flush=True)
