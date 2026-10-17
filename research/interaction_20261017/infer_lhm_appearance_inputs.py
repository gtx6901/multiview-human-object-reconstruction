"""Fresh LHM encoding/fusion for an explicit actor input set; no identity reuse."""
import os
from pathlib import Path
import sys,argparse,json,time,runpy,dataclasses,hashlib,random
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016';sys.path.insert(0,str(LHM))
from runtime_imports import configure
configure();sys.path.insert(0,str(ROOT/'deps'))
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--run',required=True);o=p.parse_args();source=ROOT/o.inputs;inputreport=json.loads((source/'report.json').read_text());assert inputreport['success']
inputs=torch.load(source/'inputs.pt',weights_only=True);N=len(inputs);assert 1<=N<=16 and inputs.shape[1:]==(3,840,504)
sys.argv=[str(LHM/'load_model_probe.py'),'--name',o.run];scope=runpy.run_path(str(LHM/'load_model_probe.py'));model=scope['model'];out=scope['out'];start=time.time();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch._dynamo.config.disable=True
from chunk_attention import install,validate
attention=validate();install()
def cpu(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();return x.cpu()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [cpu(v) for v in x]
 return x
model.encoder.cuda();features=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for i in range(N):
  f=model.forward_encode_image(inputs[i:i+1].cuda());assert torch.isfinite(f).all();features.append(f.cpu());print('fresh actor image encoded',i,flush=True)
features=torch.cat(features);torch.save(features,out/'image_features.pt');model.encoder.cpu();torch.cuda.empty_cache();features=features.cuda();model.forward_encode_image=lambda image:features
model.transformer.cuda();model.motion_embed_mlp.cuda();model.renderer.smplx_model.cuda()
with torch.inference_mode():
 query,params=model.renderer.get_query_points(None,{'betas':torch.zeros(1,10,device='cuda')},torch.device('cuda'))
 with torch.autocast('cuda',dtype=torch.bfloat16):result=model.forward_latent_points(inputs[None].cuda(),None,query_points=query,ref_imgs_bool=torch.ones(1,N,dtype=torch.bool,device='cuda'))
 torch.save({'result':cpu(result),'query':cpu(query),'smplx':cpu(params)},out/'latent.pt');print('fresh actor latent fused',flush=True)
 model.transformer.cpu();model.motion_embed_mlp.cpu();torch.cuda.empty_cache();model.renderer.cuda();model.renderer.hyper_step(10000000)
 with torch.autocast('cuda',dtype=torch.bfloat16):attrs,query,params=model.renderer.forward_gs(result[0],query,params)
 torch.save({'attrs':[cpu({f.name:getattr(a,f.name) for f in dataclasses.fields(a)}) for a in attrs],'query':cpu(query),'smplx':cpu(params)},out/'gaussian_attributes.pt')
(out/'appearance_report.json').write_text(json.dumps({'success':True,'input_images':N,'input_source':o.inputs,'source_inputs_sha256':hashlib.sha256((source/'inputs.pt').read_bytes()).hexdigest(),'attention_validation':attention,'seconds':time.time()-start,'scope':'All supplied images independently encoded from this actor, then jointly fused/decoded with fixed pretrained LHM weights; no helmet-actor features reused, no camera calibration or body motion inferred. Beta0 is an initialization prior. New Gaussian asset remains unadopted until canonical export/render and same-actor comparison. Not full-sequence fitting or recovered novel-view ground truth; licensed tensors remain private.'},indent=2));print('independent actor appearance candidate decoded',flush=True)
