"""Independent LHM fusion/decoding excluding a held-out camera's features."""
import os
from pathlib import Path
import sys,runpy,json,time,hashlib,random,dataclasses
ROOT=Path(__file__).resolve().parent;LHM=ROOT.parent/'lhmpp_20261016'
sys.path[:0]=[str(LHM),str(ROOT/'deps')]
name='heldout717_appearance_v1';sys.argv=[str(LHM/'load_model_probe.py'),'--name',name]
scope=runpy.run_path(str(LHM/'load_model_probe.py'));model=scope['model'];out=scope['out']
import torch,numpy as np
from chunk_attention import install,validate
torch.set_num_threads(4);torch._dynamo.config.disable=True;torch.manual_seed(0);np.random.seed(0);random.seed(0)
start=time.time();validation=validate();install()
meta=json.loads((LHM/'inputs/fcs_frame0150_all8/manifest.json').read_text());views=meta['views'];heldout='21305717';ids=[i for i,v in enumerate(views) if v['serial']!=heldout];assert len(ids)==7
source=LHM/'runs/encode_all8_v2';inputs=torch.load(source/'inputs.pt',weights_only=True)[ids];features=torch.load(source/'image_features.pt',weights_only=True)[ids].cuda()
# encode_probe.py evaluates each camera independently. Neither joint latent nor
# decoded Gaussian attributes from the eight-view run are loaded in this branch.
model.forward_encode_image=lambda image:features
model.transformer.to('cuda');model.motion_embed_mlp.to('cuda');model.renderer.smplx_model.to('cuda')
def cpu(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all();return value.cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
    return value
with torch.inference_mode():
    query,params=model.renderer.get_query_points(None,{'betas':torch.zeros(1,10,device='cuda')},torch.device('cuda'))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model.forward_latent_points(inputs[None].cuda(),None,query_points=query,ref_imgs_bool=torch.ones(1,7,dtype=torch.bool,device='cuda'))
    torch.save({'result':cpu(result),'query':cpu(query),'smplx':cpu(params)},out/'latent.pt');print('independent seven-camera latent complete',flush=True)
    model.transformer.cpu();model.motion_embed_mlp.cpu();torch.cuda.empty_cache();model.renderer.to('cuda');model.renderer.hyper_step(10000000)
    with torch.autocast('cuda',dtype=torch.bfloat16):attrs,query,params=model.renderer.forward_gs(result[0],query,params)
    serial=[cpu({f.name:getattr(attr,f.name) for f in dataclasses.fields(attr)}) for attr in attrs]
    torch.save({'attrs':serial,'query':cpu(query),'smplx':cpu(params)},out/'gaussian_attributes.pt')
report={'success':True,'heldout_camera':heldout,'appearance_training_cameras':[views[i]['serial'] for i in ids],'view_count':7,'seconds':time.time()-start,'attention_validation':validation,'source_encoder_script_sha256':hashlib.sha256((LHM/'encode_probe.py').read_bytes()).hexdigest(),'source_independent_encoder_features_sha256':hashlib.sha256((source/'image_features.pt').read_bytes()).hexdigest(),'scope':'New seven-view joint latent and Gaussian decode from per-camera independent frozen encoder features. Held-out tensor entries excluded before fusion; no all8 latent/Gaussian attributes loaded. Fixed pretrained weights and zero beta. Appearance branch only: body/object seven-view fitting, held-out render/metrics and baseline comparison remain incomplete. Existing all8 body/object initialization must not be used as strict held-out evidence.'}
(out/'heldout_appearance_report.json').write_text(json.dumps(report,indent=2));print('seven-view appearance decoded',flush=True)
