"""Strict checkpoint loading with meta allocation only for large neural modules."""
from pathlib import Path
import os,json,traceback,time,argparse
from runtime_imports import configure,ROOT
configure();os.environ['SPCONV_DISABLE_JIT']='1';os.chdir(ROOT)
import torch
import inspect,numpy as np
# Old FLAME pickle depends on chumpy's pre-Python3.11 / pre-NumPy1.24 names.
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if name not in np.__dict__:setattr(np,name,value)
from accelerate import init_empty_weights,Accelerator
from safetensors import safe_open
from core.models import modeling_humana4o_lrm as upstream
p=argparse.ArgumentParser();p.add_argument('--name',default='load_probe_v1');args=p.parse_args()
out=ROOT/'runs'/args.name;out.mkdir(parents=True,exist_ok=False)
config=json.loads((ROOT/'checkpoint_config.json').read_text())
config['transformer_decoder']['point_backbone']['enable_flash']=False
def defer(fn):
    def wrapped(*args,**kwargs):
        with init_empty_weights(include_buffers=False):return fn(*args,**kwargs)
    return wrapped
cls=upstream.ModelHumanA4OLRM
cls.build_transformer=defer(cls.build_transformer)
cls._build_neural_renderer=defer(cls._build_neural_renderer)
# ArcFace loads its separate verified checkpoint in its constructor; keep it real.
from core.models.encoders.dinov2_wrapper import Dinov2Wrapper
original_dino=Dinov2Wrapper._build_dinov2
def dino_from_checkpoint(model_name,modulation_dim=None,pretrained=True):
    with init_empty_weights(include_buffers=False):
        return original_dino(model_name,modulation_dim=modulation_dim,pretrained=False)
Dinov2Wrapper._build_dinov2=staticmethod(dino_from_checkpoint)
Accelerator()
start=time.time()
try:
    model=cls(**config);model.eval();print('constructed',flush=True)
    with safe_open(ROOT/'weights/LHMPP-700M/model.safetensors',framework='pt',device='cpu') as f:
        state={k:f.get_tensor(k) for k in f.keys()}
    # Upstream PatchDPT4DecoderOnly.predict_rgbs never calls this inherited block.
    # A parameterless fail-fast Module prevents accidental execution, rather than
    # silently keeping random weights or a no-op transformer.
    assert type(model.neural_renderer).__name__=='PatchDPT4DecoderOnly'
    assert not any(k.startswith('neural_renderer.transformer_block.') for k in state)
    model.neural_renderer.transformer_block=torch.nn.Module()
    assert not any(k.startswith('id_face_net.') for k in state)
    state.update({'id_face_net.'+k:v for k,v in model.id_face_net.state_dict().items()})
    result=model.load_state_dict(state,strict=True,assign=True)
    meta=[n for n,p in model.named_parameters() if p.is_meta]
    assert not meta,meta
    report={'success':True,'strict':True,'seconds':time.time()-start,'parameter_count':sum(p.numel() for p in model.parameters()),'missing':result.missing_keys,'unexpected':result.unexpected_keys,'attention':'upstream non-flash branch; needs numerical/forward validation','scope':'load only, no reconstruction performed'}
except Exception:report={'success':False,'seconds':time.time()-start,'error':traceback.format_exc()}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if not report['success']:raise SystemExit(1)
