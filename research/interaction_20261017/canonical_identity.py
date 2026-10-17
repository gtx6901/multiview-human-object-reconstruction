"""Stable tensor identity for semantic anchors tied to a rendered avatar."""
import os
import hashlib,json,torch
def semantic_identity(bundle):
 h=hashlib.sha256()
 def add(name,value):
  h.update(name.encode('utf-8')+b'\0')
  if isinstance(value,dict):
   for key in sorted(value):add(str(key),value[key])
  elif isinstance(value,torch.Tensor):
   t=value.detach().cpu().contiguous();h.update(str(t.dtype).encode()+str(tuple(t.shape)).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  else:h.update(json.dumps(value,sort_keys=True).encode('utf-8'))
 for key in ['rig','features','decoder','decoder_config','background_embedding']:add(key,bundle[key])
 for key in ['betas','scale']:add(key,bundle['poses'][key])
 return h.hexdigest()
