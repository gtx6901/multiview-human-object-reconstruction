"""Independent actual model replay from frozen project tensors."""
import os
import sys,json,time
from cari_runtime import configure,ROOT
configure()
import torch
sys.path.insert(0,os.environ["HOI_WARP_ASCII_ROOT"])
import warp as wp
wp.config.kernel_cache_dir=os.path.join(os.environ["HOI_WARP_ASCII_ROOT"], "kernel_cache")
from cari_utils_runtime import load
load();torch.set_num_threads(4)
source=ROOT/sys.argv[1];out=ROOT/sys.argv[2];out.mkdir(exist_ok=False);start=time.time()
original=torch.hub.load
def local(repo,name,*args,**kwargs):
    assert repo=='facebookresearch/dinov2';kwargs.update(pretrained=False,source='local')
    return original(str(ROOT/'vendor/dinov2'),name,*args,**kwargs)
torch.hub.load=local
checkpoint=torch.load(ROOT/'weights/step031397.pth',weights_only=False,map_location='cpu')
from learning.models import get_model
model=get_model(checkpoint['cfg']);model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;model.eval().cuda()
frozen=torch.load(source/'input_batch.pt',weights_only=True);previous=torch.load(source/'raw_predictions.pt',weights_only=True)
with torch.inference_mode():result=model(frozen['A'].cuda(),frozen['B'].cuda(),frozen['poseA_norm'].cuda(),{k:v.cuda() for k,v in frozen['batch'].items()})
checks=[]
for key,value in result.items():
    error=float((value.cpu()-previous[key]).abs().max());assert error<1e-6,(key,error)
    checks.append({'key':key,'max_abs_difference':error})
torch.save({k:x.cpu() for k,x in result.items()},out/'raw_predictions.pt')
(out/'report.json').write_text(json.dumps({'success':True,'source':str(source),'checks':checks,'seconds':time.time()-start,'scope':'Independent model process replays actual frozen input tensors. No reconstruction preprocessing or fitting repeated, no GT/quality validation.'},indent=2));print('replay passed',flush=True)
