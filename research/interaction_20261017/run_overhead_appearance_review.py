"""Single all-eight-camera appearance candidate with grounded motion held fixed."""
import os
from pathlib import Path
import sys,os,json,subprocess,time,argparse
ROOT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
p=argparse.ArgumentParser();p.add_argument('--tag',default='overhead_appearance');p.add_argument('--version',default='v1');p.add_argument('--appearance-run',default='overhead16_appearance_v1');p.add_argument('--before',help='Reuse an already successful matched baseline render');o=p.parse_args()
out=ROOT/f'{o.tag}_loop_{o.version}';out.mkdir(exist_ok=False)
avatar_out=f'avatar_{o.tag}_grounded_{o.version}';after_out=f'{o.tag}_after_review20_{o.version}';before_out=o.before or f'{o.tag}_before_review20_{o.version}';comparison_out=f'{o.tag}_comparison20_{o.version}'
frames=[0,30,42,60,90,120,135,145,155,165,175,180,190,200,210,225,240,260,280,299]
assert json.loads((ROOT.parent/'lhmpp_20261016/runs'/o.appearance_run/'multitime_appearance_report.json').read_text())['success']
rows=[];start=time.time()
steps=[('export',['export_avatar_motion.py','--out',avatar_out,'--poses','grounded_legs_all300_v1','--appearance-run',o.appearance_run])]
if o.before:
 r=json.loads((ROOT/o.before/'report.json').read_text());assert r['success'] and r['frames']==frames and r['avatar_source']=='grounded_legs_all300_v1'
for label,avatar in ([('after',avatar_out)] if o.before else [('before','grounded_legs_all300_v1'),('after',avatar_out)]):
 steps.append((label,['render_dynamic_hybrid.py',after_out if label=='after' else before_out,'NONE','helmet_lhm_depth_tight_all300_v1','helmet_texture_consistent_lhm_v1','--avatar',avatar,'--data','helmet_native000_299_v1','--cameras','lhm_portrait_framing_v1/cameras.json','--occlusion','gaussian-center','--all-input-frames','--frames',*map(str,frames)]))
steps.append(('storyboard',['make_integrated_review_storyboard.py','--out',comparison_out,'--before',before_out,'--after',after_out,'--avatar','grounded_legs_all300_v1']))
for name,args in steps:
 print('starting',name,flush=True)
 with (out/f'{name}.log').open('w') as log:r=subprocess.run([sys.executable,str(ROOT/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
 rows.append({'stage':name,'returncode':r.returncode});(out/'status.json').write_text(json.dumps(rows,indent=2))
 assert r.returncode==0,f'Inspect {name}.log; completed stages must not be repeated'
 if name=='export':
  import torch
  a=torch.load(ROOT/'grounded_legs_all300_v1/avatar.pt',weights_only=True);b=torch.load(ROOT/avatar_out/'avatar.pt',weights_only=True)
  for key in a['poses']:
   if isinstance(a['poses'][key],torch.Tensor):assert torch.equal(a['poses'][key],b['poses'][key]),key
  assert a['decoder_config']==b['decoder_config']
  assert all(torch.equal(a['decoder'][k],b['decoder'][k]) for k in a['decoder'])
  del a,b
 print('completed',name,flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'frames':frames,'stages':rows,'seconds':time.time()-start,'scope':'Static8 preserved exactly plus native person crops from160/225 across all8 cameras, jointly fused once into a shared LHM appearance. Identical saved grounded full300 pose tensors and pretrained decoder, same fixed legacy helmet asset/motion. Changed appearance includes Gaussian geometry/features/skinning. Actual20times x8 true cameras and2 novel cameras, no full300-video claim. Source-bearing input sheets and comparison panels private. Candidate requires visual review before adoption.'},indent=2))
