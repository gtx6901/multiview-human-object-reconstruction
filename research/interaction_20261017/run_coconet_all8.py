"""Schedule real single-view temporal model on all8 without changing n_view."""
import os
from pathlib import Path
import sys,subprocess,json,time
ROOT=Path(__file__).resolve().parent;start=time.time();meta=json.loads((ROOT/'helmet_contiguous180_211_v1/manifest.json').read_text())
out=ROOT/'coconet_all8_180_211_v1';out.mkdir(exist_ok=False);records=[]
for view in meta['views']:
 for offset in [0,16]:
    serial=view['serial'];name=f'coconet_all8_180_211_v1/{serial}_{offset:02d}';log=out/f'{serial}_{offset:02d}.log'
    with log.open('w') as f:result=subprocess.run([sys.executable,str(ROOT/'run_coconet_project_probe.py'),name,serial,str(offset)],stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,log
    report=json.loads((ROOT/name/'report.json').read_text());assert report['success'];records.append({'serial':serial,'offset':offset,'output':name,'frames':report['frames']})
    print(serial,offset,'actual forward complete',flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'views':8,'frames_per_view':32,'windows':records,'seconds':time.time()-start,'scope':'All8 real project views passed separately through original n_view=1 temporal model in two 16-frame windows. This is evidence of 256 real forward samples, not a jointly multi-camera CoCoNet architecture or fused/validated motion. Custom object initialization and vertex-color rendering as described in per-window reports.'},indent=2))
