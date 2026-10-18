"""Check response convention and correct a stale explanatory metadata suffix."""
import os
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
out=ROOT/'transferred_camera_response_check_v1';out.mkdir(exist_ok=False)
folder=ROOT/'helmet_profile_shared_response_texture_v1';p=folder/'report.json';r=json.loads(p.read_text());c=r['color_correction'];h=json.loads((ROOT/c['human_response_source']/'report.json').read_text());views=json.loads((ROOT/'helmet_native000_299_v1/manifest.json').read_text())['views'];expected=1/np.array([h['fixed_camera_rgb_gains'][v['serial']] for v in views]);actual=np.array(c['gain_linear_rgb']);error=float(abs(expected-actual).max());assert error<1e-6 and np.array_equal(c['bias_linear_rgb'],np.zeros((8,3))) and np.array_equal(actual[1],np.ones(3))
old_suffix=' Exposure gain is shared acrossRGB with bounded±2EV, chromatic correction is limited to exp(±.15), and black-level bias is shared acrossRGB within±.01 assumed-linear units. This reduces artificial chroma in dark lining; it does not recover missing source detail or guarantee registration.'
changed=old_suffix in c['scope'];c['scope']=c['scope'].replace(old_suffix,'')
if changed:
 r['metadata_correction']='Removed helmet-fitted-response explanatory suffix from transferred-human response. Active gains/bias, texture, geometry and samples unchanged; reciprocal response verified independently.'
 p.write_text(json.dumps(r,indent=2));(folder/'camera_color_correction.json').write_text(json.dumps(c,indent=2))
(out/'report.json').write_text(json.dumps({'success':True,'gain_inverse_max_error':error,'zero_bias':True,'reference_identity':True,'explanatory_metadata_corrected':changed,'scope':'Checks actual saved gains equal inverse human observed/model convention, zero bias and identity reference. Only stale descriptive metadata corrected; no texture/sample/geometry change. Does not prove physical response accuracy.'},indent=2));print('transferred response convention verified',flush=True)
