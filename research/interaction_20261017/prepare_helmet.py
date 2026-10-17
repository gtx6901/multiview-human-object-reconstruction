"""Freeze calibrated eight-view helmet observations without changing sources."""
import os
from pathlib import Path
import hashlib, json, argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='helmet_all8_v2');parser.add_argument('--range',nargs=2,type=int,dest='frame_range');options=parser.parse_args()
    out = ROOT / options.out
    out.mkdir(parents=True, exist_ok=False)
    static = WORK / 'research/rebuild_20261014/datasets/fcs_frame0150'
    meta = json.loads((static / 'manifest.json').read_text())
    geom = json.loads((static / 'initialization/geometry.json').read_text())
    calibration = WORK / 'dataAndVedio/260828_ex.txt'
    # Require the exact source calibration used by the existing avatar alignment.
    assert sha(calibration) == meta['calibration_sha256']
    sequence = 'seuvcl_202608281121_006'
    files = list((WORK / 'dataAndVedio/vedio').glob(f'*/fcs_2026828am/{sequence}/*.mp4'))
    videos = {p.stem: p for p in files}
    assert len(videos) == len(meta['views']) == 8
    samples = list(range(*options.frame_range)) if options.frame_range else list(range(0, 300, 30)) + [299]
    assert samples and min(samples)>=0 and max(samples)<300
    records = []
    for view in meta['views']:
        serial = view['serial']; path = videos[serial]; cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert n == 300 and (w, h) == (view['width'], view['height'])
        k = np.array(view['K_native']); dist = np.array(view['distortion_opencv5'])
        ku = np.array(view['K_undistorted'])
        mx, my = cv2.initUndistortRectifyMap(k, dist, None, ku, (w, h), cv2.CV_32FC1)
        valid = cv2.remap(np.full((h,w),255,np.uint8),mx,my,cv2.INTER_NEAREST)
        viewdir = out / serial; viewdir.mkdir()
        width = 1024; height = round(h*width/w)
        affine = np.array([[width/w,0,.5*width/w-.5],[0,height/h,.5*height/h-.5],[0,0,1.]])
        Image.fromarray(cv2.resize(valid,(width,height),interpolation=cv2.INTER_NEAREST)).save(viewdir/'valid.png')
        timeline = []
        while True:
            ok, bgr = cap.read()
            if not ok: break
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)
            timeline.append({'frame':idx,'media_pts_ms':pts})
            if idx in samples:
                rgb = cv2.remap(bgr,mx,my,cv2.INTER_LINEAR)
                rgb = cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
                Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)).save(viewdir/f'{idx:04d}.png')
        cap.release(); assert len(timeline)==n
        records.append({'id':view['id'],'serial':serial,'source':str(path),'source_sha256':sha(path),
                        'frames':n,'fps_reported':fps,'timeline':timeline,'K':(affine@ku).tolist(),
                        'K_native':k.tolist(),'distortion_opencv5':dist.tolist(),
                        'K_undistorted_native':ku.tolist(),'native_to_output':affine.tolist(),
                        'W':geom['W_unit'][view['id']],'width':width,'height':height})
        print(serial,'prepared',flush=True)
    result={'sequence':sequence,'views':records,'sample_frames':samples,
            'calibration':str(calibration),'calibration_sha256':sha(calibration),
            'coordinate_convention':geom.get('convention'),'world_normalization':{
                k:geom[k] for k in ['center_raw','body_height_raw_units']},
            'synchronization':'Same-index provisional correspondence only; PTS equality does not certify hardware synchronization.',
            'scope':'All eight views retained. Original calibration, inherited static body-height world units, not measured metres. No object masks or CARI4D output yet.'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(len(samples)*256,8*212),'white');draw=ImageDraw.Draw(sheet)
    for row,v in enumerate(records):
        for col,f in enumerate(samples):
            im=Image.open(out/v['serial']/f'{f:04d}.png');im.thumbnail((256,192))
            sheet.paste(im,(col*256,row*212+20));draw.text((col*256+2,row*212+2),f"{v['serial']} f{f}",fill='black')
    sheet.save(out/'all8_timeline.jpg')
    print(out,flush=True)

if __name__=='__main__':main()
