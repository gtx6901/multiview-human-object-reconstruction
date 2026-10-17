"""Compare metadata, actual OpenCV frames and independent FFmpeg decode."""
import os
from pathlib import Path
import sys,json,subprocess,re
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'deps'))
import cv2,imageio_ffmpeg
out=ROOT/'backpack_decode_audit_v1';out.mkdir(exist_ok=False);rows=[]
sources=json.loads((ROOT/'backpack_source_review_v1/report.json').read_text())['views']
for v in sources:
    cap=cv2.VideoCapture(v['source']);pts=[];positions=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)));positions.append(float(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    cap.release()
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',v['source'],'-map','0:v:0','-an','-threads','1','-f','null','-'],capture_output=True,text=True,timeout=120)
    text=result.stderr;(out/f"{v['serial']}_ffmpeg.txt").write_text(text,encoding='utf-8');counts=re.findall(r'frame=\s*(\d+)',text);assert result.returncode==0 and counts
    row={'serial':v['serial'],'metadata_frame_count':v['frames'],'opencv_decoded_frames':len(pts),'ffmpeg_decoded_frames':int(counts[-1]),'opencv_pts_ms':pts,'opencv_positions':positions,'metadata_fps':v['fps_reported']};rows.append(row);print(v['serial'],len(pts),int(counts[-1]),flush=True)
(out/'report.json').write_text(json.dumps({'success':True,'views':rows,'scope':'Actual sequential decode counts and media PTS, independent FFmpeg null-output check. Original source unmodified. Container frame-count metadata is not accepted as actual frame coverage.'},indent=2))
