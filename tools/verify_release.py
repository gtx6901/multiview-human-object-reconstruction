"""Standard-library checks for this curated archive; no model inference claim."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent.parent
manifest=json.loads((ROOT/'artifact_manifest.json').read_text(encoding='utf-8'))
listed={r['path'] for r in manifest['files']}
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.relative_to(ROOT).parts and p.name!='artifact_manifest.json'}
assert actual==listed,{'missing':sorted(listed-actual),'extra':sorted(actual-listed)}
denied={'.pt','.pth','.pkl','.safetensors','.h5','.npy','.npz','.log','.zip','.whl','.dll','.pyd'}
for row in manifest['files']:
    path=ROOT/row['path'];assert path.resolve().is_relative_to(ROOT) and path.suffix not in denied
    value=path.read_bytes();assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256'],row['path']
    if path.suffix=='.py':compile(value.decode('utf-8'),str(path),'exec')
    if path.suffix in {'.py','.md','.json','.yml','.yaml','.txt'}:
        text=value.decode('utf-8');assert not re.search(r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',text)
        assert not re.search(r'[CD]:[\\/]Users[\\/]|[CD]:[\\/]Document[\\/]',text)
    if path.suffix=='.mp4':assert value[4:8]==b'ftyp'
    if path.suffix=='.jpg':assert value[:2]==bytes([255,216])
summary=json.loads((ROOT/'evidence/stage_summary.json').read_text(encoding='utf-8'))
assert summary['video_frames']==136 and summary['final_render']['independent_exact_pngs']==296
assert summary['manual_review']['stage_review_complete'] and not summary['manual_review']['quality_acceptance_claimed']
print('Verified',len(listed),'public files; archive integrity and source compilation only.')
