"""A small canonical asset set and an explicitly separate per-frame state table."""
import os
from pathlib import Path
import json,hashlib
import torch

def _asset(root,relative):
    path=(root/relative).resolve();assert path.is_relative_to(root.resolve()),'Scene asset escapes its own directory'
    return path

def load_shared_scene(root):
    root=Path(root);manifest=json.loads((root/'scene.json').read_text());assert manifest['format']=='shared-hoi-scene-v1'
    for row in manifest['assets']:
        path=_asset(root,row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    human=torch.load(_asset(root,'human_canonical.pt'),weights_only=True);mesh=torch.load(_asset(root,'helmet_canonical.pt'),weights_only=True);motion=torch.load(_asset(root,'motion.pt'),weights_only=True)
    assert 'poses' not in human and 'vertices_local' not in motion['helmet']
    poses={**motion['human'],'frames':motion['frames'],'betas':human['betas'],'scale':human['scale']}
    avatar={**human,'poses':poses};object_motion={**motion['helmet'],'frames':motion['frames'],'vertices_local':mesh['vertices_local'],'faces':mesh['faces'],'logscale':mesh['logscale']}
    return {'avatar':avatar,'mesh':mesh,'object_motion':object_motion,'cameras':json.loads(_asset(root,'cameras.json').read_text()),'manifest':manifest}
