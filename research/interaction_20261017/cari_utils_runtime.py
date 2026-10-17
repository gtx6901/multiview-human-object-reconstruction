"""Load real upstream Utils; defer only its unused Open3D visualization import."""
import os
import sys,types
from cari_runtime import CARI
def load():
    if 'Utils' in sys.modules:return sys.modules['Utils']
    source=(CARI/'Utils.py').read_text()
    assert source.count('import open3d as o3d')==1
    assert source.count('def toOpen3dCloud(points,colors=None,normals=None):')==1
    source=source.replace('import open3d as o3d','').replace('def toOpen3dCloud(points,colors=None,normals=None):','def toOpen3dCloud(points,colors=None,normals=None):\n  import open3d as o3d')
    assert source.count('import matplotlib.pyplot as plt')==1 and source.count('plt.')==0
    source=source.replace('import matplotlib.pyplot as plt','')
    module=types.ModuleType('Utils');module.__file__=str(CARI/'Utils.py');sys.modules['Utils']=module
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module
