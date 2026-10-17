"""Narrow compatibility hook: defer unused legacy PointTransformer import.

No fake CUDA operator is supplied. If that encoder is actually requested,
its real upstream import runs and fails normally unless pointops is installed.
The selected checkpoint uses e2e_points / PV3, not e2e_smplx_pt.
"""
import os
from pathlib import Path
import importlib.abc,importlib.util,sys,os,json
ROOT=Path(__file__).resolve().parent
TARGET='core.models.modeling_humana4o_lrm'
OLD='from core.models.encoders.pointtransformer.point_transformer_warp import (\n    PointTransformerEncoder,\n)'
NEW='def PointTransformerEncoder(*args, **kwargs):\n    from core.models.encoders.pointtransformer.point_transformer_warp import PointTransformerEncoder as actual\n    return actual(*args, **kwargs)'
RASTER_TARGET='core.models.rendering.base_gs_render'
RASTER_OLD='from diff_gaussian_rasterization import (\n    GaussianRasterizationSettings,\n    GaussianRasterizer,\n)'
RASTER_NEW='def GaussianRasterizationSettings(*args, **kwargs):\n    from diff_gaussian_rasterization import GaussianRasterizationSettings as actual\n    return actual(*args, **kwargs)\ndef GaussianRasterizer(*args, **kwargs):\n    from diff_gaussian_rasterization import GaussianRasterizer as actual\n    return actual(*args, **kwargs)'
PATCHES={TARGET:('core/models/modeling_humana4o_lrm.py',OLD,NEW),RASTER_TARGET:('core/models/rendering/base_gs_render.py',RASTER_OLD,RASTER_NEW)}
class DeferredLegacy(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in PATCHES:return importlib.util.spec_from_file_location(fullname,ROOT/'vendor'/PATCHES[fullname][0],loader=self)
    def create_module(self,spec):return None
    def exec_module(self,module):
        relative,old,new=PATCHES[module.__name__];filename=str(ROOT/'vendor'/relative);source=Path(filename).read_text(encoding='utf-8');assert source.count(old)==1
        exec(compile(source.replace(old,new),filename,'exec'),module.__dict__)
def configure():
    assert json.loads((ROOT/'checkpoint_config.json').read_text())['latent_query_points_type']=='e2e_points'
    assert json.loads((ROOT/'checkpoint_config.json').read_text())['gs_rendering']=='featbacksplat'
    os.environ['HF_HOME']=str(ROOT/'hf_cache');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(ROOT/'deps'));sys.path.insert(0,str(ROOT/'vendor'))
    sys.meta_path.insert(0,DeferredLegacy())
if __name__=='__main__':
    configure()
    import core.models
    print('Model definition imported, not instantiated or verified for inference.')
