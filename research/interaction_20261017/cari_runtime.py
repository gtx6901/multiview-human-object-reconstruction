"""Local CARI4D compatibility and licensed-asset wiring; no replacement operators."""
import os
from pathlib import Path
import sys,os,inspect,pickle
ROOT=Path(__file__).resolve().parent;CARI=ROOT/'vendor/CARI4D'
ASSETS=ROOT/'cari_assets/official_demo/assets'
SMPLH=ROOT/'smplh_assets_v5/smplh'
def configure():
    for path in [ROOT/'deps',ROOT.parent/'lhmpp_20261016/deps',CARI]:
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    extra=os.environ.get("HOI_SHARED_SITE", "")
    if extra not in sys.path:sys.path.append(extra)
    os.environ['HF_HOME']=str(ROOT/'hf_cache')
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    import numpy as np
    for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
        if name not in np.__dict__:setattr(np,name,value)
    import lib_smpl.io as io
    import lib_smpl.smpl_module as module
    from scipy.sparse import csr_matrix
    official=io.load_smpl_model_data
    def compatible_load(path):
        data=pickle.load(open(path,'rb'),encoding='latin1')
        for key in ['bs_type','bs_style']:
            value=data.get(key)
            if isinstance(value,np.ndarray) and value.ndim==0:value=value.item()
            if isinstance(value,bytes):value=value.decode('ascii')
            if value is not None:data[key]=value
        if isinstance(data['J_regressor'],np.ndarray):
            dense=data['J_regressor'];data['J_regressor']=csr_matrix(dense)
            assert np.array_equal(dense,data['J_regressor'].toarray())
        return official(data)
    module.load_smpl_model_data=compatible_load
    import lib_smpl.const as constants
    constants.SMPL_ASSETS_ROOT=str(ASSETS)
def make_smplh(gender='male'):
    from lib_smpl.smpl_module import SMPLLayer
    return SMPLLayer(gender=gender,hands=True,model_root=str(SMPLH),num_betas=10)
