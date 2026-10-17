"""Defer unused training visualization/evaluation imports for actual UniDepth inference."""
import os
from pathlib import Path
import importlib.abc,importlib.util,ast,sys
ROOT=Path(__file__).resolve().parent/'vendor/UniDepth'
class InferenceImports(importlib.abc.MetaPathFinder,importlib.abc.Loader):
 def find_spec(self,fullname,path=None,target=None):
    if fullname=='unidepth.utils':return importlib.util.spec_from_file_location(fullname,ROOT/'unidepth/utils/__init__.py',loader=self,submodule_search_locations=[str(ROOT/'unidepth/utils')])
    if fullname=='unidepth.utils.chamfer_distance':return importlib.util.spec_from_file_location(fullname,ROOT/'unidepth/utils/chamfer_distance.py',loader=self)
 def create_module(self,spec):return None
 def exec_module(self,module):
    filename=module.__spec__.origin;source=Path(filename).read_text()
    if module.__name__=='unidepth.utils':
        tree=ast.parse(source);lazy={};body=[]
        for node in tree.body:
            if isinstance(node,ast.ImportFrom) and node.module in ['visualization','evaluation_depth']:
                for alias in node.names:lazy[alias.asname or alias.name]=(node.module,alias.name)
            else:body.append(node)
        tree.body=body;exec(compile(tree,filename,'exec'),module.__dict__)
        def resolve(name):
            if name not in lazy:raise AttributeError(name)
            mod,symbol=lazy[name];value=getattr(__import__('unidepth.utils.'+mod,fromlist=[symbol]),symbol);setattr(module,name,value);return value
        module.__getattr__=resolve
    else:
        old='    knn_points = lambda x : x';assert source.count(old)==1
        source=source.replace(old,'    def knn_points(*args, **kwargs):\n        raise RuntimeError("Actual UniDepth KNN extension is required for evaluation; no placeholder operator is permitted")')
        exec(compile(source,filename,'exec'),module.__dict__)
def configure():
 sys.meta_path.insert(0,InferenceImports())
