"""Control for the false penalty on a body-occluded helmet back surface."""
import os
from pathlib import Path
import torch,json
from visible_face_coverage import face_coverage
ROOT=Path(__file__).resolve().parent;out=ROOT/'visible_face_coverage_controls_v1';out.mkdir(exist_ok=False)
a=torch.ones(4,8,8);obj=torch.tensor([5.,2.,3.005,2.])[:,None,None].expand_as(a);human=torch.tensor([3.,3.,3.,0.])[:,None,None].expand_as(a);xy=torch.tensor([[[4,4]]]*4);valid=torch.ones(4,1,dtype=torch.bool)
num,den,raw=face_coverage(a,obj,human,xy,valid);assert num==1 and den==3 and raw==4
back=face_coverage(a[:1],obj[:1],human[:1],xy[:1],valid[:1]);assert back[0]==0 and back[1]==1 and back[2]==1
report={'success':True,'opaque_mesh_behind_face_not_penalized':True,'opaque_mesh_in_front_penalized':True,'uncertain_depth_unknown_and_margin_not_front':True,'scope':'Synthetic depth-order controls for visible-face overlap. Does not certify LHM expected depth as a true physical surface or validate real image quality.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
