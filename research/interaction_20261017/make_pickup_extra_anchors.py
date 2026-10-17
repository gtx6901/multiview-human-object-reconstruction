import os
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent;out=ROOT/'pickup_extra_prompts_v1';out.mkdir(exist_ok=False)
records=[{'serial':'21305717','frame':45,'box':[342,215,416,277],'pos':[[375,244],[389,261]],'neg':[[375,200],[425,240],[350,283]],'status':'prompted','prompt_source':'Actual frame45 source grid inspection; distinguish helmet gray band from white shirt.'},{'serial':'21305717','frame':105,'box':[310,270,408,360],'pos':[[345,300],[360,337]],'neg':[[425,335],[405,285],[367,250],[322,260]],'status':'prompted','prompt_source':'Actual frame105 source grid inspection; reject shirt/hand/strap leakage.'}]
(out/'prompts.json').write_text(json.dumps({'records':records},indent=2))
