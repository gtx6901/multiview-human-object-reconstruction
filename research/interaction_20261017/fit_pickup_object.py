"""One free rigid helmet trajectory constrained by visible multicamera masks."""
import os
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'lhmpp_20261016/deps')]
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from pytorch3d.transforms import axis_angle_to_matrix
from mesh_renderer import MeshRenderer
torch.set_num_threads(4);out=ROOT/'helmet_pose_pickup000_135_v1';out.mkdir(exist_ok=False);start=time.time()
data=ROOT/'helmet_pickup_inspect000_135_v1';objects_root=ROOT/'helmet_track_pickup000_135_v2';human_root=ROOT/'people_pickup000_135_v1'
meta=json.loads((data/'manifest.json').read_text());objects={(x['serial'],x['frame']):x for x in json.loads((objects_root/'report.json').read_text())['records']};humans={(x['serial'],x['frame']):x for x in json.loads((human_root/'report.json').read_text())['records']}
mesh=torch.load(ROOT/'helmet_hull_frame30_v1/mesh_and_anchor.pt',weights_only=True);base=mesh['vertices_local'].cuda();faces=mesh['faces'].cuda();logscale=mesh['logscale'].cuda();fixed_scale=logscale.exp();renderer=MeshRenderer();W=torch.tensor([v['W'] for v in meta['views']],device='cuda');Ks=np.array([v['K'] for v in meta['views']]);cache={}
def observations(frame,width):
    key=(frame,width)
    if key in cache:return cache[key]
    height=width*3//4;s=width/1024;a=np.array([[s,0,s*.5-.5],[0,s,s*.5-.5],[0,0,1]])
    indices=[];masks=[];valids=[];distances=[]
    for vi,v in enumerate(meta['views']):
        rec=objects[(v['serial'],frame)]
        if rec['mask'] is None:continue
        m=np.asarray(Image.open(objects_root/rec['mask']))>127;human=humans[(v['serial'],frame)]
        if human['mask'] is not None:h=np.asarray(Image.open(human_root/human['mask']))>127
        else:h=np.zeros_like(m)
        valid=np.asarray(Image.open(data/v['serial']/'valid.png'))>127
        roi=np.zeros_like(m);x0,y0,x1,y1=rec['roi_xyxy'];roi[y0:y1,x0:x1]=True
        valid=valid&roi&(~h|m)
        indices.append(vi);masks.append(cv2.resize(m.astype('float32'),(width,height),interpolation=cv2.INTER_AREA));valids.append(cv2.resize(valid.astype('float32'),(width,height),interpolation=cv2.INTER_NEAREST));distances.append(cv2.resize(distance_transform_edt(~m).astype('float32')/1024,(width,height)))
    ids=torch.tensor(indices,device='cuda',dtype=torch.long);K=torch.tensor(np.array([a@Ks[i] for i in indices]),device='cuda',dtype=torch.float32)
    value=(ids,K,torch.tensor(np.array(masks),device='cuda'),torch.tensor(np.array(valids),device='cuda'),torch.tensor(np.array(distances),device='cuda'),height);cache[key]=value
    return value
def fit(frame,seed,width=384,steps=45):
    ids,K,target,visible,dist,height=observations(frame,width)
    if len(ids)<2:return {**seed,'loss':None,'view_indices':ids.cpu().tolist(),'ious':[],'status':'temporal carry; fewer than two usable silhouette views'}
    aa=torch.nn.Parameter(torch.zeros(3,device='cuda'));translation=torch.nn.Parameter(seed['translation_world'].clone());opt=torch.optim.Adam([{'params':[aa],'lr':.025},{'params':[translation],'lr':.0025}]);best=None
    for step in range(steps+1):
        opt.zero_grad();R=axis_angle_to_matrix(aa)@seed['R'];world=base@R.T*fixed_scale+translation
        alpha=renderer.alpha_batch(world[None].expand(len(ids),-1,-1),faces,K,W[ids],width,height);pred=alpha*visible
        intersection=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1);iou=intersection/union
        loss=(1-iou+2*(pred*dist).sum((1,2))/target.sum((1,2)).clamp_min(1)).mean()+.006*aa.square().mean()
        if best is None or float(loss)<best['loss']:best={'R':R.detach().clone(),'translation_world':translation.detach().clone(),'loss':float(loss),'view_indices':ids.cpu().tolist(),'ious':iou.detach().cpu().tolist(),'status':'multiview silhouette estimate; geometry/orientation ambiguity remains'}
        if step<steps:loss.backward();opt.step()
        for group,lr in zip(opt.param_groups,[.025,.0025]):group['lr']=lr*(1-.8*step/steps)
    return best
initial={'R':mesh['R'].cuda(),'translation_world':mesh['translation_world'].cuda()};states={30:fit(30,initial,steps=90)}
for direction in [1,-1]:
    previous=states[30]
    for frame in range(30+direction,136 if direction>0 else -1,direction):
        state=fit(frame,previous);states[frame]=state;previous=state
    print('direction complete',direction,flush=True)
frames=list(range(136));sequence={'frames':torch.tensor(frames),'R':torch.stack([states[f]['R'].cpu() for f in frames]),'translation_world':torch.stack([states[f]['translation_world'].cpu() for f in frames]),'logscale':logscale.cpu(),'vertices_local':base.cpu(),'faces':faces.cpu()};torch.save(sequence,out/'sequence.pt')
metrics=[]
for frame in frames:
    s=states[frame];metrics.append({'frame':frame,'loss':s['loss'],'observed_views':[meta['views'][i]['serial'] for i in s['view_indices']],'soft_visible_iou':s['ious'],'status':s['status']})
    if frame not in [0,30,60,90,120,135]:continue
    with torch.no_grad():
        world=base@s['R'].T*fixed_scale+s['translation_world'];factor=.5;A=np.array([[factor,0,factor*.5-.5],[0,factor,factor*.5-.5],[0,0,1]]);K=torch.tensor(np.array([A@k for k in Ks]),device='cuda',dtype=torch.float32);alpha=renderer.alpha_batch(world[None].expand(8,-1,-1),faces,K,W,512,384).cpu().numpy();sheet=Image.new('RGB',(4*512,2*408),'white')
        for i,v in enumerate(meta['views']):
            arr=np.asarray(Image.open(data/v['serial']/f'{frame:04d}.png').resize((512,384))).copy();shown=alpha[i]>.5;arr[shown]=np.uint8(arr[shown]*.5+np.array([255,0,180])*.5);sheet.paste(Image.fromarray(arr),(i%4*512,i//4*408+24));ImageDraw.Draw(sheet).text((i%4*512+3,i//4*408+4),v['serial']+f' frame{frame}',fill='black')
        sheet.save(out/f'{frame:04d}_all8_overlay.jpg')
report={'success':True,'frames':frames,'retained_cameras':8,'shape_source':'helmet_hull_frame30_v1','metrics':metrics,'temporal_carry_frames':[f for f in frames if states[f]['loss'] is None],'seconds':time.time()-start,'scope':'Free rigid rotation/translation, one closed fixed-scale mesh across pickup/inspection. Visible candidate masks constrain one common pose; unknown views excluded and ROI exterior unknown. All8 observed human masks model potential occlusion; no hand/head attachment. Anchor frame30, forward/backward sequential fitting. Orientation remains underconstrained by silhouettes, cavity filled; no physical contact solver or CoCoNet predictions used. Temporal carry flagged rather than invented observations.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('complete',flush=True)
