"""A shared nonplanar rim family; geometry prior, not observed cavity recovery."""
import os
import numpy as np,trimesh
def clip_signed_vertex_field(vertices,faces,field):
 """Positive-field part of each triangle, with linear edge interpolation."""
 result=[];triangles=[]
 for indices in faces:
  points=vertices[indices];values=field[indices];polygon=[]
  for a,b in [(0,1),(1,2),(2,0)]:
   inside=values[a]>=0
   if inside:polygon.append(points[a])
   if inside!=(values[b]>=0):polygon.append(points[a]+(points[b]-points[a])*values[a]/(values[a]-values[b]))
  if len(polygon)<3:continue
  start=len(result);result.extend(polygon)
  triangles.extend([[start,start+j,start+j+1] for j in range(1,len(polygon)-1)])
 assert result and triangles
 return np.array(result),np.array(triangles,dtype=np.int64)
def cut_opening(closed,normal,origin,up,front,side_drop=0.,back_drop=0.):
 if side_drop==0 and back_drop==0:
  v,f,_=trimesh.intersections.slice_faces_plane(closed.vertices,closed.faces,normal,origin)
 else:
  u=np.array(up,dtype=float);u/=np.linalg.norm(u);fr=np.array(front,dtype=float);fr-=u*(fr@u);fr/=np.linalg.norm(fr);q=closed.vertices-closed.center_mass;q-=np.outer(q@u,u);cosine=(q@fr)/np.maximum(np.linalg.norm(q,axis=1),1e-10);cosine=np.clip(cosine,-1,1)
  field=(closed.vertices-origin)@normal+side_drop*(1-cosine**2)+back_drop*np.maximum(-cosine,0)**2
  v,f=clip_signed_vertex_field(closed.vertices,closed.faces,field)
 outer=trimesh.Trimesh(v,f,process=False);outer.merge_vertices(digits_vertex=10);outer.remove_unreferenced_vertices();assert outer.is_winding_consistent and len(outer.split(only_watertight=False))==1
 return outer
