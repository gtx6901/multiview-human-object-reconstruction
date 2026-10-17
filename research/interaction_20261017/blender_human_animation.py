"""Real Blender original-topology rig conversion, animation and GLB audit."""
import os
import bpy,numpy as np,json,sys,ast,math
from pathlib import Path
from mathutils import Matrix,Vector
from mathutils.kdtree import KDTree
job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text());out=Path(job['out']);d=np.load(job['data'])
# Read genuine addon joint order as data; no copied or guessed bone ordering.
tree=ast.parse((Path(job['official_addon_source'])/'utils/model_spec.py').read_text())
names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and getattr(n.target,'id','')=='_SMPLX_JOINT_NAMES')
assert len(names)==55
bpy.ops.wm.read_factory_settings(use_empty=True)
data=bpy.data.meshes.new('OriginalSMPLXTopology');data.from_pydata(d['template'].tolist(),[],d['faces'].tolist());data.update()
obj=bpy.data.objects.new('SMPLX-mesh-neutral',data);bpy.context.collection.objects.link(obj)
uv=data.uv_layers.new(name='OfficialSMPLXUV');uv.data.foreach_set('uv',d['vt'][d['ft'].reshape(-1)].astype('float32').reshape(-1))
obj.shape_key_add(name='Basis');cloth=obj.shape_key_add(name='ClothOffset');cloth.data.foreach_set('co',(d['template']+d['offset']).reshape(-1));cloth.value=1
for k in range(486):
    key=obj.shape_key_add(name=f'Pose{k:03d}');key.slider_min=-10;key.slider_max=10
    key.data.foreach_set('co',(d['template']+d['posedirs'][:,:,k]).reshape(-1))
arm=bpy.data.objects.new('SMPLX-neutral',bpy.data.armatures.new('OriginalSMPLX55'));bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active=arm;arm.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for i,name in enumerate(names):
    bone=arm.data.edit_bones.new(name);bone.head=d['joints'][i];bone.tail=d['joints'][i]+np.array([0,.03,0]);bone.roll=0
    if i:bone.parent=arm.data.edit_bones[names[int(d['parents'][i])]]
bpy.ops.object.mode_set(mode='OBJECT');obj.parent=arm
modifier=obj.modifiers.new('Original55JointLBS','ARMATURE');modifier.object=arm;modifier.use_deform_preserve_volume=False
for i,name in enumerate(names):
    group=obj.vertex_groups.new(name=name)
    for vi in np.flatnonzero(d['weights'][:,i]>0):group.add([int(vi)],float(d['weights'][vi,i]),'REPLACE')
arm.matrix_world=Matrix.Rotation(math.pi/2,4,'X')
material=bpy.data.materials.new('NeuralBakedRGB');material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear();output=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission');image=nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(job['texture']);image.image.pack();material.node_tree.links.new(image.outputs['Color'],emission.inputs['Color']);material.node_tree.links.new(emission.outputs[0],output.inputs['Surface']);data.materials.append(material)
for poly in data.polygons:poly.use_smooth=True
obj['body_model']='smplx';obj['scope']=job['scope'];obj['official_stock_blender_library']=False
scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=300;scene.render.fps=30
for t in range(300):
    for i,name in enumerate(names):
        bone=arm.pose.bones[name];bone.rotation_mode='QUATERNION';bone.rotation_quaternion=d['quaternion'][t,i];bone.keyframe_insert('rotation_quaternion',frame=t+1,group=name)
    arm.pose.bones[names[0]].location=d['translation'][t];arm.pose.bones[names[0]].keyframe_insert('location',frame=t+1,group=names[0])
    for k,value in enumerate(d['coeff'][t]):
        key=obj.data.shape_keys.key_blocks[f'Pose{k:03d}'];key.value=float(value);key.keyframe_insert('value',frame=t+1)
    if t%50==0:print('animation keyed',t,flush=True)
def evaluated_points(object):
    evaluated=object.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh();p=np.empty(len(mesh.vertices)*3,dtype='float32');mesh.vertices.foreach_get('co',p);p=p.reshape(-1,3);matrix=np.array(evaluated.matrix_world);p=p@matrix[:3,:3].T+matrix[:3,3];evaluated.to_mesh_clear();return p
G=np.array(arm.matrix_world)[:3,:3];checks=[]
for n,t in enumerate(d['sample_frames']):
    scene.frame_set(int(t)+1);bpy.context.view_layer.update();ref=d['reference'][n]@G.T;actual=evaluated_points(obj);error=float(np.max(np.linalg.norm(actual-ref,axis=1)));assert error<3e-5,(int(t),error);checks.append({'frame':int(t),'max_original_lbs_vertex_error':error})
scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(out/'human_animated_candidate.blend'))
# Blender's GLTF importer derives slider limits from INITIAL morph weights and
# clamps later negative coefficients. Encode signed deltas as positive pairs,
# preserving exact geometry and keeping every exported animation weight in[0,1].
obj.data.shape_keys.animation_data_clear()
for k in reversed(range(486)):obj.shape_key_remove(obj.data.shape_keys.key_blocks[f'Pose{k:03d}'])
compatible_keys=[]
for k in range(486):
    for sign,label in [(1,'Positive'),(-1,'Negative')]:
        values=np.maximum(sign*d['coeff'][:,k],0)/2
        if values.max()<1e-9:continue
        assert values.min()>=0 and values.max()<=1.000001
        key=obj.shape_key_add(name=f'Corrective{k:03d}_{label}');key.data.foreach_set('co',(d['template']+sign*2*d['posedirs'][:,:,k]).reshape(-1));key.slider_min=0;key.slider_max=1
        for t,value in enumerate(values):key.value=float(value);key.keyframe_insert('value',frame=t+1)
        compatible_keys.append(key.name)
scene.frame_set(1);bpy.context.view_layer.update()
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);arm.select_set(True);bpy.context.view_layer.objects.active=arm
result=bpy.ops.export_scene.gltf(filepath=str(out/'human_animated_candidate.glb'),export_format='GLB',use_selection=True,export_animations=True,export_animation_mode='SCENE',export_anim_scene_split_object=False,export_frame_range=True,export_force_sampling=True,export_all_influences=True,export_morph_normal=False,export_yup=True,export_extras=True);assert result=={'FINISHED'}
bpy.ops.wm.read_factory_settings(use_empty=True);bpy.context.scene.render.fps=30
assert bpy.ops.import_scene.gltf(filepath=str(out/'human_animated_candidate.glb'))=={'FINISHED'}
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)];arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE'];assert len(meshes)==len(arms)==1
loaded=meshes[0];assert len(loaded.data.polygons)==20908 and loaded.data.uv_layers.active and len(arms[0].pose.bones)>=55
roundtrip=[]
for n,t in enumerate(d['sample_frames']):
    bpy.context.scene.frame_set(int(t)+1);bpy.context.view_layer.update();points=evaluated_points(loaded);reference=d['reference'][n]@G.T;kd=KDTree(len(reference))
    for i,p in enumerate(reference):kd.insert(Vector(p),i)
    kd.balance();error=max(kd.find(Vector(p))[2] for p in points);assert error<2e-4,(int(t),error);roundtrip.append({'frame':int(t),'max_nearest_reference_vertex_error':error})
report={'success':True,'blender_version':bpy.app.version_string,'frames':300,'bones':55,'pose_corrective_shape_keys':486,'gltf_nonnegative_corrective_targets':len(compatible_keys),'gltf_corrective_weight_range':[0,1],'cloth_offset_shape_key':True,'original_lbs_checks':checks,'glb_roundtrip_checks':roundtrip,'scope':job['scope']+' Actual Blender armature/shape-key animation and GLB export/reimport checked at5 frames. OriginalBLEND retains486 signed pose correctives; GLB losslessly reparameterizes them as nonnegative positive/negative delta targets to avoid importer slider clamping. No stock official addon library import claimed; no global self-intersection/contact or independent novel-view quality acceptance.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('animatedGLB verified',flush=True)
