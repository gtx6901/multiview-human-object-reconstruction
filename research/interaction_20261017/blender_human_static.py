"""Executed by real Blender: import canonical OBJ, render, GLB export/reimport audit."""
import os
import bpy,json,sys
from pathlib import Path
from mathutils import Vector
job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text());out=Path(job['out'])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=job['obj'],forward_axis='NEGATIVE_Z',up_axis='Y')
objects=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert len(objects)==1
obj=objects[0];obj.name='FCSHumanStaticCandidate';assert len(obj.data.vertices)==10475 and len(obj.data.polygons)==20908
assert obj.data.uv_layers.active and len(obj.data.uv_layers.active.data)==20908*3
material=bpy.data.materials.new('NeuralBakedAppearance_Unlit');material.use_nodes=True;nodes=material.node_tree.nodes;nodes.clear();output=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission');image=nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(job['texture'],check_existing=True);image.image.colorspace_settings.name='sRGB'
material.node_tree.links.new(image.outputs['Color'],emission.inputs['Color']);material.node_tree.links.new(emission.outputs['Emission'],output.inputs['Surface']);emission.inputs['Strength'].default_value=1
obj.data.materials.clear();obj.data.materials.append(material)
for polygon in obj.data.polygons:polygon.use_smooth=True
obj['scope']='Static candidate distilled from all8-trained neural avatar; original SMPL-X topology with cloth offsets. Rig/animation pending official addon model data. Local licensed derived asset.'
def bounds(object):
    p=[object.matrix_world@Vector(v) for v in object.bound_box];return [min(v[i] for v in p) for i in range(3)],[max(v[i] for v in p) for i in range(3)]
bpy.context.view_layer.update();lo,hi=bounds(obj);center=Vector([(a+b)/2 for a,b in zip(lo,hi)]);extent=max(b-a for a,b in zip(lo,hi))
camera=bpy.data.objects.new('ReviewCamera',bpy.data.cameras.new('ReviewCamera'));bpy.context.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.ortho_scale=extent*1.35;camera.location=center+Vector((extent*1.6,-extent*3.,extent*.5));camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler();bpy.context.scene.camera=camera
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=768;scene.render.resolution_y=768;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out/'human_static_preview.png');scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
world=bpy.data.worlds.new('White');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);scene.world=world
bpy.ops.wm.save_as_mainfile(filepath=str(out/'human_static_candidate.blend'))
bpy.ops.render.render(write_still=True)
bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
result=bpy.ops.export_scene.gltf(filepath=str(out/'human_static_candidate.glb'),export_format='GLB',use_selection=True,export_animations=False,export_yup=True,export_extras=True);assert result=={'FINISHED'}
source_counts={'vertices':len(obj.data.vertices),'faces':len(obj.data.polygons),'uv_loops':len(obj.data.uv_layers.active.data)}
bpy.ops.wm.read_factory_settings(use_empty=True);result=bpy.ops.import_scene.gltf(filepath=str(out/'human_static_candidate.glb'));assert result=={'FINISHED'}
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH'];assert len(meshes)==1;loaded=meshes[0];bpy.context.view_layer.update();newlo,newhi=bounds(loaded)
error=max(abs(a-b) for a,b in zip(lo+hi,newlo+newhi));assert error<2e-5
assert len(loaded.data.polygons)==20908 and loaded.data.uv_layers.active and loaded.data.materials
images=[i for i in bpy.data.images if i.size[0]==1024 and i.size[1]==1024];assert images
report={'success':True,'blender_version':bpy.app.version_string,'obj_import_counts':source_counts,'glb_reimport_counts':{'vertices':len(loaded.data.vertices),'faces':len(loaded.data.polygons)},'max_roundtrip_bbox_error':error,'texture_images_1024':len(images),'scope':'Real Blender imported original-topology textured OBJ, rendered unlit candidate, savedBLEND, exportedGLB and reimported it with triangle/UV/texture/bounds checks. GLTF may split vertices at UV seams. STATIC asset only: no official-addon rig import, shape keys or animation completion. Synthetic neural appearance and inferred texels remain; local licensed derived model must not be published with source code.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print('Blender static GLB roundtrip verified',flush=True)
