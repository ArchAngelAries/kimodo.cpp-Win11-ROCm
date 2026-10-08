"""Headless import check for kimodo.cpp exports (run inside Blender 5.2 in background mode).

Reads the BVH and GLB files, reports what Blender creates, and saves nothing.
Invoke: blender.exe --background --factory-startup --python blender_check.py -- <file.bvh> <file.glb>
"""
import json
import sys

import bpy

args = sys.argv[sys.argv.index('--') + 1:]
bvh_path, glb_path = args[0], args[1]


def emit(tag, data):
    print('BLENDER_CHECK', tag, json.dumps(data, sort_keys=True))


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def fcurves_of(action):
    # Blender 5.x actions are layered: fcurves live in layers > strips > channelbags.
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    found = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                found.extend(bag.fcurves)
    return found


def action_summary(action):
    if action is None:
        return None
    curves = fcurves_of(action)
    keys = sum(len(fc.keyframe_points) for fc in curves)
    return {'name': action.name, 'frame_range': [round(v, 3) for v in action.frame_range],
            'fcurves': len(curves), 'keyframes_total': keys}


# --- BVH --------------------------------------------------------------------
# The exporter writes centimetres (--scale 100). Import at 1.0 (as written) and at
# 0.01 (metres), so the bone lengths show which scale gives real-world size.
for bvh_scale in (1.0, 0.01):
    fresh_scene()
    bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=bvh_scale)
    scene = bpy.context.scene
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    bvh_action = None
    for arm in armatures:
        if arm.animation_data and arm.animation_data.action:
            bvh_action = arm.animation_data.action
    emit('bvh', {
        'global_scale': bvh_scale,
        'armatures': [a.name for a in armatures],
        'bones': sum(len(a.data.bones) for a in armatures),
        'scene_frames': [scene.frame_start, scene.frame_end],
        'scene_fps': round(scene.render.fps / scene.render.fps_base, 3),
        'action': action_summary(bvh_action),
        'hips_to_spine1_length': round(armatures[0].data.bones[0].length, 4) if armatures else None,
    })

# --- GLB --------------------------------------------------------------------
fresh_scene()
bpy.ops.import_scene.gltf(filepath=glb_path)
scene = bpy.context.scene
armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
glb_actions = list(bpy.data.actions)
emit('glb', {
    'armatures': [a.name for a in armatures],
    'bones': sum(len(a.data.bones) for a in armatures),
    'mesh_objects': len(meshes),
    'actions': [action_summary(a) for a in glb_actions],
    'scene_frames': [scene.frame_start, scene.frame_end],
    'scene_fps': round(scene.render.fps / scene.render.fps_base, 3),
})
