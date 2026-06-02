import bpy
import math
import random
import mathutils
from constants import *

# scene setup
def setup_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# configure render engine
def configure_render_engine(scene):
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640  

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()

    for device in cycles_prefs.devices:
        device.use = True

    scene.cycles.device = 'GPU'

    scene.cycles.samples = 128

    scene.render.use_motion_blur = False

# setup camera and lights
def setup_camera_light(scene):
    bpy.ops.object.camera_add(location=(0, -1, 0))
    cam = bpy.context.object
    scene.camera = cam

    cam.data.lens_unit = 'FOV'

    cam.data.angle = math.radians(69)

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    cam.data.sensor_fit = 'HORIZONTAL'

    cam.data.dof.use_dof = False
    cam.rotation_euler = (math.radians(90), 0, 0)

    # simple stable light
    bpy.ops.object.light_add(type='AREA', location=(0, -1, 1))
    light = bpy.context.object
    light.data.energy = 25
    light.data.size = 0.2

    return cam, light

# create a material with random roughness and specular
def create_material(color):
    m = bpy.data.materials.new(name=str(color))
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]

    bsdf.inputs["Base Color"].default_value = (*color, 1)

    if color == (0.02, 0.02, 0.02):
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular"].default_value = 0.0
        bsdf.inputs["Metallic"].default_value = 0.0
    else:
        bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)
        bsdf.inputs["Specular"].default_value = random.uniform(0.2, 0.6)

    return m

# create a transparent material
def create_transparent_material():
    m = bpy.data.materials.new(name="TRANSPARENT")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]

    bsdf.inputs["Alpha"].default_value = 0.0
    m.blend_method = 'BLEND'

    return m

# create default materials for the scene
def create_materials():
    return {
        "GREEN": create_material((0, 1, 0)),
        "WHITE": create_material((1, 1, 1)),
        "BLACK": create_material((0.02, 0.02, 0.02)),
        "TRANSPARENT": create_transparent_material(),
    }

# set a random world background
def set_random_background(scene):
    world = scene.world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputWorld')
    bg = nodes.new(type='ShaderNodeBackground')

    links.new(bg.outputs['Background'], output.inputs['Surface'])

    bg_type = random.choice(BACKGROUND_TYPES)

    if bg_type == "LIGHT_SOLID":
        val = random.uniform(0.7, 0.85)
        bg.inputs['Color'].default_value = (val, val, val, 1)

    elif bg_type == "DARK_SOLID":
        val = random.uniform(0.05, 0.3)
        bg.inputs['Color'].default_value = (val, val, val, 1)

    elif bg_type == "NEUTRAL_COLOR":
        r = random.uniform(0.6, 0.9)
        g = random.uniform(0.6, 0.9)
        b = random.uniform(0.5, 0.8)
        bg.inputs['Color'].default_value = (r, g, b, 1)

    # Procedural gradient background
    elif bg_type in ["HORIZONTAL_GRADIENT", "VERTICAL_GRADIENT"]:
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        gradient = nodes.new(type='ShaderNodeTexGradient')
        mapping = nodes.new(type='ShaderNodeMapping')

        # Rotate gradient if vertical
        if bg_type == "VERTICAL_GRADIENT":
            mapping.inputs['Rotation'].default_value[2] = math.radians(90)

        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
        links.new(gradient.outputs['Color'], bg.inputs['Color'])

    # Procedural noise background
    elif bg_type == "NOISE_TEXTURE":
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        noise = nodes.new(type='ShaderNodeTexNoise')

        noise.inputs['Scale'].default_value = random.uniform(2, 6)
        noise.inputs['Detail'].default_value = 2

        links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])
        links.new(noise.outputs['Color'], bg.inputs['Color'])

    # Dirty / uneven background for realism
    elif bg_type == "DIRTY_BACKGROUND":
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        noise = nodes.new(type='ShaderNodeTexNoise')
        ramp = nodes.new(type='ShaderNodeValToRGB')

        noise.inputs['Scale'].default_value = random.uniform(5, 15)
        noise.inputs['Detail'].default_value = 5

        ramp.color_ramp.elements[0].color = (0.3, 0.3, 0.3, 1)
        ramp.color_ramp.elements[1].color = (0.7, 0.7, 0.7, 1)

        links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bg.inputs['Color'])

    bg.inputs['Strength'].default_value = random.uniform(0.7, 1.3)

# create the object from bit materials
def create_object(materials, bits):
    obj_list = []
    bit_material = {
        0: materials["BLACK"],
        1: materials["WHITE"]
    }
    pos = []
    pos.append(OFFSET / 2 + 0.05 / 2)
    pos.append(pos[0] + 0.05 / 2 + 0.07 / 2 + OFFSET)

    # add a cube and assign material
    def add_cube(location, scale, material):
        bpy.ops.mesh.primitive_cube_add(size=1)
        o = bpy.context.object
        o.location = location
        if material == materials["BLACK"]:
            o.location.y += pos[0]
            scale = (scale[0], scale[1] * 0.05, scale[2])
        o.scale = scale
        o.data.materials.append(material)
        obj_list.append(o)

    add_cube((pos[0], 0, 0), (0.05, 0.05, 0.05), bit_material[bits[1]])
    add_cube((pos[1], 0, 0), (0.07, 0.05, 0.05), materials["GREEN"])
    add_cube((-pos[0], 0, 0), (0.05, 0.05, 0.05), bit_material[bits[0]])
    add_cube((-pos[1], 0, 0), (0.07, 0.05, 0.05), materials["GREEN"])

    bpy.ops.object.select_all(action='DESELECT')
    for o in obj_list:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obj_list[0]
    bpy.ops.object.join()

    return bpy.context.object

# setup compositor nodes for lens distortion and noise
def setup_compositor(scene):
    scene.use_nodes = False