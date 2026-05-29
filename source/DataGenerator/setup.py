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
    scene.render.resolution_y = 480

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()

    for device in cycles_prefs.devices:
        device.use = True

    scene.cycles.device = 'GPU'
    scene.cycles.samples = random.randint(128, 256)

    # enable motion blur
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = random.uniform(0.1, 0.5)

# setup camera and lights
def setup_camera_light(scene):
    # add camera
    bpy.ops.object.camera_add(location=(0, -1, 0))
    cam = bpy.context.object
    scene.camera = cam

    cam.data.lens_unit = 'FOV'
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.angle = math.radians(69)  # RealSense RGB

    # disable depth of field
    cam.data.dof.use_dof = False

    # apply slight jitter to camera rotation
    cam.rotation_euler = (
        math.radians(90 + random.uniform(-1, 1)),
        math.radians(random.uniform(-1, 1)),
        math.radians(random.uniform(-1, 1)),
    )

    # add key light
    bpy.ops.object.light_add(type='AREA', location=(0, -1, 1))
    key = bpy.context.object
    key.data.energy = random.uniform(10, 80)
    key.data.size = random.uniform(0.1, 0.5)

    # add fill light
    bpy.ops.object.light_add(type='POINT', location=(
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(0.5, 2)
    ))
    fill = bpy.context.object
    fill.data.energy = random.uniform(5, 40)

    # set ambient background strength
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs[1].default_value = random.uniform(0.1, 0.3)

    return cam, key

# create a material with random roughness and specular
def create_material(color):
    m = bpy.data.materials.new(name=str(color))
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]

    bsdf.inputs["Base Color"].default_value = (*color, 1)
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

    bg.inputs['Color'].default_value = (
        random.uniform(0.2, 0.8),
        random.uniform(0.2, 0.8),
        random.uniform(0.2, 0.8),
        1
    )
    bg.inputs['Strength'].default_value = random.uniform(0.5, 1.5)

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
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()

    render = nodes.new(type='CompositorNodeRLayers')
    lens = nodes.new(type='CompositorNodeLensdist')
    mix = nodes.new(type='CompositorNodeMixRGB')
    noise = nodes.new(type='CompositorNodeRGB')
    comp = nodes.new(type='CompositorNodeComposite')

    lens.inputs[1].default_value = random.uniform(0.002, 0.01)   # distort
    lens.inputs[2].default_value = random.uniform(0.0, 0.001)    # dispersion

    noise_strength = random.uniform(0.001, 0.005)
    noise.outputs[0].default_value = (
        noise_strength,
        noise_strength,
        noise_strength,
        1.0
    )

    mix.blend_type = 'ADD'
    mix.inputs[0].default_value = random.uniform(0.005, 0.015)  # factor

    # connect compositor nodes
    links.new(render.outputs[0], lens.inputs[0])
    links.new(lens.outputs[0], mix.inputs[1])
    links.new(noise.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], comp.inputs[0])