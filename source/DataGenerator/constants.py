import os
import bpy

# Light properties
LIGHT_ENERGY_MIN = 8        # Minimum light energy
LIGHT_ENERGY_MAX = 40       # Maximum light energy

LIGHT_COLOR_MIN = (0.9, 0.9, 0.9)  # Minimum RGB color for light
LIGHT_COLOR_MAX = (1.0, 1.0, 1.0)  # Maximum RGB color for light

LIGHT_SIZE_MIN = 0.05       # Minimum area light size
LIGHT_SIZE_MAX = 0.3        # Maximum area light size

# Random object tilt
TILT_X_MIN = -6
TILT_X_MAX = 6
TILT_Y_MIN = -6
TILT_Y_MAX = 6

# Offset between cubes in object
OFFSET = 0.002

# Rotation for dataset variation
ROT_MIN = -48
ROT_MAX = 48
ROT_STEP = 4

# Camera distance
CAM_DIST_MIN = 0.2   
CAM_DIST_MAX = 1.3   
CAM_DIST_STEP = 0.04

# Class mapping from string to YOLO class id
CLASS_MAP = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 3,
}

# Background types to randomly choose from
BACKGROUND_TYPES = [
    "LIGHT_SOLID",
    "DARK_SOLID",
    "NEUTRAL_COLOR",
    "HORIZONTAL_GRADIENT",
    "VERTICAL_GRADIENT",
    "NOISE_TEXTURE",
    "DIRTY_BACKGROUND"
]

# File paths
BASE_DIR = os.path.dirname(bpy.data.filepath)
if BASE_DIR == "":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = "/workspace/preview"

# Eval distance test
EVAL_DIST_MIN = 0.2
EVAL_DIST_MAX = 1.8
EVAL_DIST_STEP = 0.05
EVAL_DIST_ROT = 0  # Fixed rotation

# Eval rotation test
EVAL_ROT_MIN = -80
EVAL_ROT_MAX = 80
EVAL_ROT_STEP = 10
EVAL_ROT_DIST = 0.5  # Fixed distance

# Disable randomness
EVAL_TILT_X = 0
EVAL_TILT_Y = 0
EVAL_LIGHT_ENERGY = 20
EVAL_LIGHT_COLOR = (1, 1, 1)
EVAL_LIGHT_SIZE = 0.15