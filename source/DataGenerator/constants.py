import os
import bpy

# Light properties 
LIGHT_ENERGY_MIN = 8        # Minimum light energy
LIGHT_ENERGY_MAX = 40       # Maximum light energy

LIGHT_COLOR_MIN = (0.9, 0.9, 0.9)  # Min RGB color for light
LIGHT_COLOR_MAX = (1.0, 1.0, 1.0)  # Max RGB color for light

LIGHT_SIZE_MIN = 0.05       # Minimum area light size
LIGHT_SIZE_MAX = 0.3        # Maximum area light size

# Random object tilt 
TILT_X_MIN = -3
TILT_X_MAX = 3
TILT_Y_MIN = -3
TILT_Y_MAX = 3

# Offset between cubes in object
OFFSET = 0.002

# Rotation for dataset variation 
ROT_MIN = -40
ROT_MAX = 40
ROT_STEP = 5

# --- Camera distance ---
CAM_DIST_MIN = 0.4
CAM_DIST_MAX = 1.4
CAM_DIST_STEP = 0.05

# Class mapping (string -> YOLO class ID)
CLASS_MAP = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 3,
}

# Background types to randomly choose from
BACKGROUND_TYPES = [
    # "LIGHT_SOLID",
    # "DARK_SOLID",
    # "NEUTRAL_COLOR",
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

# EVAL: DISTANCE TEST
EVAL_DIST_MIN = 0.3
EVAL_DIST_MAX = 1.5
EVAL_DIST_STEP = 0.02
EVAL_DIST_ROT = 0  # fixed rotation

# EVAL: ROTATION TEST
EVAL_ROT_MIN = -50
EVAL_ROT_MAX = 50
EVAL_ROT_STEP = 2
EVAL_ROT_DIST = 0.5  # fixed distance

# Disable randomness
EVAL_TILT_X = 0
EVAL_TILT_Y = 0
EVAL_LIGHT_ENERGY = 20
EVAL_LIGHT_COLOR = (1, 1, 1)
EVAL_LIGHT_SIZE = 0.15