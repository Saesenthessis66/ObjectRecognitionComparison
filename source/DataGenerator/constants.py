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
TILT_X_MIN = -5
TILT_X_MAX = 5
TILT_Y_MIN = -5
TILT_Y_MAX = 5

# Offset between cubes in object
OFFSET = 0.002

# Rotation for dataset variation 
ROT_MIN = -5
ROT_MAX = 5
ROT_STEP = 5

# --- Camera distance ---
CAM_DIST_MIN = 0.4
CAM_DIST_MAX = 0.4
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