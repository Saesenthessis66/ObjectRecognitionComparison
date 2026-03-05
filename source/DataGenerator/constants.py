import os
import bpy

LIGHT_ENERGY_MIN = 8
LIGHT_ENERGY_MAX = 40

LIGHT_COLOR_MIN = (0.9, 0.9, 0.9)
LIGHT_COLOR_MAX = (1.0, 1.0, 1.0)

LIGHT_SIZE_MIN = 0.05
LIGHT_SIZE_MAX = 0.3

TILT_X_MIN = -5
TILT_X_MAX = 5

TILT_Y_MIN = -5
TILT_Y_MAX = 5

OFFSET = 0.002

ROT_MIN = -5
ROT_MAX = 5
ROT_STEP = 5

CAM_DIST_MIN = 0.4
CAM_DIST_MAX = 0.4
CAM_DIST_STEP = 0.05

CLASS_MAP = {
    "00": 0,
    "01": 1,
    "10": 2,
    "11": 3,
}

BACKGROUND_TYPES = [
    # "LIGHT_SOLID",
    # "DARK_SOLID",
    # "NEUTRAL_COLOR",
    "HORIZONTAL_GRADIENT",
    "VERTICAL_GRADIENT",
    "NOISE_TEXTURE",
    "DIRTY_BACKGROUND"
]

BASE_DIR = os.path.dirname(bpy.data.filepath)
if BASE_DIR == "":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "preview")