import numpy as np

TAR_H = 401
TAR_W = 401


NUM_GPUS = 1
BATCH_SIZE = 10
PREFETCH_SIZE = 1

NUM_KP = 16
NUM_EDGE = 30
MAX_FRAME_SIZE = 10
NUM_RECURRENT = 2

RADIUS = 32

KP_LB = 5
FACE_KP_LB = 0
BODY_KP_LB = 0
AREA_LB = 4000

HEATMAP_LOSS_WEIGHT = 4.0
SHORT_OFFSET_LOSS_WEIGHT = 1.0
MIDDLE_OFFSET_LOSS_WEIGHT = 0.5


SURREAL_H = 240
SURREAL_W = 320
TRAIN_DATA_BASE_DIR = 'dataset/surreal/cmu/train/'
VAL_DATA_BASE_DIR = 'dataset/surreal/cmu/val/'

SURREAL_KP_MAP = [
    15, #턱 0
    12, #목 1
    9,  #가슴 2
    16, 18, 20, #왼쪽 어깨, 팔꿈치, 손목 3 4 5
    17, 19, 21, #오른쪽 어깨, 팔꿈치, 손목 6 7 8
    3, #배 9
    1, 4, 7, #왼쪽 엉덩이, 무릎, 발목 10 11 12
    2, 5, 8, #왼쪽 엉덩이, 무릎, 발목 13 14 15
]
EDGES = np.array([
    (0, 1),   # 0
    (1, 2),   # 1
    (2, 3),   # 2
    (3, 4),   # 3
    (4, 5),   # 4
    (2, 6),   # 5
    (6, 7),   # 6
    (7, 8),   # 7
    (2, 9),   # 8
    (9, 10),  # 9
    (10, 11), # 10
    (11, 12), # 11
    (9, 13),  # 12
    (13, 14), # 13
    (14, 15), # 14
])

EDGES = np.concatenate([EDGES, EDGES[:, ::-1]], axis=0)
