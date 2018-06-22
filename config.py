import numpy as np

TAR_H = 400
TAR_W = 400

NUM_KP = 15
KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle']
KP_LB = 5
FACE_KP_LB = 0
BODY_KP_LB = 0
AREA_LB = 4000

NUM_EDGE = 19
EDGES = np.array([
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),

    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (7, 9),

    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),

    (13, 15),
    (14, 16),
    (5, 6),
    (11, 12)
])

RADIUS = 32

COCO_FILES = []
COCO_FILES.append(('train2017','annotations/person_keypoints_train2017.json', 'train2017'))
#COCO_FILES.append(('val2017', '../KerasPersonLab/ANNO_FILE/person_keypoints_val2017.json', '../KerasPersonLab/IMG_DIR/val2017/'))

NUM_GPUS = 1
BATCH_SIZE = 1
PREFETCH_SIZE = 1

HEATMAP_LOSS_WEIGHT = 4.0
SHORT_OFFSET_LOSS_WEIGHT = 1.0
MIDDLE_OFFSET_LOSS_WEIGHT = 0.5


SURREAL_H = 240
SURREAL_W = 320
TRAIN_DATA_BASE_DIR = '/home/ubuntu/personlab/cmu/train/'
VAL_DATA_BASE_DIR = '/home/ubuntu/personlab/cmu/val/'


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
SURREAL_EDGE = np.array([
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (2, 6),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
])
EDGES = SURREAL_EDGE

NUM_KP = 16
NUM_EDGE = 15
MAX_FRAME_SIZE = 10
NUM_RECURRENT = 2
