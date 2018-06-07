import numpy as np

TAR_H = 401
TAR_W = 401

NUM_KP = 17
KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle']
KP_LB = 5
FACE_KP_LB = 0
BODY_KP_LB = 0
AREA_LB = 3000

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
BATCH_SIZE = 2
PREFETCH_SIZE = 1

HEATMAP_LOSS_WEIGHT = 4.0
SHORT_OFFSET_LOSS_WEIGHT = 1.0
MIDDLE_OFFSET_LOSS_WEIGHT = 0.5
