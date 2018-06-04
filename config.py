import numpy as np

TAR_H = 800
TAR_W = 800

NUM_KP = 17
KP_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle']
KP_LB = 5
FACE_KP_LB = 0
BODY_KP_LB = 0

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


COCO_FILES = [
    ('train2017',
     'annotations/person_keypoints_train2017.json',
     'train2017')
]
BATCH_SIZE = 2
PREFETCH_SIZE = 1
