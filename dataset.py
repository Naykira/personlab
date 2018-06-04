import config
import random
import tensorflow as tf
import numpy as np
import cv2 as cv
from pycocotools.coco import COCO



def load():
    for name, ann_file, img_dir in config.COCO_FILES:
        coco = COCO(ann_file)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        img_infos = coco.loadImgs(img_ids)

        kp_list = []
        img_pack_list = []
        for img_info in img_infos:
            img_id = img_info['id']
            img_path = img_dir + '/' + img_info['file_name']
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            yield reform_image(coco, img_path, anns)


def reform_image(coco, img_path, anns):
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    height, width, depth = img.shape

    exclude_mask = np.zeros((height, width))
    kp_list = []
    for ann in anns:
        keypoints = ann['keypoints']
        keypoints = np.transpose(np.reshape(keypoints, (config.NUM_KP, 3)))
        keypoints[2] = keypoints[2] == 2
        kp_sum = np.sum(keypoints[2])
        face_kp_sum = np.sum(keypoints[2, :5])
        body_kp_sum = np.sum(keypoints[2, 5:])
        if ann['iscrowd'] or kp_sum < config.KP_LB or face_kp_sum < config.FACE_KP_LB or body_kp_sum < config.BODY_KP_LB:
            exclude_mask += coco.annToMask(ann)
        else:
            kp_list.append(keypoints)
    exclude_mask = (exclude_mask > 0).astype(np.uint8)
    return transform(img, exclude_mask, kp_list)


def transform(img, exclude_mask, kp_list):
    h, w, d = img.shape
    m = np.identity(3)
    m = m.dot(np.array([[1., 0., config.TAR_W//2], [0., 1., config.TAR_H//2], [0., 0., 1.]])) # tranform to 0,0
    if random.randint(0, 1) == 0:
        m = m.dot(np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])) # flip
    m = m.dot(np.array([[config.TAR_W/w, 0., 0.], [0., config.TAR_H/h, 0.], [0., 0., 1.]])) # scale
    m = m.dot(np.array([[1., 0., -w//2], [0., 1., -h//2], [0., 0., 1.]])) # transform to center

    img = cv.warpAffine(img, m[0:2], (config.TAR_W, config.TAR_H))
    exclude_mask = cv.warpAffine(exclude_mask, m[0:2], (config.TAR_W, config.TAR_H))
    kp_list = [m.dot(kp) for kp in kp_list]

    r = config.RADIUS
    x = np.tile(np.arange(-r, r+1), [2 * r + 1, 1])
    y = x.transpose()
    m = np.sqrt(x * x + y * y) <= r

    hm = np.zeros([config.TAR_W, config.TAR_H, config.NUM_KP]).astype('i')
    so_x = np.zeros([config.TAR_W, config.TAR_H, config.NUM_KP]).astype('i')
    so_y = np.zeros([config.TAR_W, config.TAR_H, config.NUM_KP]).astype('i')
    mo_x = np.zeros([config.TAR_W, config.TAR_H, config.NUM_EDGE]).astype('i')
    mo_y = np.zeros([config.TAR_W, config.TAR_H, config.NUM_EDGE]).astype('i')

    for kp in kp_list:
        for kp_i in range(config.NUM_KP):
            if kp[2, kp_i] == 0:
                continue
            cx = np.round(kp[0, kp_i])
            cy = np.round(kp[1, kp_i])
            mm = (0 <= x + cx) & (x + cx < config.TAR_W) & \
                 (0 <= y + cy) & (y + cy < config.TAR_H) & m
            ym = y[mm]
            xm = x[mm]
            y_i = (ym + cy).astype('i')
            x_i = (xm + cx).astype('i')
            hm[y_i, x_i, kp_i] = 1
            so_x[y_i, x_i, kp_i] = -xm
            so_y[y_i, x_i, kp_i] = -ym

        for e_i in range(config.NUM_EDGE):
            k_i = config.EDGES[e_i, 0]
            k_j = config.EDGES[e_i, 1]
            if kp[2, k_i] == 0 or kp[2, k_j] == 0:
                continue
            cx = np.round(kp[0, kp_i])
            cy = np.round(kp[1, kp_i])
            mm = (0 <= x + cx) & (x + cx < config.TAR_W) & \
                 (0 <= y + cy) & (y + cy < config.TAR_H) & m
            ym = y[mm]
            xm = x[mm]
            y_i = (ym + cy).astype('i')
            x_i = (xm + cx).astype('i')
            dy = kp[0, k_j] - cy
            dx = kp[1, k_j] - cx
            mo_x[y_i, x_i, e_i] = -xm + dx
            mo_y[y_i, x_i, e_i] = -ym + dy

    return img, exclude_mask, hm, so_x, so_y, mo_x, mo_y
