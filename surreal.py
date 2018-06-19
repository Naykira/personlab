import random, os, itertools
import config
import tensorflow as tf
import numpy as np
import cv2 as cv
import scipy.io as sio



def get_file_list(path):
    res = []
    for fn in os.listdir(path):
        if not os.path.isfile(path + fn):
            res += get_file_list(path + fn + '/')
        elif fn.endswith('mp4'):
            name = fn.split('.')[0]
            res.append(path + name)
    return res
    

def load():
    path_list = get_file_list(config.DATA_BASE_DIR)
    X = np.tile(np.arange(config.SURREAL_W), [config.SURREAL_H, 1])
    Y = np.tile(np.arange(config.SURREAL_H), [config.SURREAL_W, 1]).transpose()
    for p in itertools.cycle(path_list):
        cap = cv.VideoCapture(p + '.mp4')
        
        frames = []
        while(True):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        frames = np.array(frames)
        num_frame = len(frames)
        if num_frame < config.MAX_FRAME_SIZE:
            continue
            
        info = sio.loadmat(p + '_info.mat')
        seg = sio.loadmat(p + '_segm.mat')
        kp_list = []
        for f_i in range(num_frame):
            kps = []
            for k_i, sk_i in enumerate(config.SURREAL_KP_MAP):
                if np.sum(seg['segm_%d' % (f_i + 1)] == (sk_i + 1)) > 0:
                    x = info['joints2D'][0, sk_i, f_i]
                    y = info['joints2D'][1, sk_i, f_i]
                    c = 2
                else:
                    x = 0
                    y = 0
                    c = 0
                kps.append((x, y, c))
            kp_list.append([np.array(kps)])
        for i in range(num_frame // config.MAX_FRAME_SIZE):
            f_i_s = i * config.MAX_FRAME_SIZE
            f_i_e = f_i_s + config.MAX_FRAME_SIZE
            yield transform(frames[f_i_s:f_i_e,...], kp_list[f_i_s:f_i_e])

        
def transform(frames, kp_list):
    fn, h, w, d = frames.shape
    is_flip = random.randint(0, 1) == 0
    sf = random.uniform(1, 1.3)
    sf = min(config.TAR_W/w, config.TAR_H/h) * sf
    offset_w = random.uniform(0, max(0, w * sf - config.TAR_W)) // 2
    offset_h = random.uniform(0, max(0, h * sf - config.TAR_H)) // 2

    m = np.identity(3)
    m = m.dot(np.array([[1., 0., w*sf//2 - offset_w], [0., 1., h*sf//2 - offset_h], [0., 0., 1.]])) # tranform to 0,0
    if is_flip:
        m = m.dot(np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])) # flip
    m = m.dot(np.array([[sf, 0., 0.], [0., sf, 0.], [0., 0., 1.]])) # scale
    m = m.dot(np.array([[1., 0., -w//2], [0., 1., -h//2], [0., 0., 1.]])) # transform to center
    
    frames = [cv.warpAffine(img, m[0:2], (config.TAR_W, config.TAR_H)) for img in frames]
    new_kp_list = []
    for kpl in kp_list:
        l = []
        for kps in kpl:
            kps = kps.transpose()
            tmp = list(kps[2, :])
            kps[2, :] = 1
            kps = m.dot(kps)
            kps[2, :] = tmp
            l.append(kps)
        new_kp_list.append(l)
    kp_list = new_kp_list

    r = config.RADIUS
    x = np.tile(np.arange(-r, r+1), [2 * r + 1, 1])
    y = x.transpose()
    m = np.sqrt(x * x + y * y) <= r

    hm = np.zeros([fn, config.TAR_H, config.TAR_W, config.NUM_KP]).astype('i')
    so_x = np.zeros([fn, config.TAR_H, config.TAR_W, config.NUM_KP]).astype('i')
    so_y = np.zeros([fn, config.TAR_H, config.TAR_W, config.NUM_KP]).astype('i')
    mo_x = np.zeros([fn, config.TAR_H, config.TAR_W, config.NUM_EDGE]).astype('i')
    mo_y = np.zeros([fn, config.TAR_H, config.TAR_W, config.NUM_EDGE]).astype('i')
    for f_i in range(fn):
        for kp in kp_list[f_i]:
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
                hm[f_i, y_i, x_i, kp_i] = 1
                so_x[f_i, y_i, x_i, kp_i] = -xm
                so_y[f_i, y_i, x_i, kp_i] = -ym

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
                mo_x[f_i, y_i, x_i, e_i] = -xm + dx
                mo_y[f_i, y_i, x_i, e_i] = -ym + dy
    return (frames, hm, so_x, so_y, mo_x, mo_y, len(frames))

