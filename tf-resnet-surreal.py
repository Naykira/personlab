
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
import tensorflow as tf
import numpy as np
import surreal_single, config
import functools, operator, copy
import tensorflow.contrib.slim as slim
from nets import resnet_v2
from nets.mobilenet import mobilenet_v2
from tensor_info2 import INPUT_TENSOR_INFO, OUTPUT_TENSOR_INFO
tf.reset_default_graph()


# In[10]:


types = tuple(t['type'] for t in INPUT_TENSOR_INFO)
config.BATCH_SIZE = 10
input_tensors = tf.data.Dataset.from_generator(surreal_single.load, types)                                .batch(config.BATCH_SIZE)                                .prefetch(config.PREFETCH_SIZE)                                .make_one_shot_iterator()                                .get_next()
tensors = {}
for tensor, info in zip(input_tensors, INPUT_TENSOR_INFO):
    tensor.set_shape(info['shape'])
    tensors[info['name']] = tensor
print(tensors)


# In[11]:


config.STRIDE = 16

MD_H = int((config.TAR_H-1)//config.STRIDE)+1
MD_W = int((config.TAR_W-1)//config.STRIDE)+1

DEPTH = [ti['shape'][-1] for ti in OUTPUT_TENSOR_INFO]
RESULT_SHAPE = (config.BATCH_SIZE, MD_H, MD_W, sum(DEPTH))
RESULT_SIZE = functools.reduce(operator.mul, RESULT_SHAPE[1:])
OUTPUT_SHAPE = (config.BATCH_SIZE, config.TAR_H, config.TAR_W, sum(DEPTH))
OUTPUT_SIZE = functools.reduce(operator.mul, OUTPUT_SHAPE[1:])


# In[12]:


def bilinear(indices):
    oy = tf.clip_by_value(indices[1], 0, MD_H-1)
    ox = tf.clip_by_value(indices[2], 0, MD_W-1)
    iy = [tf.floor(oy), tf.ceil(oy + 1e-9)]
    ix = [tf.floor(ox), tf.ceil(ox + 1e-9)]
    idx_p = []
    for y in iy:
        for x in ix:
            indices[1] = y
            indices[2] = x
            idx = tf.cast(tf.stack(indices, axis=-1), tf.int32)
            p = (1 - tf.abs(y - oy)) * (1 - tf.abs(x - ox))
            idx_p.append((idx, p))
    return idx_p

def gather_bilinear(params, indices):
    idx_p = bilinear(indices)
    res = []
    for idx, p in idx_p:
        r = tf.gather_nd(params, idx)
        res.append(r * p)
    return tf.add_n(res)

def scatter_bilinear(params, indices, shape):
    idx_p = bilinear(indices)
    res = []
    for idx, p in idx_p:
        r = tf.scatter_nd(idx, params, shape)
        if len(r.shape) > len(p.shape):
            p = tf.expand_dims(p, axis=-1)
        res.append(r * p)
    return tf.add_n(res)


# In[13]:


def resize(tensor):
    return tf.image.resize_images(
        tensor,
        (config.TAR_H, config.TAR_W),
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=True)

with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, end_points = resnet_v2.resnet_v2_101(tensors['image'], output_stride=config.STRIDE)
    model_output = end_points['resnet_v2_101/block4']
    print('model_output', model_output)
    hm_pred = slim.conv2d(model_output, config.NUM_KP, [1, 1], activation_fn=tf.sigmoid)
    so_x_pred = slim.conv2d(model_output, config.NUM_KP, [1, 1], activation_fn=None)
    so_y_pred = slim.conv2d(model_output, config.NUM_KP, [1, 1], activation_fn=None)
    mo_x_pred = slim.conv2d(model_output, config.NUM_EDGE, [1, 1], activation_fn=None)
    mo_y_pred = slim.conv2d(model_output, config.NUM_EDGE, [1, 1], activation_fn=None)

    print(hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred)
    b, y, x, i = np.mgrid[:config.BATCH_SIZE, :MD_H, :MD_W, :config.NUM_EDGE]
    for _ in range(config.NUM_RECURRENT):
        mo_p = [b, y+mo_y_pred, x+mo_x_pred, i]
        mo_x_pred = gather_bilinear(so_x_pred, mo_p) + mo_x_pred
        mo_y_pred = gather_bilinear(so_y_pred, mo_p) + mo_y_pred
    print(hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred)
    hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred = [resize(x) for x in [hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred]]
    print(hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred)
    so_x_pred, so_y_pred, mo_x_pred, mo_y_pred = [x * config.STRIDE for x in [so_x_pred, so_y_pred, mo_x_pred, mo_y_pred]]
    print(hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred)


# In[14]:


hm_loss = - tf.reduce_mean(tensors['hm'] * tf.log(hm_pred + 1e-9) + (1 - tensors['hm']) * tf.log(1 - hm_pred + 1e-9))
so_loss = tf.abs(tensors['so_x'] - so_x_pred) / config.RADIUS + tf.abs(tensors['so_y'] - so_y_pred) / config.RADIUS
mo_loss = tf.abs(tensors['mo_x'] - mo_x_pred) / config.RADIUS + tf.abs(tensors['mo_y'] - mo_y_pred) / config.RADIUS

disc_only = tf.cast(tensors['hm'], tf.float32)
disc_size = tf.reduce_sum(disc_only, axis=[1, 2]) + 1e-9
so_loss = tf.reduce_mean(tf.reduce_sum(so_loss * disc_only, axis=[1, 2]) / disc_size)

disc_only = tf.cast(tf.gather(tensors['hm'], config.EDGES[:, 0], axis=-1), tf.float32)
disc_size = tf.reduce_sum(disc_only, axis=[1, 2]) + 1e-9
mo_loss = tf.reduce_mean(tf.reduce_sum(mo_loss * disc_only, axis=[1, 2]) / disc_size)


# In[15]:


total_loss = hm_loss * 4.0 + so_loss * 1.0 + mo_loss * 0.5


# In[16]:


tf.summary.scalar('losses/hm_loss', hm_loss)
tf.summary.scalar('losses/so_loss', so_loss)
tf.summary.scalar('losses/mo_loss', mo_loss)
tf.summary.scalar('losses/total_loss', total_loss)


# In[17]:


tf.summary.histogram("pred_dist/resnet", model_output)
tf.summary.histogram("pred_dist/heatmap", hm_pred)
tf.summary.histogram("pred_dist/short_off_x", so_x_pred)
tf.summary.histogram("pred_dist/short_off_y", so_y_pred)
tf.summary.histogram("pred_dist/mid_off_x", mo_x_pred)
tf.summary.histogram("pred_dist/mid_off_y", mo_y_pred)

tf.summary.histogram("true_dist/heatmap", tensors['hm'])
tf.summary.histogram("true_dist/short_off_x", tensors['so_x'])
tf.summary.histogram("true_dist/short_off_y", tensors['so_y'])
tf.summary.histogram("true_dist/mid_off_x", tensors['mo_x'])
tf.summary.histogram("true_dist/mid_off_y", tensors['mo_y'])

optimizer = tf.train.AdamOptimizer()
train_op = slim.learning.create_train_op(total_loss, optimizer)

checkpoint_path = 'resnet/resnet_v2_101.ckpt'
variables = slim.get_model_variables()
restore_map = {}
for v in variables:
    if not v.name.startswith('resnet'):
        continue
    org_name = v.name.split(':')[0]
    restore_map[org_name] = v
    print(org_name, ':', v.name)
init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, restore_map)


# In[18]:


import time, os, shutil
#log_dir = 'logs/log_' + str(time.time())[-5:]
log_dir = 'logs/res_log_test'
shutil.rmtree(log_dir)
os.mkdir(log_dir)


# In[ ]:


def InitAssignFn(sess):
    sess.run(init_assign_op, init_feed_dict)
tf.contrib.slim.learning.train(train_op,
                               '/home/ubuntu/personlab/'+log_dir,
                               init_fn=InitAssignFn,
                               log_every_n_steps=100,
                               save_summaries_secs=30,
                              )


# In[ ]:


from tensorflow.python.tools import inspect_checkpoint
inspect_checkpoint.print_tensors_in_checkpoint_file(checkpoint_path, tensor_name='', all_tensors=False, all_tensor_names=True)


# In[ ]:



saver = tf.train.Saver()
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        checkpoint_path = 'logs/res_log_test/model.ckpt-4575'
        saver.restore(sess, checkpoint_path)
        m_out, img, hm = sess.run([model_output, tensors['image'], hm_out])


# In[ ]:


import cv2
from matplotlib import pyplot as plt

def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.:
        out = out / 255.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out
b_i = 0
plt.figure()
plt.imshow(overlay(img[b_i, ...], np.max(hm[b_i, ...], axis=-1), alpha=0.7))


# In[ ]:


m_out


# In[ ]:


config.EDGES[:, ::-1]

