
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
import tensorflow as tf
import numpy as np
import surreal, config
import functools, operator, copy
import tensorflow.contrib.slim as slim
from nets.resnet_v2 import resnet_v2_101
from nets.mobilenet import mobilenet_v2

tf.reset_default_graph()
'''
for x in surreal.load():
    print(x)
    break
'''


# In[2]:


TENSOR_INFO = [
    {
        'name': 'image',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, 3),
        'type': tf.float32,
    },{
        'name': 'hm',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_x',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'so_y',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, config.NUM_KP),
        'type': tf.float32,
    },{
        'name': 'mo_x',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, config.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'mo_y',
        'shape': (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, config.NUM_EDGE),
        'type': tf.float32,
    },{
        'name': 'seq_len',
        'shape': (config.BATCH_SIZE,),
        'type': tf.int32,
    }
]

types = tuple(t['type'] for t in TENSOR_INFO)
input_tensors = tf.data.Dataset.from_generator(surreal.load, types)                                .batch(config.BATCH_SIZE)                                .prefetch(config.PREFETCH_SIZE)                                .make_one_shot_iterator()                                .get_next()
tensors = {}
for tensor, info in zip(input_tensors, TENSOR_INFO):
    tensor.set_shape(info['shape'])
    tensors[info['name']] = tensor


# In[3]:


config.STRIDE = 16


MD_H = int(config.TAR_H/config.STRIDE)
MD_W = int(config.TAR_W/config.STRIDE)

DEPTH = [ti['shape'][-1] for ti in TENSOR_INFO[1:-1]]
RESULT_SHAPE = (config.BATCH_SIZE, MD_H, MD_W, sum(DEPTH))
RESULT_SIZE = functools.reduce(operator.mul, RESULT_SHAPE[1:])
OUTPUT_SHAPE = (config.BATCH_SIZE, config.TAR_H, config.TAR_W, sum(DEPTH))
OUTPUT_SIZE = functools.reduce(operator.mul, OUTPUT_SHAPE[1:])


class TestCell(tf.contrib.rnn.RNNCell):
    def __init__(self, is_training):
        super().__init__(self)
        self.is_training = is_training
        
    def __call__(self, frame_tensor, state):
        mbnet2_output, _ = mobilenet_v2.mobilenet_base(frame_tensor, output_stride=config.STRIDE)
        
        # parse expectation from previous frame
        state = tf.reshape(state, RESULT_SHAPE)
        hm_prev, so_x_prev, so_y_prev, mo_x_prev, mo_y_prev = tf.split(state, DEPTH, axis=-1)
        
        # prediction of current frame
        hm_pred = slim.conv2d(mbnet2_output, config.NUM_KP, [1, 1])
        #hm_pred = slim.batch_norm(hm_pred, is_training=self.is_training)
        hm_pred = hm_pred + hm_prev
        
        so_x_pred = slim.conv2d(mbnet2_output, config.NUM_KP, [1, 1])
        #so_x_pred = slim.batch_norm(so_x_pred, is_training=self.is_training)
        so_x_pred = so_x_pred + so_x_prev
        
        so_y_pred = slim.conv2d(mbnet2_output, config.NUM_KP, [1, 1])
        #so_y_pred = slim.batch_norm(so_y_pred, is_training=self.is_training)
        so_y_pred = so_y_pred + so_y_prev
        
        mo_x_pred = slim.conv2d(mbnet2_output, config.NUM_EDGE, [1, 1])
        #mo_x_pred = slim.batch_norm(mo_x_pred, is_training=self.is_training)
        mo_x_pred = mo_x_pred + mo_x_prev
        
        mo_y_pred = slim.conv2d(mbnet2_output, config.NUM_EDGE, [1, 1])
        #mo_y_pred = slim.batch_norm(mo_y_pred, is_training=self.is_training)
        mo_y_pred = mo_y_pred + mo_y_prev
        
        # expect point in next frame
        mv_x_pred = slim.conv2d(mbnet2_output, 1, [1, 1])
        mv_y_pred = slim.conv2d(mbnet2_output, 1, [1, 1])
        
        # construct expectation data
        cur_x = np.tile(np.arange(MD_W), [config.BATCH_SIZE, MD_H, 1, 1]).transpose([0, 1, 3, 2])
        cur_y = np.tile(np.arange(MD_H), [config.BATCH_SIZE, MD_W, 1, 1]).transpose([0, 3, 1, 2])
        mvp_b = np.tile(np.arange(config.BATCH_SIZE), [MD_H, MD_W, 1, 1]).transpose([3, 0, 1, 2])
        mvp_x = tf.cast(tf.clip_by_value(tf.round(cur_x + mv_x_pred), 0, MD_W-1), 'int32')
        mvp_y = tf.cast(tf.clip_by_value(tf.round(cur_y + mv_y_pred), 0, MD_H-1), 'int32')
        mvp = tf.concat([mvp_b, mvp_x, mvp_y], axis=-1)
        hm_expect = tf.scatter_nd(mvp, hm_pred, hm_pred.shape)
        so_x_expect = tf.scatter_nd(mvp, so_x_pred, so_x_pred.shape)
        print(mvp, so_y_pred)
        so_y_expect = tf.scatter_nd(mvp, so_y_pred, so_y_pred.shape)
        
        mo_end_b = np.tile(np.arange(config.BATCH_SIZE), [MD_H, MD_W, config.NUM_EDGE, 1]).transpose([3, 0, 1, 2])
        mo_end_x = tf.cast(tf.clip_by_value(tf.round(cur_x + mo_x_pred), 0, MD_W-1), 'int32')
        mo_end_y = tf.cast(tf.clip_by_value(tf.round(cur_y + mo_y_pred), 0, MD_H-1), 'int32')
        mo_end = tf.stack([mo_end_b, mo_end_x, mo_end_y], axis=-1)
        mo_x_expect_cp = tf.squeeze(tf.gather_nd(mv_x_pred, mo_end), axis=[-1]) + mo_x_pred - mv_x_pred
        mo_y_expect_cp = tf.squeeze(tf.gather_nd(mv_y_pred, mo_end), axis=[-1]) + mo_y_pred - mv_y_pred
        mo_x_expect = tf.scatter_nd(mvp, mo_x_expect_cp, mo_x_pred.shape)
        mo_y_expect = tf.scatter_nd(mvp, mo_y_expect_cp, mo_y_pred.shape)
        
        next_state = tf.concat([hm_expect, so_x_expect, so_y_expect, mo_x_expect, mo_y_expect], axis=-1)
        next_state = tf.reshape(next_state, [config.BATCH_SIZE, RESULT_SIZE])

        output = tf.concat([hm_pred, so_x_pred, so_y_pred, mo_x_pred, mo_y_pred], axis=-1)
        output = tf.image.resize_images(
            output,
            (config.TAR_H, config.TAR_W),
            method=tf.image.ResizeMethod.BICUBIC,
            align_corners=True
        )
        output = tf.reshape(output, [config.BATCH_SIZE, OUTPUT_SIZE])
        
        return output, next_state

    @property
    def state_size(self):
        return RESULT_SIZE

    @property
    def output_size(self):
        return OUTPUT_SIZE


test_cell = TestCell(is_training=True)
#with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
pred_sum, _ = tf.nn.dynamic_rnn(test_cell, tensors['image'], sequence_length=tensors['seq_len'], dtype=tf.float32)


# In[4]:


pred_sum


# In[5]:


TOTAL_SHAPE = (config.BATCH_SIZE, config.MAX_FRAME_SIZE, config.TAR_H, config.TAR_W, sum(DEPTH))
pred_sum = tf.reshape(pred_sum, TOTAL_SHAPE)
hm_out, so_x_out, so_y_out, mo_x_out, mo_y_out = tf.split(pred_sum, DEPTH, axis=-1)


# In[6]:


tf.losses.softmax_cross_entropy(tensors['hm'], hm_out, weights=4.0)
tf.losses.absolute_difference(tensors['so_x'], so_x_out, weights=1.0 / config.RADIUS)
tf.losses.absolute_difference(tensors['so_y'], so_y_out, weights=1.0 / config.RADIUS)
tf.losses.absolute_difference(tensors['mo_x'], mo_x_out, weights=0.5 / config.RADIUS)
tf.losses.absolute_difference(tensors['mo_y'], mo_y_out, weights=0.5 / config.RADIUS)
tf.losses.get_losses()


# In[7]:


losses = tf.losses.get_losses()
for l in losses:
    tf.summary.scalar(l.name, l)
loss = tf.losses.get_total_loss()
tf.summary.scalar('losses/total_loss', loss)


# In[ ]:


tf.summary.scalar('val/hm_sum', tf.reduce_sum(hm_out))
tf.summary.scalar('val/so_sum', tf.abs(tf.reduce_sum(so_x_out)) + tf.abs(tf.reduce_sum(so_y_out)))
tf.summary.scalar('val/mo_sum', tf.abs(tf.reduce_sum(mo_x_out)) + tf.abs(tf.reduce_sum(mo_y_out)))
tf.summary.scalar('val/hm_true_sum', tf.reduce_sum(tensors['hm']))
tf.summary.scalar('val/so_true_sum', tf.abs(tf.reduce_sum(tensors['so_x'])) + tf.abs(tf.reduce_sum(tensors['so_y'])))
tf.summary.scalar('val/mo_true_sum', tf.abs(tf.reduce_sum(tensors['mo_x'])) + tf.abs(tf.reduce_sum(tensors['mo_y'])))
optimizer = tf.train.AdamOptimizer()
train_op = slim.learning.create_train_op(loss, optimizer)

checkpoint_path = 'mbnet/mobilenet_v2_1.0_224.ckpt'
variables = slim.get_model_variables()
restore_map = {}
for v in variables:
    if not v.name.startswith('rnn/MobilenetV2'):
        continue
    org_name = v.name[4:].split(':')[0]
    restore_map[org_name] = v
    print(org_name, ':', v.name)
init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, restore_map)


# In[8]:


import time, os
log_dir = 'logs/log_' + str(time.time())[-5:]
#log_dir = 'logs/log_04715'
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


# # TODO
# #### ADAPT IMAGENET PRETRAINED BASE [V]
# pretrained net 로딩시킴
# #### BATCH NORMALIZATION [V]
# 
# #### dataset
# 크기 맞춰서 빈만큼 채워넣기
# 큰 비디오 여러개로 자르기
# edge 정상데이터로 변경 [V]
# 
# ----------
# 
# 논문 훑으면서 빠진부분없나 체크
