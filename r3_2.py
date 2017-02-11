
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.contrib import slim as slim
import numpy as np

# In[4]:

num_objects = 10
num_blocks = 100
full_height, full_width, channels, batch = 240, 320, 3, 1

session = tf.Session()

# In[13]:

with tf.variable_scope("inputs"):
    state = tf.placeholder(tf.float32, [2, num_objects, num_blocks])
    frame = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])
    correct = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(frame, 2, slim.conv2d, 10, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 20, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 40, [3, 3], scope='conv3')
    pool3 = slim.max_pool2d(net, 3, [2, 2], scope='pool3')
    net = slim.repeat(pool3, 3, slim.conv2d, 80, [3, 3], scope='conv4')
    pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(pool4, 3, slim.conv2d, 80, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.dropout(slim.conv2d(net, 320, [7, 7]), is_training=True, scope="fc1")
    net = slim.dropout(slim.conv2d(net, 320, [1, 1]), is_training=True, scope="fc2")
    net = slim.conv2d(net, num_objects, [1, 1], scope="score_fr")

height, width = tf.shape(net).eval(session=session)[1:3]

with tf.variable_scope("r3Cell"):
    initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)


    def get_variable(name, shape):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


    W_z = get_variable("W_z", [num_objects, height, width, num_blocks])
    W_i = get_variable("W_i", [num_objects, height, width, num_blocks])
    W_f = get_variable("W_f", [num_objects, height, width, num_blocks])
    W_o = get_variable("W_o", [num_objects, height, width, num_blocks])
    W_nst = get_variable("W_nst", [num_blocks, height, width, num_objects])

    R_z = get_variable("R_z", [num_objects, num_blocks, num_blocks])
    R_i = get_variable("R_i", [num_objects, num_blocks, num_blocks])
    R_f = get_variable("R_f", [num_objects, num_blocks, num_blocks])
    R_o = get_variable("R_o", [num_objects, num_blocks, num_blocks])

    b_z = get_variable("b_z", [num_objects, num_blocks])
    b_i = get_variable("b_i", [num_objects, num_blocks])
    b_f = get_variable("b_f", [num_objects, num_blocks])
    b_o = get_variable("b_o", [num_objects, num_blocks])


    def ex(m1, m2):
        return tf.einsum('bjki,ijkm-> im', m1, m2)


    def ey(m1, m2):
        return tf.einsum('ij,ijl->il', m1, m2)


    c_prev, y_prev = tf.split(1, 2, state)
    y_prev = tf.reshape(y_prev, [num_objects, num_blocks])
    c_prev = tf.reshape(c_prev, [num_objects, num_blocks])
    g = h = tf.tanh
    z = g(ex(net, W_z) + ey(y_prev, R_z) + b_z)
    i = tf.sigmoid(ex(net, W_i) + ey(y_prev, R_i) + b_i)
    f = tf.sigmoid(ex(net, W_f) + ey(y_prev, R_f) + b_f)
    c = tf.mul(i, z) + tf.mul(f, c_prev)
    o = tf.sigmoid(ex(net, W_o) + ey(y_prev, R_o) + b_o)
    y = tf.mul(h(c), o)
    nst = tf.expand_dims(tf.einsum('ij,jkli->kli', y, W_nst), axis=0)

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.conv2d_transpose(nst, channels, [4, 4], stride=2, scope='upscore2')
    score_pool4 = slim.conv2d(pool4, channels, [1, 1], scope='score_pool4')
    dim_net4 = tf.shape(net)
    fuse4 = tf.add(net, score_pool4[:, :dim_net4[1], :dim_net4[2], :])

    net = slim.conv2d_transpose(fuse4, channels, [4, 4], stride=2, scope='upscore_pool4')
    score_pool3 = slim.conv2d(pool3, channels, [1, 1], scope='score_pool3')
    dim_net3 = tf.shape(net)
    fuse3 = tf.add(net, score_pool3[:, :dim_net3[1], :dim_net3[2], :])

    predict = slim.conv2d_transpose(fuse3, channels, [16, 16], stride=8, scope='upscore8')

dim_netp = tf.shape(predict)
loss = tf.reduce_sum(tf.abs(correct[:, :dim_netp[1], :dim_netp[2], :] - predict))
optimizer = tf.train.AdamOptimizer().minimize(loss)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('./train', session.graph)
session.run(tf.global_variables_initializer())


cstate = np.zeros([2, num_objects, num_blocks])
for i in range(12000):
    cframe = vids[i:i + 1]
    nframe = vids[i + 1:i + 2]
    if i % 100 == 0:
        _, closs, cstate = session.run([optimizer, loss, state],
                                       feed_dict={state: cstate,
                                                  frame: cframe, correct: nframe})
        print(closs / 100)
        closs = 0

    else:
        _, t_closs, cstate = session.run([optimizer, loss, state],
                                         feed_dict={state: cstate,
                                                    frame: cframe, correct: nframe})
        closs += t_closs
saver.save(session, './train.ckpt')

