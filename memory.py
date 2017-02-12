
# coding: utf-8

# In[51]:

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import imageio
import numpy as np


# In[104]:

class Memory:
    def __init__(self, height, width, num_objects, num_blocks, num_memories, num_gates=4):
        # Init and Reg
        initializer = tf.truncated_normal_initializer(0.0, 0.1, dtype=tf.float32)
        regularizer = l2_regularizer(.0005)
        
        # Good Ol' Get Variable
        def get_variable(name, shape):
            return tf.get_variable(name, shape, initializer=initializer,
                                   regularizer=regularizer, dtype=tf.float32)
        
        self._translate = get_variable("translate", 
                                       [num_gates, height, width, num_memories])
        
        
        # Gate Shapes and Sizes (could have used reduce)        
        self._W_shape = [num_objects, height, width, num_blocks]
        self._W_size = num_objects*height*width*num_blocks
        
        self._R_shape = [num_objects, num_blocks, num_blocks]
        self._R_size = num_objects*num_blocks*num_blocks + self._W_size
        
        self._b_shape = [num_objects, num_blocks]
        self._b_size = num_objects*num_blocks + self._R_size
        
        self._frame_shape = [num_objects, height, width]
        
        self._gate_size = self._W_size + self._R_size + self._b_size
        
        self._memory = get_variable("memory",
                                   [num_gates, num_memories, self._gate_size])
        
        self.shapes = [self._W_shape, self._R_shape, self._b_shape]
        self.sizes = [self._W_size, self._R_size, self._b_size]
        
    def __call__(self, frame):

        # Assigns weights for each gate for each memory, for each object
        mem_weights = tf.einsum("ijkm,jkl->im", self._translate, frame[0])
        
        # Get the weights for the gates 
        these_memories = tf.einsum("ijk,ij->ik", self._memory, mem_weights)
        
        return these_memories


# In[113]:

class R3Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_blocks, num_objects, num_memories, score_fr_shp):
        self._num_blocks = num_blocks
        self._num_objects = num_objects
        self.score_fr_shp = score_fr_shp
        self.memory = Memory(score_fr_shp[1].value, score_fr_shp[2].value, num_objects, num_blocks, num_memories)
        self._W_size, self._R_size, self._b_size = self.memory.sizes
        self._W_shape, self._R_shape, self._b_shape = self.memory.shapes
        
    @property
    def input_size(self):
        return self._num_blocks*self._num_objects
    
    @property
    def output_size(self):
        return self._num_blocks*self._num_objects
    
    @property
    def state_size(self):
        return 2*self._num_blocks*self._num_objects
    
    def __call__(self, inputs, state, scope=None):
        
        initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

        def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

        # LSTM Setup
        with tf.variable_scope("lstm_setup"):
            num_objects, num_blocks = self._num_objects, self._num_blocks
            score_fr = tf.reshape(inputs, self.score_fr_shp) 
            height, width = score_fr.get_shape()[1:3]
            g = h = tf.tanh
            
            c_prev, y_prev = tf.split(1, 2, state)
            y_prev = tf.reshape(y_prev, [num_objects, num_blocks])
            c_prev = tf.reshape(c_prev, [num_objects, num_blocks])
            
            def ex(m1, m2):
                return tf.einsum('bjki,ijkm-> im', m1, m2)

            def ey(m1, m2):
                return tf.einsum('ij,ijl->il', m1, m2)
        
        # Get Memory for this Frame
        gates = self.memory(score_fr)
                
        # LSTM Forget
        with tf.variable_scope("forget"):
            W_f = tf.reshape(gates[0, :self._W_size], self._W_shape)
            R_f = tf.reshape(gates[0, self._W_size:self._R_size], self._R_shape)
            b_f = tf.reshape(gates[0, self._R_size:self._b_size], self._b_shape)
            f = tf.sigmoid(ex(score_fr, W_f) + ey(y_prev, R_f) + b_f)
        
        # LSTM Input Layer
        with tf.variable_scope("input"):
            W_i = tf.reshape(gates[1, :self._W_size], self._W_shape)
            R_i = tf.reshape(gates[1, self._W_size:self._R_size], self._R_shape)
            b_i = tf.reshape(gates[1, self._R_size:self._b_size], self._b_shape)
            i = tf.sigmoid(ex(score_fr, W_i) + ey(y_prev, R_i) + b_i)
            
        # LSTM Candidate Values (Block Input)
        with tf.variable_scope("candidate"):
            W_z = tf.reshape(gates[2, :self._W_size], self._W_shape)
            R_z = tf.reshape(gates[2, self._W_size:self._R_size], self._R_shape)
            b_z = tf.reshape(gates[2, self._R_size:self._b_size], self._b_shape)
            z = g(ex(score_fr, W_z) + ey(y_prev, R_z) + b_z)
            
        # LSTM Output Gate
        with tf.variable_scope("output"):
            W_o = tf.reshape(gates[3, :self._W_size], self._W_shape)
            R_o = tf.reshape(gates[3, self._W_size:self._R_size], self._R_shape)
            b_o = tf.reshape(gates[3, self._R_size:self._b_size], self._b_shape)
            o = tf.sigmoid(ex(score_fr, W_o) + ey(y_prev, R_o) + b_o)
        
        # LSTM New State
        with tf.variable_scope("new_state"):
            
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            y = tf.mul(h(c), o)
           
        return tf.reshape(y, [1, -1]), tf.concat(1, [tf.reshape(c, [1, -1]), tf.reshape(y, [1, -1])])


# In[114]:

# vid = imageio.get_reader("./vids/Street.mp4", 'FFMPEG')
# full_width, full_height = vid.get_meta_data()['size']
# iter_vid = vid.iter_data()


full_height, full_width = 1080, 1920
num_objects = 20
num_blocks = 100
num_memories = 100
channels=3
batch = 1


# In[115]:

tf.reset_default_graph()
initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)


def get_variable(name, shape):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def conv2d(x, filter_name, filter_shape):
    w = get_variable(filter_name, filter_shape)
    return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME"))


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def deconv2d(x, output_shp, filter_name, filter_shape, stride):
    w = tf.get_variable(filter_name, filter_shape)
    return tf.nn.conv2d_transpose(x, w, output_shp, [1, stride, stride, 1])


def fil_sh(p_fshp, fshp, sz=3):
    return [sz, sz, p_fshp, fshp]

# Inputs
with tf.variable_scope("inputs"):
    frame = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])
    correct = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])

# First Convolution
with tf.variable_scope("conv1"):
    fshp = full_width/2/2/2/2
    conv1_1 = conv2d(frame, "filter1_1", fil_sh(channels, fshp))
    conv1_2 = conv2d(conv1_1, "filter1_2", fil_sh(fshp, fshp))
    pool1 = maxPool(conv1_2)

# Second Convolution
with tf.variable_scope("conv2"):
    p_fshp = fshp
    fshp *= 2
    conv2_1 = conv2d(pool1, "filter2_1", fil_sh(p_fshp, fshp))
    conv2_2 = conv2d(conv2_1, "filter2_2", fil_sh(fshp, fshp))
    pool2 = maxPool(conv2_2)

# Third Convolution
with tf.variable_scope("conv3"):
    p_fshp = fshp
    fshp *= 2
    fl_fshp3 = fshp
    conv3_1 = conv2d(pool2, "filter3_1", fil_sh(p_fshp, fshp))
    conv3_2 = conv2d(conv3_1, "filter3_2", fil_sh(fshp, fshp))
    conv3_3 = conv2d(conv3_2, "filter3_3", fil_sh(fshp, fshp))
    pool3 = maxPool(conv3_3)

# Fourth Convolution
with tf.variable_scope("conv4"):
    p_fshp = fshp
    fshp *= 2
    fl_fshp4 = fshp
    conv4_1 = conv2d(pool3, "filter4_1", fil_sh(p_fshp, fshp))
    conv4_2 = conv2d(conv4_1, "filter4_2", fil_sh(fshp, fshp))
    conv4_3 = conv2d(conv4_2, "filter4_3", fil_sh(fshp, fshp))
    pool4 = maxPool(conv4_3)

# Fifth Convolution
with tf.variable_scope("conv5"):
    conv5_1 = conv2d(pool4, "filter5_1", fil_sh(fshp, fshp))
    conv5_2 = conv2d(conv5_1, "filter5_2", fil_sh(fshp, fshp))
    conv5_3 = conv2d(conv5_2, "filter5_3", fil_sh(fshp, fshp))
    pool5 = maxPool(conv5_3)

# Fully Connected Layers
with tf.variable_scope("fcn"):
    p_fshp = fshp
    fshp = full_width
    fcn6 = tf.nn.dropout(conv2d(pool5, "filter6", fil_sh(p_fshp, fshp, 7)), .5)
    fcn7 = tf.nn.dropout(conv2d(fcn6, "filter7", fil_sh(fshp, fshp, 1)), .5)

# Score
with tf.variable_scope("score"):
    p_fshp = fshp
    fshp = num_objects
    score_fr = conv2d(fcn7, "filter_scr", fil_sh(p_fshp, fshp, 1))
    inputs = tf.reshape(score_fr, [1, 1, -1])  
    score_fr_shp = score_fr.get_shape()

cell = R3Cell(num_blocks, num_objects, num_memories, score_fr_shp)


output, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, 
                            dtype=tf.float32)
output_rshp = tf.reshape(output, [num_objects, num_blocks])
W_nst = get_variable("W_nst", [num_blocks, *score_fr_shp[1:]])
nst = tf.expand_dims(tf.einsum('ij,jkli->kli', output_rshp, W_nst), axis=0)

# Upsampling
with tf.variable_scope("upsampling"):
    output_shp = tf.stack([*pool4.get_shape()[:3], tf.Dimension(num_objects)])
    upscore2 = deconv2d(nst, output_shp, "defilter2", fil_sh(fshp, fshp, 4), 2)

with tf.variable_scope("upsampling_p4"):
    score_pool4 = conv2d(pool4, "filter_p4", fil_sh(fl_fshp4, fshp, 1))
    fuse_pool4 = tf.add(upscore2, score_pool4)
    output_shp = tf.stack([*pool3.get_shape()[:3], tf.Dimension(num_objects)])
    upscore_pool4 = deconv2d(fuse_pool4, output_shp, "defilter4", fil_sh(fshp, fshp, 4), 2)

with tf.variable_scope("upsampling_p3"):
    score_pool3 = conv2d(pool3, "filter_p3", fil_sh(fl_fshp3, fshp, 1))
    fuse_pool3 = tf.add(upscore_pool4, score_pool3)
    output_shp = tf.stack([1, full_height, full_width, channels])
    upscore8 = deconv2d(fuse_pool3, output_shp, "defilter8", fil_sh(channels, fshp, 16), 8)

session = tf.Session()
loss = tf.reduce_mean(tf.square(tf.sub(second, upscore8)))
optimizer = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('./train', session.graph)

session.run(tf.global_variables())
