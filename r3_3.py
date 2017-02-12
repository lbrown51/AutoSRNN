import imageio
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

class R3Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_blocks, num_objects, score_fr_shp):
        self._num_blocks = num_blocks
        self._num_objects = num_objects
        self.score_fr_shp = score_fr_shp

    @property
    def input_size(self):
        return self._num_blocks * self._num_objects

    @property
    def output_size(self):
        return self._num_blocks * self._num_objects

    @property
    def state_size(self):
        return 2 * self._num_blocks * self._num_objects

    def __call__(self, inputs, state, scope=None):
        # Initialization and Regularization (Self-explanatory)
        initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)
        regularizer = l2_regularizer(.0005)

        # Ensures that at call time, if the variables exist
        # We don't recreate them.
        def get_variable(name, shape):
            return tf.get_variable(name, shape, initializer=initializer,
                                   regularizer=regularizer, dtype=tf.float32)

        # LSTM Setup
        with tf.variable_scope("lstm_setup"):
            num_objects, num_blocks = self._num_objects, self._num_blocks
            score_fr = tf.reshape(inputs, self.score_fr_shp)
            height, width = score_fr.get_shape()[1:3]

            g = h = tf.tanh
            c_prev, y_prev = tf.split(1, 2, state)
            y_prev = tf.reshape(y_prev, [num_objects, num_blocks])
            c_prev = tf.reshape(c_prev, [num_objects, num_blocks])

            # Einstein Summations for Higher Rank LSTM Operations
            def ex(m1, m2):
                return tf.einsum('bjki,ijkm-> im', m1, m2)

            def ey(m1, m2):
                return tf.einsum('ij,ijl->il', m1, m2)

        # LSTM Forget
        with tf.variable_scope("forget"):
            W_f = get_variable("W_f", [self._num_objects, height, width, num_blocks])
            R_f = get_variable("R_f", [num_objects, num_blocks, num_blocks])
            b_f = get_variable("b_f", [num_objects, num_blocks])
            f = tf.sigmoid(ex(score_fr, W_f) + ey(y_prev, R_f) + b_f)

        # LSTM Input Layer
        with tf.variable_scope("input"):
            W_i = get_variable("W_i", [num_objects, height, width, num_blocks])
            R_i = get_variable("R_i", [num_objects, num_blocks, num_blocks])
            b_i = get_variable("b_i", [num_objects, num_blocks])
            i = tf.sigmoid(ex(score_fr, W_i) + ey(y_prev, R_i) + b_i)

        # LSTM Candidate Values (Block Input)
        with tf.variable_scope("candidate"):
            W_z = get_variable("W_z", [num_objects, height, width, num_blocks])
            R_z = get_variable("R_z", [num_objects, num_blocks, num_blocks])
            b_z = get_variable("b_z", [num_objects, num_blocks])
            z = g(ex(score_fr, W_z) + ey(y_prev, R_z) + b_z)

        # LSTM Output Gate
        with tf.variable_scope("output"):
            W_o = get_variable("W_o", [num_objects, height, width, num_blocks])
            R_o = get_variable("R_o", [num_objects, num_blocks, num_blocks])
            b_o = get_variable("b_o", [num_objects, num_blocks])
            o = tf.sigmoid(ex(score_fr, W_o) + ey(y_prev, R_o) + b_o)

        # LSTM New State
        with tf.variable_scope("new_state"):
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            y = tf.mul(h(c), o)

        return tf.reshape(y, [1, -1]), tf.concat(1, [tf.reshape(c, [1, -1]), tf.reshape(y, [1, -1])])


class R3:
    def __init__(self, full_height, full_width, num_objects, num_blocks, batch=1, channels=3):

        # Initialization and Regularization (Self-explanatory)
        initializer = tf.truncated_normal_initializer(0.0, 0.1, dtype=tf.float32)
        regularizer = l2_regularizer(.0005)

        # We don't need to always use this, but it helps us stay consistent
        def get_variable(name, shape):
            return tf.get_variable(name, shape, initializer=initializer,
                                   regularizer=regularizer, dtype=tf.float32)

        # Returns a 2d Convolution Operation after Initializing Weights
        # The name input, as labelled is actually for the filter rather than
        # the convolution operation itself.
        def conv2d(x, filter_name, filter_shape):
            w = get_variable(filter_name, filter_shape)
            return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME"))

        # Self Explanatory
        def maxPool(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Returns the Deconvolution (or Convolution Transpose) Operation
        def deconv2d(x, output_shp, filter_name, filter_shape, stride):
            w = tf.get_variable(filter_name, filter_shape)
            return tf.nn.conv2d_transpose(x, w, output_shp, [1, stride, stride, 1])

        # Returns the filter shapes for conv2d and deconv2d
        def fil_sh(p_fshp, fshp, sz=3):
            return [sz, sz, p_fshp, fshp]

        # Inputs
        with tf.variable_scope("inputs"):
            frame = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])
            correct = tf.placeholder(tf.float32, [batch, full_height, full_width, channels])

        # First Convolution
        with tf.variable_scope("conv1"):
            fshp = full_width / 2 / 2 / 2 / 2
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

        # LSTM Steps
        # Dynamic RNN can take variable lengths so we'll use that
        # eventually.
        cell = R3Cell(num_blocks, num_objects, score_fr_shp)
        output, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                                          dtype=tf.float32)

        # Take LSTM Output and Transorm->Multiply to shape of Score_fr
        # Equivalent to using a fully connected layer to get the
        # dimensions we want.
        output_rshp = tf.reshape(output, [num_objects, num_blocks])
        W_nst = get_variable("W_nst", [num_blocks, *score_fr_shp[1:]])
        nst = tf.expand_dims(tf.einsum('ij,jkli->kli', output_rshp, W_nst), axis=0)

        # Upsampling -- We are gonna turn it into a picture again
        with tf.variable_scope("upsampling"):
            output_shp = tf.stack([*pool4.get_shape()[:3], tf.Dimension(num_objects)])
            upscore2 = deconv2d(nst, output_shp, "defilter2", fil_sh(fshp, fshp, 4), 2)

        # Fuse With Pool4 and Upsample
        with tf.variable_scope("upsampling_p4"):
            score_pool4 = conv2d(pool4, "filter_p4", fil_sh(fl_fshp4, fshp, 1))
            fuse_pool4 = tf.add(upscore2, score_pool4)
            output_shp = tf.stack([*pool3.get_shape()[:3], tf.Dimension(num_objects)])
            upscore_pool4 = deconv2d(fuse_pool4, output_shp, "defilter4", fil_sh(fshp, fshp, 4), 2)

        # Fuse with Pool3 and Upsample
        with tf.variable_scope("upsampling_p3"):
            score_pool3 = conv2d(pool3, "filter_p3", fil_sh(fl_fshp3, fshp, 1))
            fuse_pool3 = tf.add(upscore_pool4, score_pool3)
            output_shp = tf.stack([1, full_height, full_width, channels])
            upscore8 = deconv2d(fuse_pool3, output_shp, "defilter8", fil_sh(channels, fshp, 16), 8)

        # Train, Loss, and Optimizer
        with tf.variable_scope("train"):
            loss = tf.reduce_mean(tf.square(tf.sub(correct, upscore8)))
            optimizer = tf.train.AdamOptimizer()
            self.train_step = optimizer.minimize(loss)

