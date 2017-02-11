import tensorflow as tf
import numpy as np

class FCN:
    def __init__(self, height, width, channels, num_image_types):
        def conv2d(x, filt):
            w = tf.random_normal(shape=filt)
            return tf.nn.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME"))

        def maxPool(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        def deconv2d(x, filt, stride):
            x_shape = tf.shape(x)
            out_shape = tf.pack([x_shape[0], x_shape[1], x_shape[2], num_image_types])
            w = tf.random_uniform(maxval=1, shape=filt)
            return tf.nn.conv2d_transpose(x, w, out_shape, [1, stride, stride, 1])

        def get_shape(num_conv, change=False, cdim=3, channels=channels):
            if num_conv == 1 and change:
                return [cdim, cdim, channels, 64*num_conv]
            elif change:
                return [cdim, cdim, 64*(int(num_conv/2)), 64*num_conv]
            else:
                return [cdim, cdim, 64*num_conv, 64*num_conv]

        self.image = tf.placeholder(tf.float32, [None, height, width, channels], name='image')

        # First Convolution
        conv1_1 = conv2d(self.image, get_shape(1, True))
        conv1_2 = conv2d(conv1_1, get_shape(1))
        pool1 = maxPool(conv1_2)

        # Second Convolution
        conv2_1 = conv2d(pool1, get_shape(2, True))
        conv2_2 = conv2d(conv2_1, get_shape(2))
        pool2 = maxPool(conv2_2)

        # Third Convolution
        conv3_1 = conv2d(pool2, get_shape(4, True))
        conv3_2 = conv2d(conv3_1, get_shape(4))
        conv3_3 = conv2d(conv3_2, get_shape(4))
        pool3 = maxPool(conv3_3)

        # Fourth Convolution
        conv4_1 = conv2d(pool3, get_shape(8, True))
        conv4_2 = conv2d(conv4_1, get_shape(8))
        conv4_3 = conv2d(conv4_2, get_shape(8))
        pool4 = maxPool(conv4_3)

        # Fifth Convolution
        conv5_1 = conv2d(pool4, get_shape(8))
        conv5_2 = conv2d(conv5_1, get_shape(8))
        conv5_3 = conv2d(conv5_2, get_shape(8))
        pool5 = maxPool(conv5_3)

        # Fully Connected Layers
        fcn6 = tf.nn.dropout(conv2d(pool5, [7, 7, 512, 4096]), .5)
        fcn7 = tf.nn.dropout(conv2d(fcn6, [1, 1, 4096, 4096]), .5)

        score_fr = conv2d(fcn7, [1, 1, 4096, num_image_types])
        upscore2 = deconv2d(score_fr, [4, 4, num_image_types, num_image_types], 2)

        score_pool4 = conv2d(pool4, [1, 1, 512, num_image_types])
        fuse_pool4 = tf.add(upscore2, score_pool4)
        upscore_pool4 = deconv2d(fuse_pool4, [4, 4, num_image_types, num_image_types], 2)

        score_pool3 = conv2d(pool3, [1, 1, 256, num_image_types])
        fuse_pool3 = tf.add(upscore_pool4, score_pool3)

        self.upscore8 = deconv2d(fuse_pool3, [16, 16, num_image_types, num_image_types], 8)


# Based on both https://github.com/fchollet/keras/issues/3354
# and https://github.com/MarvinTeichmann/tensorflow-fcn
# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/


class R3LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_objects, num_blocks, height, width, num_image_types, channels=3):
        self._num_objects, self._num_blocks, self._num_image_types = num_objects, num_blocks, num_image_types
        self.height, self.width, self.channels = height, width, channels
        self.fcn = FCN(height, width, channels, num_image_types)

    @property
    def input_size(self):
        return self._num_objects, self.height, self.width, self._num_blocks

    @property
    def output_size(self):
        return self._num_objects, self._num_blocks

    @property
    def state_size(self):
        return 2 * self._num_blocks * self._num_objects

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1, dtype=tf.float32)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

            c_prev, y_prev = tf.split(1, 2, state)
            y_prev = tf.reshape(y_prev, [self._num_objects, self._num_blocks])
            c_prev = tf.reshape(c_prev, [self._num_objects, self._num_blocks])
            W_z = get_variable("W_z", [self._num_objects, self.height, self.width, self.channels, self._num_blocks])
            W_i = get_variable("W_i", [self._num_objects, self.height, self.width, self.channels, self._num_blocks])
            W_f = get_variable("W_f", [self._num_objects, self.height, self.width, self.channels, self._num_blocks])
            W_o = get_variable("W_o", [self._num_objects, self.height, self.width, self.channels, self._num_blocks])

            R_z = get_variable("R_z", [self._num_objects, self._num_blocks, self._num_blocks])
            R_i = get_variable("R_i", [self._num_objects, self._num_blocks, self._num_blocks])
            R_f = get_variable("R_f", [self._num_objects, self._num_blocks, self._num_blocks])
            R_o = get_variable("R_o", [self._num_objects, self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [self._num_objects, self._num_blocks])
            b_i = get_variable("b_i", [self._num_objects, self._num_blocks])
            b_f = get_variable("b_f", [self._num_objects, self._num_blocks])
            b_o = get_variable("b_o", [self._num_objects, self._num_blocks])

            def ex(m1, m2):
                return tf.einsum('jkli,ijklm->im', m1, m2)

            def ey(m1, m2):
                return tf.einsum('ij,ijl->il', m1, m2)

            g = h = tf.tanh
            # m1 = tf.Variable(tf.random_normal([10,20,20,3]))
            # m2 = tf.Variable(tf.random_normal([10,20,20,3,100]))
            # b = tf.einsum('ijkl,ijklm->im', m1, m2)
            # image = tf.get_variable("test", [self.height, self.width, self.channels, self._num_objects])
            image = self.fcn.upscore8.eval({self.fcn.image: inputs})
            print(image)
            z = g(ex(image, W_z) + ey(y_prev, R_z) + b_z)
            i = tf.sigmoid(ex(image, W_i) + ey(y_prev, R_i) + b_i)
            f = tf.sigmoid(ex(image, W_f) + ey(y_prev, R_f) + b_f)
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            o = tf.sigmoid(ex(image, W_o) + ey(y_prev, R_o) + b_o)
            y = tf.mul(h(c), o)

            return y, tf.stack([c, y])

# base source
# Jim Fleming https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.psyd10grs
# https://github.com/fomorians/lstm-odyssey/tree/master/variants


num_objects = 10
dim_hidden = 10
height = 4096
width = 4096
num_image_types = 60
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
session = tf.InteractiveSession(config=config)
r3Cell = R3LSTMCell(num_objects, dim_hidden, height, width, num_image_types)
init_state = r3Cell.zero_state(1, tf.float32)
inputs = np.random.normal(size=(1, 4096, 4096, 3))
(cell_outputs, state) = r3Cell(inputs, init_state)
print("Hello")
print(cell_outputs)



