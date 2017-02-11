import tensorflow as tf
from tensorflow.contrib import slim as slim


class R3Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_blocks, height, width, num_objects, channels=3):
        self._num_blocks, self._num_objects = num_blocks, num_objects
        self.height, self.width, self.channels = height, width, channels
        self.st = r3st(height, width, channels, num_objects)

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
            z = g(ex(self.st, W_z) + ey(y_prev, R_z) + b_z)
            i = tf.sigmoid(ex(self.st, W_i) + ey(y_prev, R_i) + b_i)
            f = tf.sigmoid(ex(self.st, W_f) + ey(y_prev, R_f) + b_f)
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            o = tf.sigmoid(ex(self.st, W_o) + ey(y_prev, R_o) + b_o)
            y = tf.mul(h(c), o)

            return y, tf.pack([c, y])


def r3st(height, width, channels, num_object_types):
        frame = tf.placeholder(tf.float32, [height, width, channels])
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, .01),
                        weights_regularizere=slim.l2_regularizer(.0005)):
            net = slim.repeat(frame, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            pool3 = slim.max_pool2d(net, 3, [2, 2], scope='pool3')
            net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.dropout(slim.conv2d(net, [7, 7], 4096), is_training=True, scope="fc1")
            net = slim.dropout(slim.conv2d(net, [1, 1], 4096), is_training=True, scope="fc2")
        net = slim.conv2d(net, [1, 1], num_object_types, scope="score_fr")
            net = slim.conv2d_transpose(net, [4, 4], num_object_types, scope='upscore2')
            fuse4 = tf.add(net, slim.conv2d(pool4, [1, 1], num_object_types, scope='score_pool4'))
            net = slim.conv2d_transpose(fuse4, [4, 4], num_object_types, scope='upscore_pool4')
            fuse3 = tf.add(net, slim.conv2d(pool3, [1, 1], num_object_types, scope='score_pool3'))
            net = slim.conv2d_transpose(fuse3, [16, 16], num_object_types, scope='upscore8')

            return net


def main():
    num_objects = 20
    dim_hidden = 10
    height = 4096
    width = 4096
    num_object_types = 20
    num_blocks = 100
    graph = tf.Graph()
    session = tf.Session()

    r3Cell = R3Cell(num_blocks, height, width, num_objects)
    init_state = r3Cell.zero_state(1)


