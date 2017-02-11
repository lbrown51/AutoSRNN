import tensorflow as tf


class R3LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_objects, num_blocks, height, width):
        self._num_blocks = num_blocks
        self._num_objects = num_objects
        self.height, self.width = height, width

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
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, y_prev = tf.unstack(state)

            W_z = get_variable("W_z", [self._num_objects, self.height, self.width, self._num_blocks])
            W_i = get_variable("W_i", [self._num_objects, self.height, self.width, self._num_blocks])
            W_f = get_variable("W_f", [self._num_objects, self.height, self.width, self._num_blocks])
            W_o = get_variable("W_o", [self._num_objects, self.height, self.width, self._num_blocks])

            R_z = get_variable("R_z", [self._num_objects, self._num_blocks, self._num_blocks])
            R_i = get_variable("R_i", [self._num_objects, self._num_blocks, self._num_blocks])
            R_f = get_variable("R_f", [self._num_objects, self._num_blocks, self._num_blocks])
            R_o = get_variable("R_o", [self._num_objects, self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [self._num_objects, self._num_blocks])
            b_i = get_variable("b_i", [self._num_objects, self._num_blocks])
            b_f = get_variable("b_f", [self._num_objects, self._num_blocks])
            b_o = get_variable("b_o", [self._num_objects, self._num_blocks])

            def ex(m1, m2):
                return tf.einsum('ijkl,ijk->il', m1, m2)

            def ey(m1, m2):
                return tf.einsum('ij,ijl->il', m1, m2)

            g = h = tf.tanh
            # m1 = tf.Variable(tf.random_normal([10,20,20,3]))
            # m2 = tf.Variable(tf.random_normal([10,20,20,3,100]))
            # b = tf.einsum('ijkl,ijklm->im', m1, m2)
            z = g(ex(inputs, W_z) + ey(y_prev, R_z) + b_z)
            i = tf.sigmoid(ex(inputs, W_i) + ey(y_prev, R_i) + b_i)
            f = tf.sigmoid(ex(inputs, W_f) + ey(y_prev, R_f) + b_f)
            c = tf.mul(i, z) + tf.mul(f, c_prev)
            o = tf.sigmoid(ex(inputs, W_o) + ey(y_prev, R_o) + b_o)
            y = tf.mul(h(c), o)

            return y, tf.stack([c, y])

# base source
# Jim Fleming https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.psyd10grs


num_objects = 10
dim_hidden = 10
height = 20
width = 20
r3Cell = R3LSTMCell(num_objects, dim_hidden, height, width)
