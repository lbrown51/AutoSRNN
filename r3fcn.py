import tensorflow as tf


class FCN:
    def __init__(self, height, width, channels, num_image_types):
        def conv2d(x, filt):
            w = tf.random_normal(shape=filt)
            return tf.relu(tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME"))

        def maxPool(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        def deconv2d(x, filt, stride):
            w = tf.random_uniform(maxval=1, shape=filt)
            return tf.nn.conv2d_transpose(x, w, [1, stride, stride, 1])



        image = tf.placeholder(tf.float32, [None, height, width, channels])

        # First Convolution
        conv1_1 = conv2d(image, [3, 3, channels, 64])
        conv1_2 = conv2d(conv1_1, [3, 3, 64, 64])
        pool1 = maxPool(conv1_2)

        # Second Convolution
        conv2_1 = conv2d(pool1, [3, 3, 64, 128])
        conv2_2 = conv2d(conv2_1, [3, 3, 128, 128])
        pool2 = maxPool(conv2_2)

        # Third Convolution
        conv3_1 = conv2d(pool2, [3, 3, 128, 256])
        conv3_2 = conv2d(conv3_1, [3, 3, 256, 256])
        conv3_3 = conv2d(conv3_2, [3, 3, 256, 256])
        pool3 = maxPool(conv3_3)

        # Fourth Convolution
        conv4_1 = conv2d(pool3, [3, 3, 256, 512])
        conv4_2 = conv2d(conv4_1, [3, 3, 512, 512])
        conv4_3 = conv2d(conv4_2, [3, 3, 512, 512])
        pool4 = maxPool(conv4_3)

        # Fifth Convolution
        conv5_1 = conv2d(pool4, [3, 3, 512, 512])
        conv5_2 = conv2d(conv5_1, [3, 3, 512, 512])
        conv5_3 = conv2d(conv5_2, [3, 3, 512, 512])
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