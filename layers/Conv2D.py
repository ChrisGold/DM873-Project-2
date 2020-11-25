import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import *

# pylint: disable=g-classes-have-attributes


class Conv2D(Layer):
    def __init__(self, filters=32, strides=1, **kwargs):
        # TODO: Compute output shape
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.bias_init = tf.zeros_initializer()
        kernel_size = (2, 2, self.filters)
        self.kernel_init = tf.keras.initializers.GlorotUniform()(kernel_size)
        self.strides = strides
        self.padding = 'valid'

    def build(self, input_shape):

        super(Conv2D, self).build(input_shape)
        self.bias = self.add_weight(initial_value=self.bias_init(shape=(self.filters,), dtype='float32'), trainable=True)
        self.kernel = self.add_weight(initial_value=self.kernel_init, trainable=True)

    def call(self, x, **kwargs):
        feature_maps = K.conv2d(x, self.kernel, self.strides, self.padding)
        return tf.nn.relu(feature_maps)

