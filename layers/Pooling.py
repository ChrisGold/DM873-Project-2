import tensorflow as tf
from tensorflow.keras.layers import *


class MaxPooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        super(MaxPooling, self).build(input_shape)

    def call(self, x, **kwargs):
        y = tf.nn.max_pool(x, self.pool_size, self.strides, self.padding)
        return y
