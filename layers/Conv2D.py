import tensorflow as tf
from keras import backend
from tensorflow.keras.layers import *


class Conv2D(Layer):
    def __init__(self, units=32, **kwargs):
        # TODO: Compute output shape
        # TODO: Initialise a kernel
        super(Conv2D, self).__init__(**kwargs)
        self.units = units
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=bias_init(shape=(self.units,), dtype='float32'), trainable=True)
        self.w = None  # w is initialized in build

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True,
            dtype='float32'
        )
        super(Conv2D, self).build(input_shape)

    def call(self, x, **kwargs):
        return None
        # TODO use backend.conv2d(x)
