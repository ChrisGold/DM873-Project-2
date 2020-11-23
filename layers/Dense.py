import tensorflow as tf
from tensorflow.keras.layers import *


class Dense(Layer):
    def __init__(self, units=32, **kwargs):
        super(Dense, self).__init__(**kwargs)
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
        super(Dense, self).build(input_shape)

    def call(self, x, **kwargs):
        y = tf.matmul(x, self.weights) + self.bias
        return tf.nn.relu(y)
