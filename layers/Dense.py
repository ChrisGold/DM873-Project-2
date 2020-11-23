import tensorflow as tf
from tensorflow.keras.layers import *

class Dense(Layer):
    def __init__(self, units=32, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units
        bias_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=bias_init(shape=(self.units,), dtype='float32'), trainable=True)

    def build(self, input_shape):
        weights_init = tf.random_normal_initializer()(shape=(input_shape[-1], self.units), dtype='float32')

        self.w = self.add_weight(initial_value=weights_init, trainable=True)

        super(Dense, self).build(input_shape)

    def call(self, input, **kwargs):
        placeholder = tf.matmul(input, self.weights) + self.bias
        return tf.relu(placeholder)


