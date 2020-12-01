import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import *
from keras import backend as K


class Dense(Layer):
    def __init__(self, units=32, activation='relu', **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.b = None  # b is initialized in build
        self.w = None  # w is initialized in build
        self.activation = activation

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            dtype='float32'
        )
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.random_normal(),
            trainable=True,
            dtype='float32'
        )
        super(Dense, self).build(input_shape)

    def call(self, x, **kwargs):
        y = K.dot(x, self.w) + self.b
        activation = activations.get(self.activation)
        if activation is not None:
            y = activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


