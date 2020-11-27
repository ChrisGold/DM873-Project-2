import tensorflow as tf
from tensorflow.keras.layers import *


class Pooling(Layer):
    def __init__(self, poolingtype='max', **kwargs):
        self.poolingtype = poolingtype
        super(Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Pooling, self).build(input_shape)

    def call(self, x, **kwargs):
        return None
