from math import ceil

import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.keras.layers import *


@tf.keras.utils.register_keras_serializable()
class MaxPooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super(MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        if strides is not None:
            self.strides = strides
        else:
            self.strides = pool_size
        self.padding = padding

    def build(self, input_shape):
        super(MaxPooling, self).build(input_shape)

    def call(self, x, **kwargs):
        y = tf.nn.max_pool2d(x, self.pool_size, self.strides, self.padding)
        return y

    def compute_output_shape(self, input_shape):
        rows = ceil(float(input_shape[1]) / float(self.strides[0]))
        cols = ceil(float(input_shape[2]) / float(self.strides[1]))
        return input_shape[0], rows, cols, input_shape[-1]

    #def get_config(self):
    #    config = {'pool_size': self.pool_size,
    #              'padding': self.padding,
    #              'strides': self.strideStride,}
    #             # 'data_format': self.data_format}

    #    base_config = super(MaxPooling, self).get_config()
    #    return dict(list(base_config.items()) + list(config.items()))


    def get_config(self):
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config


if __name__ == '__main__':
    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=64),
            MaxPooling(pool_size=(2, 2)),
            tf.keras.layers.Dense(units=10, activation='sigmoid',)
        ])
        return model



