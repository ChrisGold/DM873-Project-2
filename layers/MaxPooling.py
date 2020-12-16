import tensorflow as tf
from keras.utils import conv_utils
from tensorflow.keras.layers import *


class MaxPooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super(MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        super(MaxPooling, self).build(input_shape)

    def call(self, x, **kwargs):
        y = tf.nn.max_pool(x, self.pool_size, self.strides, self.padding)
        return y

    def compute_output_shape(self, input_shape):
        conv_len = conv_utils.conv_output_length(input_shape[1], self.pool_size[0], self.padding, self.strides[0])
        return input_shape[0], conv_len, input_shape[2]

if __name__ == '__main__':

    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = (224, 224, 3)),
            Dense(units=32),
            Dense(units=32),
            Dense(units=64),
            MaxPooling(pool_size=(2,2)),
            Dense(units=128),
            ])
        return model

    def create_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = (224, 224, 3)),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=64),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dense(units=128),
            ])
        return model


    model = create_model()
    print(model.summary())
    keras = create_keras_model()
    print(keras.summary())