from math import ceil

import tensorflow as tf
from tensorflow.keras.layers import *




@tf.keras.utils.register_keras_serializable()
class MaxPooling(Layer):
    """
    The MaxPooling layer takes as arguments the pool size, the strides, and the padding mode.
    The pool size defaults to (2, 2) and if no strides are given, the strides are set to the pool size,
    resulting in non-overlapping pools.
    The padding mode is by default "valid",
    which is fine for the default case that the pool size is significantly smaller than the input.
    MaxPooling implements a 2-Dimensional pooling operation using the backend primitive "max_pool2d".
    MaxPooling divides the 2D input data into pools of a given size and moving window distance and takes
    the largest value of every pool as the output.
    Intuitively, this can be understood as selecting the "most important" feature
    of a small area as a way to downsample data.
    MaxPooling does not have trainable weights; instead it is used
    for "glue logic" in extracting features and downsampling them.
    The pool size acts like a tweakable meta-variable.
    """


    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        """
        The constructor method

        :param pool_size: The size over which the pooling happens
        :param strides: The delta by which to move the pooling window
        :param padding: The padding mode passed on to max_pool2d
        """
        super(MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        if strides is not None:
            self.strides = strides
        else:
            self.strides = pool_size
        self.padding = padding

    def build(self, input_shape):
        """
        This method builds the layer. Because there are no trainable layers, it just calls to superclass
        :param input_shape: The input shape
        :return: The built layer instance
        """
        super(MaxPooling, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Invoke this layer on a given input
        :param x: The input
        :return: The output, the result of a maxpooling operation
        """
        y = tf.nn.max_pool2d(x, self.pool_size, self.strides, self.padding)
        return y

    def compute_output_shape(self, input_shape):
        """
        Determines the output shape from the input shape
        :param input_shape: The input shape
        :return: The calculated output shape
        """
        # Round up to compensate for the padding
        rows = ceil(float(input_shape[1]) / float(self.strides[0]))
        cols = ceil(float(input_shape[2]) / float(self.strides[1]))
        return input_shape[0], rows, cols, input_shape[-1]

    def get_config(self):
        """
        Get the configuration data for serialization.
        :return: The config object from the superclass, updated to include pool_size, strides and padding
        """
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config


if __name__ == '__main__':
    def create_model():
        """
        A simple model to test this layer
        :return:
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=64),
            MaxPooling(pool_size=(2, 2)),
            tf.keras.layers.Dense(units=10, activation='sigmoid', )
        ])
        return model
