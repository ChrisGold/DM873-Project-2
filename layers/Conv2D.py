from typing import Any, Union

import tensorflow as tf
from keras import activations
from keras.utils import conv_utils
from tensorflow.keras.layers import *
# pylint: disable=g-classes-have-attributes
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.ops.variables import PartitionedVariable


class Conv2D(Layer):
    bias: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]
    kernel: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]

    def __init__(self, filters=32, strides=(1, 1), padding='valid', activation='relu', dilation_rate=(1, 1), **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.bias = None
        self.kernel_size = (2, 2)
        self.kernel_init = tf.keras.initializers.GlorotUniform()(self.kernel_size)
        self.kernel = None
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        super(Conv2D, self).build(input_shape)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    dtype='float32',
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        self.kernel = self.add_weight(shape=self.kernel_size,
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      trainable=True)

    def call(self, x, **kwargs):
        y = tf.keras.backend.conv2d(x, self.kernel, strides=(1, 1), padding='valid', data_format=None,
                                    dilation_rate=(1, 1))
        activation = activations.get(self.activation)
        if activation is not None:
            y = activation(y)
        return y

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        convX = conv_utils.conv_output_length(
            input_shape[1],
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0]
        )
        convY = conv_utils.conv_output_length(
            input_shape[2],
            self.kernel_size[1],
            padding=self.padding,
            stride=self.strides[1],
            dilation=self.dilation_rate[1]
        )
        return batch_size, convX, convY, self.filters
