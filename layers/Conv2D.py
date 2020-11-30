from typing import Any, Union

import tensorflow as tf
from keras import backend as K, activations
from tensorflow.keras.layers import *

# pylint: disable=g-classes-have-attributes
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.ops.variables import PartitionedVariable


class Conv2D(Layer):
    # TODO: Compute output shape
    bias: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]
    kernel: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]

    def __init__(self, filters=32, strides=1, padding='valid', activation='relu', **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.bias_init = tf.zeros_initializer()
        self.bias = None
        self.kernel_size = (2, 2, self.filters)
        self.kernel_init = tf.keras.initializers.GlorotUniform()(self.kernel_size)
        self.kernel = None
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def build(self, input_shape):

        super(Conv2D, self).build(input_shape)
        self.bias = self.add_weight(initial_value=self.bias_init(shape=(self.filters,), dtype='float32'),
                                    trainable=True)
        self.kernel = self.add_weight(initial_value=self.kernel_init, trainable=True)

    def call(self, x, **kwargs):
        y = K.conv2d(x, self.kernel, self.strides, self.padding)
        activation = activations.get(self.activation)
        if activation is not None:
            y = activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]-self.kernel_size[0], input_shape[2]-self.kernel_size[1], self.filters
