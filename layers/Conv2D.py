from typing import Any, Union, Tuple

import tensorflow as tf
import keras.backend as K
from keras import activations
from keras.utils import conv_utils
from tensorflow.keras.layers import *
# pylint: disable=g-classes-have-attributes
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.ops.variables import PartitionedVariable


@tf.keras.utils.register_keras_serializable()
class Conv2D(Layer):
    """
    The convolutional layer takes as arguments filters, kernelsize, strides, padding, activation and dilationrate.
    The filters determines how many kernels is applied in each layer and is 32 by default.
    This value is important for the user to tune as a hyperparameter.
    The kernesize determines the size of each kernel to be run over the input and is 3 by 3 as default.
    The parameter strides is initialized to 1 by 1, and does the convolution operation without skipping pixels.
    The user can set activation function to be applied to the results after convolution and before output.
    The parameter padding is set to 'valid',  but the user can specify if they want another type
    of padding at the edges of the image. The dilation rate is set to one to not do any dilution.
    """
    bias: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]
    kernel: Union[Union[PartitionedVariable, ShardedVariable, Conv2D], Any]

    def __init__(self, filters: int = 32, kernel_size: Tuple[int, int] = (3, 3), strides: Tuple[int, int] = (1, 1),
                 padding: bool = 'VALID',activation='relu', dilation_rate=(1, 1), batch_size=1, **kwargs):
        """
        Initialising the Convulutional Layer
        :param filters: The number of filters to use, by default 32
        :param kernel_size: The size of the Kernel to use in the form of a tuple, by default (3,3)
        :param strides: The delta for moving the convolutional window, by default (1,1)
        :param padding: The padding mode for padding the sides, by default VALID
        :param activation: The activation function to apply, by default relu
        :param dilation_rate: The dilation rate, by default (1,1)
        :param batch_size: The batch size, by default 1
        :param kwargs:
        """
        self.filters = filters
        self.bias = None
        self.kernel_size = kernel_size
        self.kernel = None
        self.strides = strides
        self.padding = padding
        # Init activation here to avoid a nasty surprise in the call function
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = None
        self.dilation_rate = dilation_rate
        self.batch_size = batch_size
        if K.image_data_format() == 'channels_first':
            self.channel_axis = 0
        else:
            self.channel_axis = -1
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Initialize the kernel and the bias
        :param input_shape: The shape of the input
        :return: Returns this layer with two weights conv_bias and kernel
        """
        shape = (self.kernel_size[0], self.kernel_size[1], (input_shape[self.channel_axis]), self.filters)
        self.bias = self.add_weight(name='conv_bias',
                                    shape=(self.filters,),
                                    dtype='float32',
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      trainable=True)
        super(Conv2D, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Applies the convolutional operation to the input, adds the bias and applies the activation function
        :param x: The input
        :return: The output
        """
        y = tf.keras.backend.conv2d(x, self.kernel)
        y = K.bias_add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape based on the input shape and the parameters
        :param input_shape: The input shape
        :return: A vector describing a the dimensionality of a tensor of rank four
        """
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

    def get_config(self):
        """
        Save config parameters for serialization. This function returns a config object with fields
        filters
        kernel_size
        strides
        padding
        activation
        dilation_rate
        batch_size
        :return: The config object inherited from the superclass function
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "activation": self.activation,
            "dilation_rate": self.dilation_rate,
            "batch_size": self.batch_size,
        })
        return config


if __name__ == '__main__':
    def create_model():
        """
        A helper function for creating a small model for testing our layer
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            Conv2D(),
            Conv2D(),
        ])
        return model


    def create_keras_model():
        """
        A helper function for creating a small model using the Keras layers for testing
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.Conv2D(32, (3, 3)),
        ])
        return model


    keras = create_keras_model()
    print(keras.summary())
    model = create_model()
    print(model.summary())
