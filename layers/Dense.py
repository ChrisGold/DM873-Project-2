import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import *
from keras import backend as K


@tf.keras.utils.register_keras_serializable()
class Dense(Layer):
    """
    The Dense layer implementation is quite simple. The layer inherits the layer class and takes as arguments units
    and an activation function.
    The unit parameter sets the amount of biases and weights created, as every unit is supposed to have one of each.
    The activation function is set as relu as default,
    since this is a good all-round activation function and commonly used.
    The activation function can be altered if needed.
    The weights are initialized with a random normal function to ensure that the weights have different values
    and do not start out in a symmetrical configuration which might collapse the mapping space.
    The biases are initialized with zeros to avoid adding further complexity and interfering with the weights.
    The layer outputs the dot-product of the input and the weights with the bias added.
    If an activation function is given as an argument, then this is applied to the results before it is outputted.
    """

    def __init__(self, units=32, activation='relu', **kwargs):
        """
        Initializes the Dense layer.
        :param units: The size of the output vector, default is 32
        :param activation: The activation function, default is relu
        """
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.b = None  # b is initialized in build
        self.w = None  # w is initialized in build
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = None

    def build(self, input_shape):
        """
        Builds the layer by creating the bias and w weights.
        :param input_shape: The shape of the input
        :return: The built layer with weights w and b
        """
        self.b = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            dtype='float32',
            name='dense_bias',
        )
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
            initializer=tf.keras.initializers.random_normal(),
            dtype='float32',
            name="dense_weights",
        )
        super(Dense, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        Invokes the layer on an input
        The result is the dot product of the input and the weights,
        to which the bias is added and the activation function is applied
        :param x: The input
        :return: The output
        """
        y = K.dot(x, self.w) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        """
        Returns the output shape for this dense layer. The output shape is a tuple of the
        :param input_shape:
        :return:
        """
        return input_shape[0], self.units

    def get_config(self):
        """
        Save the parameters to the config object returned from the superclass
        :return: A config object with fields units and activation
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            "activation": self.activation,
        })
        return config


if __name__ == '__main__':
    def create_model():
        """
        A helper function for creating a small model for testing our layer
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            Dense(),
            Dense(),
            Dense(units=64),
            Dense(units=128),
        ])
        return model


    def create_keras_model():
        """
        A helper function for creating a small model using the Keras layers for testing
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=32),
            tf.keras.layers.Dense(units=64),
            tf.keras.layers.Dense(units=128),
        ])
        return model


    model = create_model()
    print(model.summary())
    keras = create_keras_model()
    print(keras.summary())
