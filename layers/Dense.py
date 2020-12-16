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
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = None

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(self.units,),
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
        if self.activation is not None:
            y = self.activation(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


if __name__ == '__main__':


    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = (224, 224, 3)),
            Dense(),
            Dense(),
            Dense(units=64),
            Dense(units=128),
            ])
        return model


    def create_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape = (224, 224, 3)),
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