import tensorflow as tf
import tensorflow.keras as K

from resnet import ResidualStack


class Encoder(K.Model):
    def __init__(self, n_hidden, n_residual_hidden, n_residual_blocks=2,
        apply_relu=True):
        self._n_hidden = n_hidden
        self._n_residual_hidden = n_residual_hidden
        self._n_residual_blocks = n_residual_blocks
        self._encoder = K.models.Sequential([
            K.layers.Conv2D(n_hidden // 2, 4, strides=2,
                padding="same", activation="relu"),
            K.layers.Conv2D(n_hidden, 4, strides=2,
               padding="same", activation="relu"),
            K.layers.Conv2D(n_hidden, 3, strides=1,
                padding="same", activation="relu"),
            ResidualStack(n_hidden, n_residual_hidden, n_residual_blocks,
                apply_relu=apply_relu),
        ])

    def call(self, x):
        return self._encoder(x)


class Decoder(K.Model):
    def __init__(self, n_hidden, n_residual_hidden, n_residual_blocks,
        apply_relu=True):
        self._n_hidden = n_hidden
        self._n_residual_hidden = n_residual_hidden
        self._n_residual_blocks = n_residual_blocks
        self._decoder = K.models.Sequential([
            K.layers.Conv2D(n_hidden, 3, strides=1,
                padding="same", activation="relu"),
            ResidualStack(n_hidden, n_residual_blocks, n_residual_blocks,
                apply_relu=apply_relu),
            K.layers.Conv2DTranspose(n_hidden // 2, 4, strides=2,
                padding="same", activation="relu"),
            K.layers.Conv2DTranspose(3, 4, strides=2,
                padding="same"),
        ])

    def call(self, x):
        return self._decoder(x)
