import tensorflow as tf
import tensorflow.keras as K


class ResConvBlock(K.Model):
    def __init__(self, n_input_filters, n_residual_filters):
        super(ResConvBlock, self).__init__()
        self.block = K.models.Sequential([
            K.layers.ReLU(),
            K.layers.Conv2D(n_residual_filters, 3, strides=1,
                padding='same'),
            K.layers.ReLU(),
            K.layers.Conv2D(n_input_filters, 1, strides=1,
                padding='valid'),
        ])

    def call(self, x):
        return x +  self.block(x)


class ResidualStack(K.Model):
    def __init__(self, n_input_filters, n_residual_filters, n_residual_blocks):
        super(ResidualStack, self).__init__()
        self._n_input_filters = n_input_filters
        self._n_residual_filters = n_residual_filters
        self._n_residual_blocks = n_residual_blocks

        self.stack = K.models.Sequential()
        for _ in range(n_residual_blocks):
            self.stack.add(ResConvBlock(n_input_filters, n_residual_filters))
        self.stack.add(K.layers.ReLU())

    def call(self, x):
        return self.stack(x)
