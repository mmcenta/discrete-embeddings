import tensorflow as tf


class ResConvBlock(tf.keras.Model):
    def __init__(self, n_input_filters, n_residual_filters):
        super(self, ResConvBlock).__init__()
        self.block = tf.keras.models.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_residual_filters, 3, strides=1,
                padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_input_filters, 1, strides=1,
                padding='valid'),
        ])

    def call(self, x):
        return x +  self.block(x)


class ResidualStack(tf.keras.Model):
    def __init__(self, n_input_filters, n_residual_filters, n_residual_blocks):
        super(self, ResidualStack).__init__()
        self._n_input_filters = n_input_filters
        self._n_residual_filters = n_residual_filters
        self._n_residual_blocks = n_residual_blocks

        self.stack = tf.keras.models.Sequential()
        for _ in range(n_residual_blocks):
            self.stack.add(ResConvBlock(n_input_filters, n_residual_filters))
        self.stack.add(tf.keras.layers.ReLU())

    def call(self, x):
        return self.stack(x)
