import tensorflow as tf


class ResConvBlock(tf.keras.Model):
    def __init__(self, n_in, n_residual_hidden):
        super(self, ResConvBlock).__init__()
        self.block = tf.keras.models.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_residual_hidden, 3, strides=1,
                padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_in, 1, strides=1,
                padding='valid'),
        ])

    def call(self, x):
        return x +  self.block(x)


class ResidualStack(tf.keras.Model):
    def __init__(self, n_in, n_residual_hidden, n_residual_blocks,
        apply_relu=False):
        super(self, ResidualStack).__init__()
        self._n_in = n_in
        self._n_residual_hidden = n_residual_hidden
        self._n_residual_blocks = n_residual_blocks
        self.apply_relu = apply_relu

        self.stack = tf.keras.models.Sequential()
        for _ in range(n_residual_blocks):
            self.stack.add(ResConvBlock(n_in, n_residual_hidden))
        if apply_relu:
            self.stack.add(tf.keras.layers.ReLU())

    def call(self, x):
        return self.stack(x)
