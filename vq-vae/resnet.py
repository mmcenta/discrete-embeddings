import tensorflow as tf


class ResConvBlock(tf.keras.Model):
    def __init__(self, n_in, n_hidden):
        super(self, ResConvBlock).__init__()
        self.block = tf.keras.models.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_in, 3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(n_hidden, 1, padding='valid'),
        ])

    def call(self, x):
        return x +  self.block(x)


class ResidualStack(tf.keras.Model):
    def __init__(self, n_in, n_hidden, n_blocks, apply_relu=False):
        super(self, ResidualStack).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.apply_relu = apply_relu

        self.stack = tf.keras.models.Sequential()
        for _ in range(n_blocks):
            self.stack.add(ResConvBlock(n_in, n_hidden))
        if apply_relu:
            self.stack.add(tf.keras.layers.ReLU())

    def call(self, x):
        return self.stack(x)
