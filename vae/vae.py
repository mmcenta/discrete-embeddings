import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, dense_init_std=0.01):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2 * latent_dim,
            kernel_initializer=tf.initializers.random_normal(mean=0.0, stddev=dense_init_std))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = tf.keras.layers.Dense(7*7*32, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding='same',
            activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same')

    def call(self, z):
        z = self.dense(z)
        z = self.reshape(z)
        z = self.deconv1(z)
        z = self.deconv2(z)
        return self.deconv3(z)


class CVAE(tf.keras.Model):
    """Convolutional Variational Auto-Encoder for MNIST"""
    def __init__(self, latent_dim, dense_init_std=0.01, exp_eps=1e-8):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.exp_eps = exp_eps
        self.encoder = Encoder(latent_dim, dense_init_std=dense_init_std)
        self.decoder = Decoder(latent_dim)

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return tf.exp(self.exp_eps + 0.5 * logvar) * eps + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, latent_dim))
        return self.decode(eps, apply_sigmoid=True)
