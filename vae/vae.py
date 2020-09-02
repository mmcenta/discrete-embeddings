import tensorflow as tf
import tensorflow_probability as tfp


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = tf.keras.Conv2D(
            filters=64, kernel_size=3, strides=2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2 * latent_dim)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = tf.keras.layers.Dense(7*7*32, activation='relu')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=2, padding='same',
            activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same')

    def __call__(self, z):
        z = self.dense(z)
        z = tf.reshape(z , shape=(7, 7, 32))
        z = self.deconv1(z)
        z = self.deconv2(z)
        return self.deconv3(z)


class CVAE(tf.keras.Model):
    """Convolutional Variational Auto-Encoder for MNIST"""
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparamterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return tf.exp(0.5 * logvar) * eps + mean

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
