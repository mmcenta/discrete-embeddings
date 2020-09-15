import tensorflow as tf

from encoder_decoder import Encoder, Decoder
from vector_quantizer import VectorQuantizer

class VQVAE(tf.keras.Model):
    def __init__(self, n_embeddings, embedding_dim, n_hidden,
        n_residual_hidden, n_residual_blocks, data_variance,
        commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self._n_embeddings = n_embeddings
        self._embedding_dim = embedding_dim
        self._n_hidden = n_hidden
        self._n_residual_hidden = n_residual_hidden
        self._n_residual_blocks = n_residual_blocks

        self._data_variance = data_variance
        self._encoder = Encoder(n_hidden, n_residual_hidden, n_residual_blocks)
        self._decoder = Decoder(n_hidden, n_residual_hidden, n_residual_blocks)
        self._pre_vq_conv = tf.keras.layers.Conv2D(embedding_dim, 1, strides=1,
            padding="same")
        self._vq = VectorQuantizer(embedding_dim, n_embeddings, commitment_cost)

    def call(self, x, is_training=False):
        z = self._pre_vq_conv(self._encoder(x))
        vq_output = self._vq(z, is_training)
        x_recon = self.decoder(vq_output['quantized'])
        recon_error = tf.reduce_mean((x_recon - x) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }
