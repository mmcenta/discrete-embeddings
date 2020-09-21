import tensorflow as tf
import tensorflow.keras as K

from encoder_decoder import Encoder, Decoder
from vector_quantizer import VectorQuantizer

class VQVAE(K.Model):
    def __init__(self, encoder, decoder, pre_vq_conv, vector_quantizer,
        data_variance):
        super(VQVAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._pre_vq_conv = pre_vq_conv
        self._vq = vector_quantizer
        self._data_variance = data_variance

    def get_embeddings(self):
        return self._vq.embeddings

    def call(self, x):
        z = self._pre_vq_conv(self._encoder(x))
        vq_output = self._vq(z)
        x_recon = self._decoder(vq_output['quantized'])
        recon_error = tf.reduce_mean((x_recon - x) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'perplexity': vq_output['perplexity'],
            'vqvae_loss': vq_output['loss'],
            'vq_output': vq_output,
        }
