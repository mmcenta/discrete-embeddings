import tensorflow as tf
import tensorflow.keras as K

from resnet import ResidualStack

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


def get_mnist_models(embedding_dim, n_filters=[16, 32]):
    """
    Gets the Encoder, Decoder and Pre-Vector Quantization Convolution models
    for use with the MNIST dataset.

    Args:
        n_embeddings: The number of embedding vectors used in vector
            quantization.
        filters: A list with the number of filters for each convolutional
            layer in the Decoder and Encoder.

    Returns:
        A Tuple (encoder, decoder, pre_conv_vq) with the corresponding models.
    """
    encoder = K.models.Sequential()
    for i, f in enumerate(n_filters):
        encoder.add(K.layers.Conv2D(f, 3, strides=(2, 2),
            padding='same', activation='relu', name='conv{}'.format(i)))

    pre_vq_conv = K.layers.Conv2D(embedding_dim, 1, strides=(1, 1),
        padding='same', name='pre_vq_conv')

    decoder = K.models.Sequential()
    for i, f in enumerate(reversed(n_filters)):
        decoder.add(K.layers.Conv2DTranspose(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='convT{}'.format(i)))
    decoder.add(K.layers.Conv2DTranspose(1, 3, strides=(1, 1),
        padding='same', name='output'))

    return encoder, decoder, pre_vq_conv


def get_cifar10_models(embedding_dim, filters=[64, 128], n_residual_filters=32,
    n_residual_blocks=2):
    """
    Gets the Encoder, Decoder and Pre-Vector Quantization Convolution models
    for use with the CIFAR-10 dataset.

    Args:
        n_embeddings: The number of embedding vectors used in vector
            quantization.
        filters: A list with the number of filters for convolutional
            layers in the Encoder and Decoder.
        n_residual_filters: The number of filters in the bottleneck of
            residual blocks in the Encoder and Decoder
        n_residual_blocks: The number of residual in the Encoder and
            Decoder.

    Returns:
        A Tuple (encoder, decoder, pre_conv_vq) with the corresponding models.
    """
    n_final_filters = filters[-1]
    encoder = K.models.Sequential()
    for i, f in enumerate(filters):
        encoder.add(K.layers.Conv2D(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='conv{}'.format(i)))
    encoder.add(K.layers.Conv2D(n_final_filters, 3, strides=(1, 1),
        padding='same', activation='relu', name="conv{}".format(len(filters))))
    encoder.add(ResidualStack(n_final_filters, n_residual_filters,
        n_residual_blocks))

    pre_vq_conv = K.layers.Conv2D(embedding_dim, 1, strides=(1, 1),
        padding='same', name='pre_vq_conv')

    decoder = K.models.Sequential()
    decoder.add(K.layers.Conv2D(n_final_filters, 3, strides=(1, 1),
        padding='same', activation='relu', name="convt0"))
    decoder.add(ResidualStack(n_final_filters, n_residual_filters,
        n_residual_blocks))
    for i, f in enumerate(reversed(filters[1:]), start=1):
        decoder.add(K.layers.Conv2DTranspose(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='convt{}'.format(i)))
    decoder.add(K.layers.Conv2DTranspose(3, 4, strides=(2, 2),
        padding="same", name='convt{}'.format(len(filters))))

    return encoder, decoder, pre_vq_conv
