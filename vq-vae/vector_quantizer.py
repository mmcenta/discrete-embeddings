import tensorflow as tf


EPS = 1e-10


class VectorQuantizer(tf.keras.Model):
    def __init__(self, embedding_dim, n_embeddings, commitment_cost,
        initializer=tf.keras.initializers.RandomUniform(-1, 1)):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._n_embeddings = n_embeddings
        self._commitment_cost = commitment_cost

        intial_w = initializer(shape=[embedding_dim, n_embeddings])
        self._w = tf.Variable(initial_value=intial_w, trainable=True,
            name="embedding")

    def call(self, x, is_training):
        # validate the input shape and flatten
        tf.assert_equal(tf.shape(x)[-1], self._n_embeddings)
        flat_x = tf.reshape(x, (-1, self._n_embeddings))

        # compute distances of the vectors in x_flat to the embedding vectors
        distances = (tf.reduce_sum(flat_x ** 2, axis=1, keepdims=True)
                     - 2 * tf.matmul(flat_x, self._w)
                     + tf.reduce_sum(self._w ** 2, axis=0, keepdims=True))

        # encode and quantize the inputs to the nearest embedding vector
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self._n_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # calculate q and e latent losses
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = x + tf.stop_gradient(quantized - x) # gradients of quantized are copied over to x
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + EPS)))

        return {
            'quantized': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
        }

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        w = tf.transpose(self.embeddings, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)