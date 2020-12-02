import numpy as np
import tensorflow as tf
import tensorflow.keras as K


def gate(inputs):
    """Gated activations"""
    x, y = tf.split(inputs, 2, axis=-1)
    return tf.tanh(x) * tf.sigmoid(y)


class MaskedConv2D(K.layers.Layer):
    """Masked convolution"""
    def __init__(self, filters, kernel_size, padding, direction, mode, **kwargs):
        super(MaskedConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.direction = direction     # Horizontal or vertical
        self.mode = mode               # Mask type "a" or "b"
        self.padding = [[0, 0], [padding[0], padding[0]],
            [padding[1], padding[1]], [0, 0]]

    def build(self, input_shape):
        filter_mid_y = self.kernel_size[0] // 2
        filter_mid_x = self.kernel_size[1] // 2
        in_dim = int(input_shape[-1])

        # Build the mask
        w_shape = (self.kernel_size[0], self.kernel_size[1], in_dim, self.filters)
        mask_filter = np.ones(w_shape, dtype=np.float32)
        if self.direction == "h":
            mask_filter[filter_mid_y + 1:, :, :, :] = 0.
            mask_filter[filter_mid_y, filter_mid_x + 1:, :, :] = 0.
        elif self.direction == "v":
            if self.mode == 'a':
                mask_filter[filter_mid_y:, :, :, :] = 0.
            elif self.mode == 'b':
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
        if self.mode == 'a':
            mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        self.mask_filter = tf.Variable(initial_value=mask_filter, dtype=tf.float32,
            trainable=False)

        # Create convolution layer parameters with masked kernel
        self.W = self.add_weight("W_{}".format(self.direction), w_shape, dtype=tf.float32,
            trainable=True)
        self.b = self.add_weight("v_b", [self.filters,], dtype=tf.float32, trainable=True)

    def call(self, inputs):
        self.W.assign(self.W * self.mask_filter)
        return tf.nn.conv2d(inputs, self.W, (1, 1), self.padding) + self.b


class GatedPixelCNNBlock(K.Model):
    """
        Basic Gated-PixelCNN block.
       This is an improvement over PixelRNN to avoid "blind spots", i.e. pixels missingt from the
       field of view. It works by having two parallel stacks, for the vertical and horizontal direction,
       each being masked  to only see the appropriate context pixels.
    """
    def __init__(self, kernel, out_dim, mask='b', residual=True, **kwargs):
        super(GatedPixelCNNBlock, self).__init__(**kwargs)
        self.residual = residual

        # The (n × 1) and (n × n) masked convolutions are implemented by
        # (n // 2 × 1) and (n // 2 × n) convolutions followed by a shift in
        # pixels by padding and cropping, as suggested in the paper

        # Create vertical stack layers
        kernel_size = (kernel // 2 + 1, kernel)
        padding = (kernel // 2, kernel // 2)
        self.v_conv = MaskedConv2D(2 * out_dim, kernel_size, padding, "v", mask,
            name="v_masked_conv")
        self.v_gate = K.layers.Lambda(lambda x: gate(x), name="v_gate")

        # Create horizontal stack layers
        kernel_size = (1, kernel // 2 + 1)
        padding = (0, kernel // 2)
        self.h_conv = MaskedConv2D(2 * out_dim, kernel_size, padding, "h", mask,
            name="h_masked_conv")

        self.v_to_h_conv = K.layers.Conv2D(2 * out_dim, 1, strides=(1, 1),
            name="v_to_h")
        self.h_gate = K.layers.Lambda(lambda x: gate(x), name="h_gate")

        # Create residual convolution
        self.res_conv = K.layers.Conv2D(out_dim, 1, strides=(1, 1),
            name="res_conv")

    def call(self, v_input, h_input):
        # Vertical stack
        v_out = self.v_conv(v_input)
        v_out = v_out[:, :int(v_input.shape[-3]), :, :]
        v_out = self.v_gate(v_out)

        # Horizontal stack
        h_out = self.h_conv(h_input)
        h_out = h_out[:, :, :int(h_input.shape[-2]), :]
        v_to_h = self.v_to_h_conv(v_out)
        h_out = self.h_gate(h_out + v_to_h)

        # Residual convolution
        h_out = self.res_conv(h_out)
        if self.residual:
            h_out = h_out + h_input

        return v_out, h_out


class CondGatedPixelCNNBLock(GatedPixelCNNBlock):
    def __init__(self, kernel, out_dim, mask='b', residual=True, **kwargs):
        super(CondGatedPixelCNNBLock, self).__init__(kernel, out_dim,
            mask=mask, residual=residual, **kwargs)
        # Add conditioning layers
        self.v_cond_dense = K.layers.Dense(2 * out_dim)
        self.h_cond_dense = K.layers.Dense(2 * out_dim)

    def call(self, v_input, h_input):
        # Vertical stack
        v_out = self.v_conv(v_input)
        v_out = v_out[:, :int(v_input.shape[-3]), :, :]
        v_cond = self.v_cond_dense(h_input)
        v_out = self.v_gate(v_out + v_cond)

        # Horizontal stack
        h_out = self.h_conv(h_input)
        h_out = h_out[:, :, :int(h_input.shape[-2]), :]
        v_to_h = self.v_to_h_conv(v_out)
        h_cond = self.h_cond_dense(h_input)
        h_out = self.h_gate(h_out + v_to_h + h_cond)

        # Residual convolution
        h_out = self.res_conv(h_out)
        if self.residual:
            h_out = h_out + h_input

        return v_out, h_out


def get_mnist_pixelcnn(input_shape, n_embeddings, n_layers, n_filters,
    h_shape=None):
    """
    """
    is_conditional = (h_shape is not None)

    # Create inputs
    codes = K.Input(shape=input_shape, dtype=tf.float32, name="pixelcnn_input")
    expanded_codes = tf.expand_dims(codes, axis=-1)
    h = None
    if is_conditional:
        h = K.Input(shape=h_shape, dtype=tf.float32, name="pixelcnn_h")

    # Build Gated PixelCNN blocks
    v_in, h_in = expanded_codes, expanded_codes
    if is_conditional:
        v_out, h_out = CondGatedPixelCNNBLock(7, n_filters, mask='a',
            residual=False)(v_in, h_in, h)
        for _ in range(n_layers - 1):
            v_out, h_out = CondGatedPixelCNNBLock(3, n_filters, mask='b',
                residual=True)(v_out, h_out, h)
    else:
        v_out, h_out = GatedPixelCNNBlock(7, n_filters, mask='a',
            residual=False)(v_in, h_in)
        for _ in range(n_layers - 1):
            v_out, h_out = GatedPixelCNNBlock(3, n_filters, mask='b',
                residual=True)(v_out, h_out)

    # Build final fully connected layers for each cell, the final outputs are
    # probability distributions over the discrete values for each cell
    z = K.layers.Conv2D(n_filters, 1, activation='relu', name='fc1')(h_out)
    logits = K.layers.Conv2D(n_embeddings, 1, activation='softmax',
        name='fc2')(z)

    # Build and return model
    return K.Model(inputs=codes, outputs=logits, name="pixelcnn")
