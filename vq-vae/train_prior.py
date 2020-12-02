import datetime
import sys
import os
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

from pixelcnn import get_mnist_pixelcnn
from util import write_metric, save_hyperparameters


CHECKPOINTS_DIR = "./checkpoints/"
LOGS_DIR = "./logs/"
MODELS_DIR = "./models/"
PLOTS_DIR = "./plots/"


@tf.function
def train_step(model, batch, optimizer, nll, conditional=False):
    """
    Executes one training step.
    """
    with tf.GradientTape() as tape:
        codes, _ = batch
        logits = model(codes)
        loss = K.losses.sparse_categorical_crossentropy(codes, logits,
            from_logits=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    nll(loss)


@tf.function
def test_step(model, batch, nll, conditional=False):
    codes, _ = batch
    logits = model(codes)
    loss = K.losses.sparse_categorical_crossentropy(codes, logits,
        from_logits=True)
    nll(loss)


def sample_from_prior(prior, shape):
    """Autoregressive sampling from prior, pixel by pixel"""
    codes = np.zeros(shape, dtype=np.int32)
    for i in range(codes.shape[1]):
        for j in range(codes.shape[2]):
            dist = tfp.distributions.Categorical(logits=prior(codes))
            sampled = dist.sample()
            codes[:, i, j] = sampled[:, i, j]
    return codes


def generate_and_plot_codes(prior, shape, length, image_path, is_cifar10=False):
    codes = sample_from_prior(prior, shape)

    # Plot each image in a grid
    im = None
    cmap = cm.get_cmap('rainbow', args.n_embeddings)
    fig, axs = plt.subplots(length, length)
    for i in range(length * length):
        idx = (i // length, i % length)
        im = axs[idx].imshow(codes[i, :, :], cmap=cmap)
        axs[idx].axis('off')
    fig.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(image_path)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
        help="Name of the base VQ-VAE model.")
    parser.add_argument('--n-embeddings', '-ne', type=int, required=True,
        help='Number of embedding vectors.')
    parser.add_argument('--batch-size', '-bs', type=int, default=32,
        help='Batch size for train and test batches. Defaults to 32.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='Number of training epochs. Defaults to 10.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=3e-4,
        help='Learning rate for training the PixelCNN. Defaults to 3e-4.')
    parser.add_argument('--n-layers', '-nl', type=int, default=12,
        help='Number of Gated PixelCNN blocks in the model. Defaults to 12.')
    parser.add_argument('--n-filters', '-nf', type=int, default=32,
        help='Number of convolutional filters. Defaults to 32.')
    parser.add_argument('--length', '-l', type=int, default=2,
        help="Length of the side of the grid that will be generated."
             " Defaults to 6.")
    parser.add_argument('-conditional', '-c', action='store_true',
        help='If set, the model will be conditioned on image labels.')
    parser.add_argument('-cifar10', action='store_true',
        help="If set, the CIFAR-10 dataset will be used instead of MNIST.")
    args = parser.parse_args()

    # Get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create necessary directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "{}_prior/{}/".format(
       args.name, current_time))
    os.makedirs(checkpoints_dir, exist_ok=True)
    plots_dir = os.path.join(PLOTS_DIR, args.name)
    os.makedirs(plots_dir, exist_ok=True)
    images_dir = os.path.join(plots_dir, "generated_codes/")
    os.makedirs(images_dir, exist_ok=True)

    print('Loading {} data...'.format('CIFAR-10' if args.cifar10 else 'MNIST'))

    # Load dataset
    if args.cifar10:
        dataset = K.datasets.cifar10.load_data()
        (train_images, train_labels), (test_images, test_labels) = dataset
    else:
        dataset = K.datasets.mnist.load_data()
        (train_images, train_labels), (test_images, test_labels) = dataset

    # Preprocess images
    def preprocess_images(images, is_cifar=False):
        # Add channel dimension and normalize color values
        if not is_cifar:
            images = images.reshape((-1, 28, 28, 1))
        return (images / 255.0) - 0.5

    train_images = preprocess_images(train_images, is_cifar=args.cifar10)
    test_images = preprocess_images(test_images, is_cifar=args.cifar10)

    # Load trained VQ-VAE model
    model = K.models.load_model(os.path.join(MODELS_DIR, args.name))

    # Quantize images
    def quantize(images):
        output = model(images)
        return output['vq_output']['encoding_indices']

    train_codes = tf.cast(quantize(train_images), dtype=tf.float32)
    test_codes = tf.cast(quantize(test_images), dtype=tf.float32)

    # Shuffle data and split into batches
    train_size = train_codes.shape[0]
    train_dataset = (tf.data.Dataset.from_tensor_slices((train_codes,
        train_labels)).shuffle(train_size).batch(args.batch_size))

    test_size = test_codes.shape[0]
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_codes,
        test_labels)).shuffle(test_size).batch(args.batch_size))

    print('Building model...')
    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    # Build model
    input_shape = list(train_codes.shape[-2:])
    hyperparams = {
        'input_shape': input_shape,
        'n_embeddings': args.n_embeddings,
        'n_layers': args.n_layers,
        'n_filters': args.n_filters,
    }
    save_hyperparameters(os.path.join(checkpoints_dir, 'hyperparams.json'),
        hyperparams)

    pixelcnn = None
    if args.cifar10:
        pass
    else:
        pixelcnn = get_mnist_pixelcnn(input_shape, args.n_embeddings,
            args.n_layers, args.n_filters)

    # Initialize logs
    print('Setting up logs...')
    train_log_dir = os.path.join(LOGS_DIR, "{}_prior/{}/train/".format(
        args.name, current_time))
    test_log_dir = os.path.join(LOGS_DIR, "{}_prior/{}/test/".format(
        args.name, current_time))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    train_nll = tf.metrics.Mean('train_nll', dtype=tf.float32)
    test_nll = tf.metrics.Mean('test_nll', dtype=tf.float32)

    # Generate data for epoch 0
    for batch in train_dataset:
        test_step(pixelcnn, batch, train_nll, conditional=args.conditional)
    write_metric(train_summary_writer, 'train_nll', train_nll, 0)
    for batch in test_dataset:
        test_step(pixelcnn, batch, test_nll, conditional=args.conditional)
    write_metric(test_summary_writer, 'test_nll', test_nll, 0)

    generated_codes_shape = [args.length * args.length] + input_shape
    generate_and_plot_codes(pixelcnn, generated_codes_shape, args.length,
        os.path.join(images_dir, "epoch0.png"), is_cifar10=args.cifar10)

    print('Checkpoints will be saved to {}.'.format(checkpoints_dir))
    print('Begin training...')

    # Run training loop
    for epoch in range(1, args.epochs + 1):
        # Reset metrics
        train_nll.reset_states()
        test_nll.reset_states()

        print("\nEpoch {}:".format(epoch))

        start_time = time.time()
        for batch in train_dataset:
            train_step(pixelcnn, batch, optimizer, train_nll,
                conditional=args.conditional)
        end_time = time.time()
        write_metric(train_summary_writer, 'train_nll', train_nll, epoch)
        print('Elapsed time: {}'.format(end_time - start_time))
        print('Train NLL: {}'.format(train_nll.result()))

        # Evaluate model on test set
        for batch in test_dataset:
            test_step(pixelcnn, batch, test_nll, conditional=args.conditional)
        write_metric(test_summary_writer, 'test_nll', test_nll, epoch)
        print("Test NLL: {}".format(test_nll.result()))

        # Generate codes to track progress
        images_path = os.path.join(images_dir, "epoch{}.png".format(epoch))
        generate_and_plot_codes(pixelcnn, generated_codes_shape, args.length,
            images_path, is_cifar10=args.cifar10)

        # Save checkpoint
        pixelcnn.save_weights(os.path.join(checkpoints_dir,
            '{}_epoch{}'.format(args.name, epoch)))
