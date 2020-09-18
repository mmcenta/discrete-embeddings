from collections import defaultdict
import glob
import os
import pickle
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from resnet import ResidualStack
from vector_quantizer import VectorQuantizer
from vqvae import VQVAE


CHECKPOINTS_DIR = "./checkpoints/"
GENERATED_SAMPLES_DIR = "./generated_samples/"
LOGS_DIR = "./logs/"
MODELS_DIR = "./models/"


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
    encoder = tf.keras.models.Sequential()
    for i, f in enumerate(n_filters):
        encoder.add(tf.keras.layers.Conv2D(f, 3, strides=(2, 2),
            padding='same', activation='relu', name='conv{}'.format(i)))

    pre_vq_conv = tf.keras.layers.Conv2D(embedding_dim, 1, strides=(1, 1),
        padding='same', name='pre_vq_conv')

    decoder = tf.keras.models.Sequential()
    for i, f in enumerate(reversed(n_filters)):
        decoder.add(tf.keras.layers.Conv2DTranspose(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='convT{}'.format(i)))
    decoder.add(tf.keras.layers.Conv2DTranspose(1, 3, strides=(1, 1),
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
    encoder = tf.keras.models.Sequential()
    for i, f in enumerate(filters):
        encoder.add(tf.keras.layers.Conv2D(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='conv{}'.format(i)))
    encoder.add(tf.keras.layers.Conv2D(n_final_filters, 3, stides=(1, 1),
        padding='same', activation='relu', name="conv{}".format(len(filters))))
    encoder.add(ResidualStack(n_final_filters, n_residual_filters,
        n_residual_blocks))

    pre_vq_conv = tf.keras.layers.Conv2D(embedding_dim, 1, strides=(1, 1),
        padding='same', name='pre_vq_conv')

    decoder = tf.keras.models.Sequential()
    decoder.add(tf.keras.layers.Conv2D(n_final_filters, 3, strides=(1, 1),
        padding='same', activation='relu', name="convt0"))
    decoder.add(ResidualStack(n_final_filters, n_residual_filters,
        n_residual_blocks))
    for i, f in enumerate(reversed(filters[1:]), start=1):
        decoder.add(tf.keras.layers.Conv2D(f, 4, strides=(2, 2),
            padding='same', activation='relu', name='convt{}'.format(i)))
    decoder.add(tf.keras.layers.Conv2D(3, 4, strides=(2, 2),
        padding="same", name='convt{}'.format(len(filters))))

    return encoder, decoder, pre_vq_conv


@tf.function
def train_step(model, image_batch, optimizer):
    """
    Executes one training step and returns the model output.
    """
    with tf.GradientTape() as tape:
        model_output = model(image_batch)
    gradients = tape.gradient(model_output['loss'], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model_output


def save_images(images, image_path):
    batch_size = images.shape[0]
    fig = plt.figure(figsize=(4, 4))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis('off')
    plt.savefig(image_path)
    plt.close()


def generate_and_save_images(model, originals, epoch, generated_samples_dir):
    model_output = model(originals)
    save_images(model_output['x_recon'], os.path.join(generated_samples_dir,
        "epoch{:02d}.png".format(epoch)))


def record_output(logs, output):
    if logs is None:
        logs = defaultdict(list)
    for k, v in output.items():
        logs[k].append(v)
    return logs


def print_metrics(title, logs, n_batches):
    lines = ['{}:'.format(title)]
    lines.append('  Loss: {}'.format(np.mean(
        logs['loss'][-n_batches:])))
    lines.append('  Reconstruction Error: {}'.format(np.mean(
        logs['recon_error'][-n_batches:])))
    lines.append('  Perplexity: {}'.format(np.mean(
        logs['perplexity'][-n_batches:])))
    lines.append('  VQVAE Loss: {}'.format(np.mean(
        logs['vqvae_loss'][-n_batches:])))
    print('\n'.join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='vqvae',
        help='Name of this run for logging.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
        help="Learning rate for training. Defaults to 1e-4.")
    parser.add_argument('--batch-size', '-bs', type=int, default=32,
        help='Batch size for train and test batches. Defaults to 32.')
    parser.add_argument('--n-embeddings', '-ne', type=int, default=10,
        help='Number of embedding vectors. Defaults to 10.')
    parser.add_argument('--embedding-dim', '-ed', type=int, default=64,
        help='Number of dimensions of the embedding space. Defaults to 64.')
    parser.add_argument('--commitment-cost', '-cc', type=float, default=0.25,
        help="Commitment cost, depends on the scale of the reconstruction "
             "loss. Defaults to 0.25.")
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='Number of training epochs. Defaults to 10.')
    parser.add_argument('--n-examples', type=int, default=16,
        help='Number of examples to generate per training epoch.'
             'Defaults to 16.')
    parser.add_argument('--cifar10', action='store_true',
        help="If set the model is trained on the CIFAR-10 dataset instead of "
             "MNIST.")
    args = parser.parse_args()

    # Check argument constraints
    assert args.batch_size >= args.n_examples

    # Create necessary directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_SAMPLES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(CHECKPOINTS_DIR, args.name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    generated_samples_dir = os.path.join(GENERATED_SAMPLES_DIR, args.name)
    os.makedirs(generated_samples_dir, exist_ok=True)

    # Load dataset
    if args.cifar10:
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    else:
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    # Preprocess images
    def preprocess_images(images, is_cifar=False):
        # Add channel dimension and normalize color values
        if not is_cifar:
            images = images.reshape((-1, 28, 28, 1))
        return (images / 255.0) - 0.5

    train_images = preprocess_images(train_images, is_cifar=args.cifar10)
    test_images = preprocess_images(test_images, is_cifar=args.cifar10)
    train_data_variance = np.var(train_images)

    # Shuffle data and split into batches
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(args.batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(args.batch_size))

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    # Pick a sample for generating sample images
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:args.n_examples, :, :, :]
    save_images(test_sample, os.path.join(generated_samples_dir,
        "original.png"))

    # Build model
    if args.cifar10:
        encoder, decoder, pre_vq_conv = get_cifar10_models(args.embedding_dim)
    else:
        encoder, decoder, pre_vq_conv = get_mnist_models(args.embedding_dim)
    vq = VectorQuantizer(args.n_embeddings, args.embedding_dim,
        args.commitment_cost)
    model = VQVAE(encoder, decoder, pre_vq_conv, vq, train_data_variance)

    # Run training loop
    n_train_batches = len(train_dataset)
    n_test_batches = len(test_dataset)
    train_logs = None
    test_logs = None
    generate_and_save_images(model, test_sample, 0, generated_samples_dir)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        for image_batch in train_dataset:
            train_output = train_step(model, image_batch, optimizer)
            train_logs = record_output(train_logs, train_output)
        end_time = time.time()

        print('Epoch: {} / time elapsed for current epoch: {}'
              .format(epoch, end_time - start_time))
        print_metrics("Train", train_logs, n_train_batches)

        for image_batch in test_dataset:
            test_output = model(image_batch)
            test_logs = record_output(test_logs, test_output)
        print_metrics("Test", test_logs, n_test_batches)
        generate_and_save_images(model, test_sample, epoch,
            generated_samples_dir)
        model.save_weights(os.path.join(checkpoints_dir,
            '{}_epoch{}'.format(args.name, epoch)))

    logs = {'train': train_logs, 'test': test_logs}
    with open(os.path.join(LOGS_DIR, "{}.pkl".format(args.name)), "wb") as f:
        pickle.dump(logs, f)
    model.save(os.path.join(MODELS_DIR, args.name))

    # Make GIF with images generated during training
    gif_file = os.path.join(generated_samples_dir, "training_animation.gif")
    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(generated_samples_dir, "epoch*"))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)