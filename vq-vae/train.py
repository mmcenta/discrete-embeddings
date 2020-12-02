import datetime
import glob
import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf
import tensorflow.keras as K

from vqvae import get_mnist_vqvae, get_cifar10_vqvae
from util import save_hyperparameters


CHECKPOINTS_DIR = "./checkpoints/"
PLOTS_DIR = "./plots/"
LOGS_DIR = "./logs/"
MODELS_DIR = "./models/"

METRICS = ['loss', 'recon_error', 'perplexity', 'vqvae_loss']


@tf.function
def train_step(model, image_batch, optimizer, metrics):
    """
    Executes one training step and returns the model output.
    """
    with tf.GradientTape() as tape:
        model_output = model(image_batch)
    gradients = tape.gradient(model_output['loss'], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    for m in METRICS:
        metrics[m](model_output[m])
    return model_output

@tf.function
def test_step(model, image_batch, metrics):
    model_output = model(image_batch)
    for m in METRICS:
        metrics[m](model_output[m])


def write_metrics(summary_writer, metrics, epoch):
    with summary_writer.as_default():
        for m in METRICS:
            tf.summary.scalar(m, metrics[m].result(), step=epoch)


def print_metrics(title, metrics):
    template = ('{} Metrics:\n  Loss: {}\n  Reconstruction Error: {}\n'
        '  Perplexity: {}\n  VQ-VAE Loss: {}')
    print(template.format(title,
        metrics['loss'].result(), metrics['recon_error'].result(),
        metrics['perplexity'].result(), metrics['vqvae_loss'].result()))


def save_images(images, image_path, is_cifar10):
    batch_size = images.shape[0]

    # Convert real values back to pixel values
    images = np.clip(images.numpy(), -0.5, 0.5)
    images = 255 * (images + 0.5)
    images = images.astype("uint8")

    # Plot each image in a grid
    fig = plt.figure(figsize=(4, 4))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        if is_cifar10:
            plt.imshow(images[i, :, :, :])
        else:
            plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis('off')
    plt.savefig(image_path)
    plt.close(fig)


def generate_and_save_images(model, originals, epoch, plots_dir, is_cifar10):
    model_output = model(originals)
    save_images(model_output['x_recon'], os.path.join(plots_dir,
        "generated_samples/epoch{:02d}.png".format(epoch)), is_cifar10)


def average_smoothing(s, window_size):
    """
    Smooths a series of scalars to its moving window average.
    Inspired by: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    Args:
        s: a series of scalars
        window_size: the size of the moving window

    Returns:
        a np.array of the same length of series containing the averages
    """
    if window_size < 2:
        return s
    s = np.r_[s[window_size-1:0:-1], s, s[-2:-window_size-1:-1]]
    window = np.ones(window_size) / window_size
    smooth_s = np.convolve(s, window, mode='valid')
    if window_size % 2 == 0:
        smooth_s = smooth_s[(window_size//2-1):-(window_size//2)]
    else:
        smooth_s = smooth_s[(window_size//2):-(window_size//2)]
    return smooth_s


def plot_codebook(embeddings, plots_dir):
    embeddings = embeddings.numpy().T

    codes = TSNE(n_components=2).fit_transform(embeddings)
    x, y = codes[:, 0], codes[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y)
    ax.set_title('T-SNE of codebook')
    fig.savefig(os.path.join(plots_dir, "codebook.png"))
    plt.close(fig)


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
    parser.add_argument('-cifar10', action='store_true',
        help="If set the model is trained on the CIFAR-10 dataset instead of "
             "MNIST.")
    args = parser.parse_args()

    # Check argument constraints
    assert args.batch_size >= args.n_examples

    # Get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create necessary directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    checkpoints_dir = os.path.join(CHECKPOINTS_DIR, "{}/{}/".format(
        args.name, current_time))
    os.makedirs(checkpoints_dir, exist_ok=True)
    plots_dir = os.path.join(PLOTS_DIR, args.name)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.join(plots_dir, 'generated_samples/'), exist_ok=True)

    # Remove samples from previous runs
    for file in glob.glob(os.path.join(plots_dir, "generated_samples/*")):
        os.remove(file)

    print('Loading {} data...'.format('CIFAR-10' if args.cifar10 else 'MNIST'))

    # Load dataset
    if args.cifar10:
        (train_images, _), (test_images, _) = K.datasets.cifar10.load_data()
    else:
        (train_images, _), (test_images, _) = K.datasets.mnist.load_data()

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

    print('Building model...')
    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    # Build model
    hyperparams = {
        'embedding_dim': args.embedding_dim,
        'n_embeddings': args.n_embeddings,
        'commitment_cost': args.commitment_cost,
        'train_data_variance': train_data_variance,
    }
    save_hyperparameters(os.path.join(checkpoints_dir, 'hyperparams.json'),
        hyperparams)

    model = None
    if args.cifar10:
        model = get_cifar10_vqvae(args.n_embeddings, args.embedding_dim,
            args.commitment_cost, train_data_variance)
    else:
        model = get_mnist_vqvae(args.n_embeddings, args.embedding_dim,
            args.commitment_cost, train_data_variance)

    # Initialize logs
    print('Setting up logs...')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(LOGS_DIR, "{}/{}/train/".format(
        args.name, current_time))
    test_log_dir = os.path.join(LOGS_DIR, "{}/{}/test/".format(
        args.name, current_time))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    train_metrics = {m: tf.metrics.Mean(m, dtype=tf.float32) for m in METRICS}
    test_metrics = {m: tf.metrics.Mean(m, dtype=tf.float32) for m in METRICS}

     # Pick a sample for generating sample images
    test_sample = None
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:args.n_examples, :, :, :]
    save_images(test_sample, os.path.join(plots_dir,
        'generated_samples/original.png'), args.cifar10)

    # Generate data for epoch 0
    generate_and_save_images(model, test_sample, 0, plots_dir, args.cifar10)
    for image_batch in train_dataset:
        test_step(model, image_batch, train_metrics)
    write_metrics(train_summary_writer, train_metrics, 0)
    for image_batch in test_dataset:
        test_step(model, image_batch, test_metrics)
    write_metrics(test_summary_writer, test_metrics, 0)

    print('Checkpoints will be saved to {}.'.format(checkpoints_dir))
    print('Begin training...')

    # Run training loop
    for epoch in range(1, args.epochs + 1):
        # Train for an epoch
        print("\nEpoch {}:".format(epoch))
        start_time = time.time()
        for image_batch in train_dataset:
            train_step(model, image_batch, optimizer, train_metrics)
        end_time = time.time()
        print('Elapsed time: {}'.format(end_time - start_time))
        write_metrics(train_summary_writer, train_metrics, epoch)
        print_metrics("Train", train_metrics)

        # Evaluate model on test set
        for image_batch in test_dataset:
            test_step(model, image_batch, test_metrics)
        write_metrics(test_summary_writer, test_metrics, epoch)
        print_metrics("Test", test_metrics)

        # Print metrics and save checkpoint
        generate_and_save_images(model, test_sample, epoch, plots_dir,
            args.cifar10)
        model.save_weights(os.path.join(checkpoints_dir,
            '{}_epoch{}'.format(args.name, epoch)))

        # Reset metrics
        for m in METRICS:
            train_metrics[m].reset_states()
            test_metrics[m].reset_states()

    # Save model
    print('Saving models to {}...'.format(MODELS_DIR, args.name))
    model.save(os.path.join(MODELS_DIR, args.name))

    # Make GIF with images generated during training
    gif_file = os.path.join(plots_dir, "training_animation.gif")
    print("Saving training animation to {}...".format(gif_file))
    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(plots_dir,
            "generated_samples/epoch*"))
        for filename in sorted(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filenames[-1])
        writer.append_data(image)

    # Plot learned embeddings in two dimensions
    print('Plotting codebook to {}...'.format(os.path.join(plots_dir,
        "codebook.png")))
    embeddings = plot_codebook(model.get_embeddings(), plots_dir)

    print('Finished.')
