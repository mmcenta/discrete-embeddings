import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_probability as tfp

from pixelcnn import get_mnist_pixelcnn
from vqvae import get_mnist_vqvae, get_cifar10_vqvae
from util import load_hyperparameters


CHECKPOINTS_DIR = "./checkpoints/"
MODELS_DIR = "./models/"
PLOTS_DIR = "./plots/"


def save_images(images, length, image_path, is_cifar10=False):
    # Convert real values back to pixel values
    images = np.clip(images, -0.5, 0.5)
    images = 255 * (images + 0.5)
    images = images.astype("uint8")

    # Plot each image in a grid
    fig = plt.figure(figsize=(4, 4))
    for i in range(images.shape[0]):
        plt.subplot(length, length, i + 1)
        if is_cifar10:
            plt.imshow(images[i, :, :, :])
        else:
            plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis('off')
    plt.savefig(image_path)
    plt.close(fig)


def get_latest_run(name):
    base_dir = os.path.join(CHECKPOINTS_DIR, name)
    run_dirs = next(os.walk(base_dir))[1]
    if len(run_dirs) == 0:
        raise ValueError("No runs found on directory: '{}'".format(base_dir))
    run_dirs.sort()
    return os.path.join(base_dir, sorted(run_dirs)[-1])


def load_vq_vae(hparams, last_checkpoint, is_cifar10=False):
    if is_cifar10:
        model = get_cifar10_vqvae(hparams['n_embeddings'],
            hparams['embedding_dim'], hparams['commitment_cost'],
            hparams['train_data_variance'])
    else:
        model = get_mnist_vqvae(hparams['n_embeddings'],
            hparams['embedding_dim'], hparams['commitment_cost'],
            hparams['train_data_variance'])
    model.load_weights(last_checkpoint)
    return model

def load_prior(hparams, last_checkpoint, is_cifar10=False):
    if is_cifar10:
        pixelcnn = None
    else:
        pixelcnn = get_mnist_pixelcnn(hparams['input_shape'],
            hparams['n_embeddings'], hparams['n_layers'], hparams['n_filters'],
            h_shape=hparams['h_shape'])
    pixelcnn.load_weights(last_checkpoint)
    return pixelcnn


def sample_from_prior(prior, shape):
    """Autoregressive sampling from prior, pixel by pixel"""
    codes = np.zeros(shape, dtype=np.int32)
    for i in range(codes.shape[1]):
        for j in range(codes.shape[2]):
            dist = tfp.distributions.Categorical(logits=prior(codes))
            sampled = dist.sample()
            codes[:, i, j] = sampled[:, i, j]
    return codes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
        help="Name of the model used for quantization.")
    parser.add_argument('--vq-vae-run', type=str, default="",
        help="Run time for loading VQ-VAE checkpoint. Defaults to latest.")
    parser.add_argument('--prior-run', type=str, default="",
        help="Run time for loading PixelCNN checkpoint. Defaults to latest.")
    parser.add_argument('--length', '-l', type=int, default=6,
        help="Length of the side of the grid that will be generated."
             " Defaults to 6.")
    parser.add_argument('-cifar10', action='store_true',
        help="If set the CIFAR-10 dataset will be used instead of MNIST.")
    args = parser.parse_args()

    # Create directories if needed
    plots_dir = os.path.join(PLOTS_DIR, args.name)
    os.makedirs(plots_dir, exist_ok=True)

    # Get checkpoints directories for VQ-VAE and PixelCNN
    vq_vae_checkpoints_dir = args.vq_vae_run
    if len(vq_vae_checkpoints_dir) == 0:
        vq_vae_checkpoints_dir = get_latest_run(args.name)

    prior_checkpoints_dir = args.prior_run
    if len(prior_checkpoints_dir) == 0:
        prior_checkpoints_dir = get_latest_run("{}_prior".format(args.name))

    # Get latest checkpoints
    vq_vae_checkpoint = tf.train.latest_checkpoint(vq_vae_checkpoints_dir)
    prior_checkpoint = tf.train.latest_checkpoint(prior_checkpoints_dir)

    # Load hyperparamers
    vq_vae_hparams = load_hyperparameters(os.path.join(
        vq_vae_checkpoints_dir, 'hyperparams.json'))
    prior_hparams = load_hyperparameters(os.path.join(
        prior_checkpoints_dir, 'hyperparams.json'))

    # Get VQ-VAE and Prior
    vq_vae = load_vq_vae(vq_vae_hparams, vq_vae_checkpoint,
        is_cifar10=args.cifar10)
    prior = load_prior(prior_hparams, prior_checkpoint,
        is_cifar10=args.cifar10)

    codes_shape = [args.length * args.length,] + prior_hparams['input_shape']
    codes = sample_from_prior(prior, codes_shape)
    quantized = vq_vae._vq.quantize(codes)
    generated_images = vq_vae._decoder(quantized)

    samples_file = os.path.join(plots_dir, 'generated_samples.png')
    save_images(generated_images, args.length, samples_file)
