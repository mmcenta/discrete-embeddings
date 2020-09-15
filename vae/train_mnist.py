import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from vae import CVAE


EXP_EPS = 1e-8
MODELS_DIR = "./models/"
CHECKPOINTS_DIR = "./checkpoints/"
GENERATED_SAMPLES_DIR = "./generated_samples/"


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def save_images(images, image_path):
    batch_size = images.shape[0]
    fig = plt.figure(figsize=(4, 4))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis('off')
    plt.savefig(image_path)


def generate_and_save_images(originals, epoch, model, model_name):
    mean, logvar = model.encode(originals)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    save_images(predictions, os.path.join(GENERATED_SAMPLES_DIR,
        "{}_epoch{:03d}.png".format(model_name, epoch)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='cvae_mnist',
        help='Name of this run for logging.')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
        help="Learning rate for training. Defaults to 1e-4.")
    parser.add_argument('--batch-size', '-bs', type=int, default=32,
        help='Batch size for train and test batches. Defaults to 32.')
    parser.add_argument('--latent-dim', '-ld', type=int, default=2,
        help='Number of dimensions of the latent space. Defaults to 2.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='Number of training epochs. Defaults to 10.')
    parser.add_argument('--n-examples', '-ne', type=int, default=16,
        help='Number of examples to generate per training epoch.'
             'Defaults to 16.')
    parser.add_argument('--clipnorm', '-cn', type=float, default=10.0,
        help="Clip gradient norms to a maximum of this value. Defaults to 10.")
    parser.add_argument('--nan-debug', '-nd', action="store_true",
        help="If set, debugging is enabled to catch Tensors with NaN values.")
    args = parser.parse_args()

    # Check argument constraints
    assert args.batch_size >= args.n_examples

    # Enable debugging if necessary
    if args.nan_debug:
        tf.debugging.enable_check_numerics()

    # Create necessary directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(GENERATED_SAMPLES_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load dataset
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    # Preprocess images
    def preprocess_images(images):
        # Add channel dimension and normalize color values
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
        # Perform thresholding binarization
        return np.where(images > 0.5, 1.0, 0.0).astype('float32')

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    # Shuffle data
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_size).batch(args.batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(test_size).batch(args.batch_size))

    # Define optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    # Pick a sample for generating sample images
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:args.n_examples, :, :, :]
    save_images(test_sample, os.path.join(GENERATED_SAMPLES_DIR,
        "{}_original.png".format(args.name)))

    # Run training loop
    model = CVAE(args.latent_dim, exp_eps=EXP_EPS)
    generate_and_save_images(test_sample, 0, model, args.name)
    for epoch in range(1, args.epochs + 1):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        start_time = time.time()
        for x in train_dataset:
            train_loss(train_step(model, x, optimizer))
        end_time = time.time()

        for x in test_dataset:
            test_loss(compute_loss(model, x))
        elbo = -test_loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapsed for current epoch: {}'
            .format(epoch, elbo, end_time - start_time))
        generate_and_save_images(test_sample, epoch, model, args.name)
        model.save_weights(os.path.join(checkpoint_dir,
            '{}_ld{}_epoch{}'.format(args.name, args.latent_dim, epoch)))
