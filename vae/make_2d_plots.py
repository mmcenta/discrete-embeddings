import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from model import CVAE


def plot_latent_images(model, n, digit_size=28):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.savefig("./plots/latent_space.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
        help="Path to model checkpoint that will be used.")
    parser.add_argument('--num-images-per-side', '-n', type=int, default=10,
        help="Number of images per side of the plotted square. Defaults to 10.")
    parser.add_argument('--output-dir', '-od', type=str, default="./plots/",
        help="Name of directory in which the plots will be saved. Defaults to ./plots/.")
    args = parser.parse_args()

    # Make plots directory
    os.makedirs("./plots/", exist_ok=True)

    # Load checkpoint
    model = CVAE(2)
    model.build((32, 28, 28, 1))
    model.load_weights(args.checkpoint)

    # Plot latent space
    plot_latent_images(model, args.num_images_per_side)
