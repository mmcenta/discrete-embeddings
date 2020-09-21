import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K


MODELS_DIR = "./models/"
PLOTS_DIR = "./plots/"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
        help="Name of the model used for quantization.")
    parser.add_argument('--n-embeddings', '-ne', type=int, required=True,
        help='Number of embedding vectors.')
    parser.add_argument('--length', '-length', type=int, default=6,
        help="Length of the side of the grid that will be generated."
             " Defaults to 6.")
    parser.add_argument('--train', action="store_true",
        help="If set images will be sampled from the training set instead of "
             "the test set.")
    parser.add_argument('--cifar10', action='store_true',
        help="If set the CIFAR-10 dataset will be used instead of MNIST.")
    args = parser.parse_args()

    # Load images
    if args.cifar10:
        (train_images, _), (test_images, _) = K.datasets.cifar10.load_data()
    else:
       (train_images, _), (test_images, _) = K.datasets.mnist.load_data()
       train_images = train_images.reshape(-1, 28, 28, 1)
       test_images = test_images.reshape(-1, 28, 28, 1)
    all_images = train_images if args.train else test_images
    all_images = (all_images / 255.0) - 0.5

    # Sample images
    L = args.length
    n_sampled = L * L
    indices = np.random.randint(all_images.shape[0], size=n_sampled)
    images = all_images[indices, :, :, :]

    # Load trained model
    model = K.models.load_model(os.path.join(MODELS_DIR, args.name))

    # Quantize images
    output = model(images)
    quantized_images = output['vq_output']['encoding_indices']

    # Convert values to RGB values
    original_images = np.clip(images, -0.5, 0.5)
    original_images = 255 * (original_images + 0.5)
    original_images = original_images.astype("uint8")

    # Plot images
    im = None
    cmap = cm.get_cmap('rainbow', args.n_embeddings)
    fig, axs = plt.subplots(L, 2 * L, figsize=(15, 9))
    for i in range(n_sampled):
        original_idx = (i // L, 2 * (i % L))
        quantized_idx = (i // L, 2 * (i % L) + 1)
        if args.cifar10:
            axs[original_idx].imshow(original_images[i, :, :, :])
        else:
            axs[original_idx].imshow(original_images[i, :, :, :], cmap="gray")
        axs[original_idx].axis('off')
        im = axs[quantized_idx].imshow(quantized_images[i, :, :], cmap=cmap)
        axs[quantized_idx].axis('off')
    fig.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plots_dir = os.path.join(PLOTS_DIR, args.name)
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, "code_visualization.png"))
    plt.close(fig)
