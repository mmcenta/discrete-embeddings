# vae-lab

This repository contains *TensorFlow 2* implementations of a few VAE papers that caught my interest. Each directory contains a paper implementation and results from experiments. Currently, the following papers are implemented:

* [Vanilla VAE](vae/) ([paper](https://arxiv.org/abs/1312.6114))
* [VQ-VAE](vq-vae/) ([paper](https://arxiv.org/abs/1711.00937))

I plan to add the following papers to this repository:
* VQ-VAE 2 ([paper](https://arxiv.org/abs/1906.00446))
* Jukebox ([paper](https://arxiv.org/abs/2005.00341))

## Setup

All the implementations require:
* Python 3;
* TensorFlow 2;
* Tensorboard (to inspect training metrics);
* Matplotlib (to generate plots);

At the root of this repository, there is a `requirements.txt` file listing the installed packages on my development environment.
