import json

import tensorflow as tf


def write_metric(summary_writer, name, metric, epoch):
    with summary_writer.as_default():
        tf.summary.scalar(name, metric.result(), step=epoch)


def save_hyperparameters(path, hyperparams):
    with open(path, 'w') as f:
        json.dump(hyperparams, f)


def load_hyperparameters(path):
    with open(path, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams
