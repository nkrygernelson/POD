'''
every dataset should have a plot sample/samples function that is custom but has a standard contract

Contract (shared by every diffusion/viz/viz_<dataset>.py):
    plot_samples(samples, out_dir, tag, labels=None, dataset=None, trajectory=None)
    samples:    (N, *data_shape) cpu tensor in data space
    out_dir:    folder to write into (already exists)
    tag:        name for the files, e.g. "epoch_005" or "final"
    labels:     (N,) class labels if sampling was conditional
    dataset:    the training dataset, for comparisons against real data
    trajectory: (steps+1, N, *data_shape) solver path, if one was recorded
Implementations use the optional arguments they need and ignore the rest.
'''

import os
from diffusion.viz.viz_functions import plot_image_grid


def plot_samples(samples, out_dir, tag, labels=None, dataset=None, trajectory=None):
    """
    Solvers and training will have a number of samples in a batch.
    MNIST: a grid of the generated digits, 10 per row (one row per label cycle).
    """
    plot_image_grid(samples, out_path=os.path.join(out_dir, f"samples_{tag}.png"), nrow=10)
