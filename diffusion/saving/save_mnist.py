'''
Contract (shared by every diffusion/saving/save_<dataset>.py):
    save_samples(samples, out_dir, tag, labels=None)
    samples: (N, *data_shape) cpu tensor in data space
    out_dir: folder to write into (already exists)
    tag:     name for the files, e.g. "epoch_005" or "final"
    labels:  (N,) class labels if sampling was conditional
'''

import os
import torch


def save_samples(samples, out_dir, tag, labels=None):
    """MNIST: the raw samples in [-1, 1] and their labels, as one .pt file."""
    torch.save({"samples": samples, "labels": labels},
               os.path.join(out_dir, f"samples_{tag}.pt"))
