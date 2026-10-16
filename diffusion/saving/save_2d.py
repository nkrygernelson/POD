'''
2D point data (blob, tree). Same contract as save_mnist.save_samples.
'''

import os
import numpy as np


def save_samples(samples, out_dir, tag, labels=None):
    """2D: (N, 2) points as .npy, labels alongside if there are any."""
    np.save(os.path.join(out_dir, f"samples_{tag}.npy"), samples.numpy())
    if labels is not None:
        np.save(os.path.join(out_dir, f"labels_{tag}.npy"), labels.numpy())
