'''
2D point data (blob, tree). Same contract as viz_mnist.plot_samples.
'''

import os
from diffusion.viz.viz_functions import plot_transport, plot_scatter_comparison


def plot_samples(samples, out_dir, tag, labels=None, dataset=None, trajectory=None):
    """
    Scatter comparison against the real data, and the transport paths if a trajectory was recorded.
    """
    if dataset is not None:
        plot_scatter_comparison(
            dataset.X.cpu().numpy(),
            samples.numpy(),
            out_path=os.path.join(out_dir, f"comparison_{tag}.png"),
        )
    if trajectory is not None:
        plot_transport(trajectory.numpy(), out_path=os.path.join(out_dir, f"transport_{tag}.png"))
