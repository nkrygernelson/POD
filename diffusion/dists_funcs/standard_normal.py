import torch

"""
Wrapper that defines standard interface for distributions used in this project.
"""


class StandardNormal:
    def __init__(self, dim):
        # dim is an int for vectors (e.g. 2) or a shape for images (e.g. [1, 28, 28])
        self.dim = dim

    def sample(self, n, device="cpu"):
        shape = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
        return torch.randn(n, *shape, device=device)
