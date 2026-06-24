import torch

class StandardNormal:
    def __init__(self, dim):
        self.dim = dim

    def sample(self, n, device="cpu"):
        return torch.randn(n, self.dim, device=device)