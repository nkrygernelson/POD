import torch
import torch.nn as nn


class EulerSolver:

    def sample(self, model: nn.Module, shape: tuple, device: torch.device) -> torch.Tensor:
        ...
