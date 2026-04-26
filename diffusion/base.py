from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseDiffusion(ABC):

    @abstractmethod
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def loss(self, model: nn.Module, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def predict_x0(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...
