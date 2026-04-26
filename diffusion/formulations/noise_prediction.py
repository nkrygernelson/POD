import torch
import torch.nn as nn
from diffusion.base import BaseDiffusion


class NoisePrediction(BaseDiffusion):

    def q_sample(self, x0, t):
        ...

    def loss(self, model, x0, t):
        ...

    def predict_x0(self, model, xt, t):
        ...
