"""
Weight functions to weight loss terms in the regression target
"""

import torch


def one_func(t):
    return torch.ones_like(t)


def linear_weight(t):
    return t


def exponential(t):
    return torch.exp(t)
