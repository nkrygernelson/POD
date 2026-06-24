import torch
def one_func(t):
    return torch.ones_like(t)

def linear_weight(t):
    return t.pow(2)

def exponential(t):
    return torch.exp(t)
    