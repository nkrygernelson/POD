import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x,t, cond=None):
        X =torch.cat([x,t], dim=-1)
        return self.nn(X)
    
