import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, output_dim), 
            nn.Softmax()
        )
    def forward(self, x):
        return self.nn(x)
    
