import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = 0
        if num_classes is not None:
            emb_dim = 16
            # index num_classes is the null / unconditional token
            self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        self.nn = nn.Sequential(
            nn.Linear(input_dim + 1 + emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t, cond=None):
        parts = [x, t]
        if self.num_classes is not None:
            if cond is None:
                cond = torch.full((x.shape[0],), self.num_classes,
                                  dtype=torch.long, device=x.device)
            parts.append(self.class_emb(cond))
        X = torch.cat(parts, dim=-1)
        return self.nn(X)