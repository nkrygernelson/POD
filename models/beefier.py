#This model is vibecoded but it is surely more expressive than my simple model.
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_features: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(num_features, in_dim) * scale)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[0]

    def forward(self, x):
        proj = 2 * math.pi * x @ self.B.t()
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class ResBlock(nn.Module):
    """Pre-norm residual MLP block with the time embedding injected inside."""
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(t_dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, h, t_emb):
        x = self.act(self.lin1(self.norm(h)))
        x = self.act(x + self.t_proj(t_emb))
        return h + self.lin2(x)


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 256, n_blocks: int = 4,
                 time_emb_dim: int = 128,
                 x_fourier_features: int = 64, x_fourier_scale: float = 3.0,
                 time_fourier_scale: float = 1.0):
        super().__init__()

        # time embedding: scalar t -> Fourier -> MLP
        self.time_ff = FourierFeatures(1, time_emb_dim // 2, scale=time_fourier_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_ff.out_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # input embedding: raw x concatenated with Fourier features of x
        self.x_ff = FourierFeatures(input_dim, x_fourier_features, scale=x_fourier_scale)
        self.in_proj = nn.Linear(self.x_ff.out_dim + input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            ResBlock(hidden_dim, time_emb_dim) for _ in range(n_blocks)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t, cond=None):
        if t.dim() == 1:
            t = t.unsqueeze(-1)                       # (B,) -> (B, 1)
        t_emb = self.time_mlp(self.time_ff(t))
        h = self.in_proj(torch.cat([x, self.x_ff(x)], dim=-1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(self.out_act(self.out_norm(h)))
