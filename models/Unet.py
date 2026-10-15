import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class FourrierEmbeddings(nn.Module):
    def __init__():
        super().__init__()


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, scale=16.0):
        super().__init__()
        self.register_buffer("W", torch.randn(emb_dim//2)*scale)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim),nn.SiLU(),
                                    nn.Linear(emb_dim,emb_dim))
    '''embeds time'''
    def forward(self, t):
          t = t.reshape(-1) # (B,), (B,1) or (B,1,1,1) -> (B,)
          proj = 2*math.pi*t[:, None]*self.W[None, :]
          return self.mlp(torch.cat([proj.sin(), proj.cos()], dim = -1))
          
          

class ResBlock(nn.Module):
    '''
    What does the resblock need to do? 
    (B, C, H, W)
    '''
    def __init__(self, in_ch, out_ch, emb_dim, dropout =0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.res_skip =  nn.Conv2d(in_ch, out_ch, 1 , padding = 0) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, emb):
          
          h = self.conv1(F.silu(self.norm1(x))) # B out_ch H, W
          h  =  h + self.emb_proj(F.silu(emb))[:,:, None, None] 
          h = self.conv2(self.dropout(F.silu(self.norm2(h))))
          return h + self.res_skip(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
    def forward(self, x):
         B, C, H, W = x.shape
         #flatten height and width 
         h = self.norm(x).flatten(2) #(B, C, H*W)
         h = h.transpose(1,2) #switches dim 1 with dim 2
         h, _ = self.attn(h,h,h,need_weights = False)
         h = h.transpose(1, 2).reshape(B, C, H, W)
         return x + h

class Downsample(nn.Module):
    '''
    stride 2, halves
    '''
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x): # (B, C, H, W) -> (B, C, H/2, W/2)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, ch_mults=(1, 2, 2), num_res_blocks=2,
                 attn_levels=(2,), emb_dim=256, dropout=0.1, time_scale=16.0, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.time_embed = TimeEmbedding(emb_dim, scale=time_scale)
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes + 1, emb_dim)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        for lvl, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, emb_dim, dropout)]
                if lvl in attn_levels:
                    layers.append(AttentionBlock(out_ch))
                self.down.append(nn.ModuleList(layers))
                ch = out_ch
                skip_chs.append(ch)
            if lvl != len(ch_mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout),
                                  AttentionBlock(ch),
                                  ResBlock(ch, ch, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_chs.pop(), out_ch, emb_dim, dropout)]
                if lvl in attn_levels:
                    layers.append(AttentionBlock(out_ch))
                self.up.append(nn.ModuleList(layers))
                ch = out_ch
            if lvl != 0:
                self.up.append(nn.ModuleList([Upsample(ch)]))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(),
                                 nn.Conv2d(ch, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _run(self, layers, h, emb):
        for layer in layers:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x, t, cond=None):
        emb = self.time_embed(t)
        if self.num_classes is not None:
            if cond is None:
                cond = torch.full((x.shape[0],), self.num_classes,
                                  dtype=torch.long, device=x.device)
            emb = emb + self.label_embed(cond.long().view(-1))

        h = self.in_conv(x)
        skips = [h]
        for layers in self.down:
            h = self._run(layers, h, emb)
            skips.append(h)

        h = self._run(self.mid, h, emb)

        for layers in self.up:
            if isinstance(layers[0], Upsample):
                h = layers[0](h)
            else:
                h = self._run(layers, torch.cat([h, skips.pop()], dim=1), emb)

        return self.out(h)
