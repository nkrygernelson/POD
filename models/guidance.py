import torch
import torch.nn as nn


class GuidedModel(nn.Module):
    def __init__(self, base_model, labels, guidance_scale, num_classes):
        super().__init__()
        self.base = base_model
        #labels move with the device
        self.register_buffer("labels", labels.long().view(-1))  
        self.w = guidance_scale
        self.null_idx = num_classes

    def forward(self, x, t, cond=None):
        null = torch.full((x.shape[0],), self.null_idx,
                          dtype=torch.long, device=x.device)
        m_cond = self.base(x, t, cond=self.labels)
        m_uncond = self.base(x, t, cond=null)
        return m_uncond + self.w * (m_cond - m_uncond)
