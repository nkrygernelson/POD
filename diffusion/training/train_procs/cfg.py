
import torch
class CFGProc:
    def __init__(self, diffusion_process, num_classes, p_uncond=0.1, **kw):
        self.dp = diffusion_process
        self.num_classes = num_classes
        self.p_uncond = p_uncond

    def training_loss(self, batch):
        x0, y = batch
        cond = y.long().clone()
        drop = torch.rand(len(x0), device=x0.device) < self.p_uncond
        cond[drop] = self.num_classes
        return self.dp.training_loss(x0, cond=cond)
