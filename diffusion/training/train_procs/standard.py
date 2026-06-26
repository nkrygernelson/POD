
class StandardProc:
    def __init__(self, diffusion_process, **kw):
        self.dp = diffusion_process

    def training_loss(self, batch):
        # batch may be a bare x, or a (x, label) tuple from a labelled dataset.
        x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self.dp.training_loss(x0)
