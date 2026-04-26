import torch
import lightning as L


class DiffusionModule(L.LightningModule):

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        ...
