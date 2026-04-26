import lightning as L


class ImageDataModule(L.LightningDataModule):

    def setup(self, stage=None):
        ...

    def train_dataloader(self):
        ...

    def val_dataloader(self):
        ...
