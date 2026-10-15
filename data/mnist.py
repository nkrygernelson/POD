from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    """MNIST preloaded into memory as tensors.

    self.X: float32 (N, 1, 28, 28) scaled to [-1, 1] to match the unit-variance prior.
    self.y: int64 (N,) class labels, so it works with train_proc=cfg.
    __getitem__ returns (x, y) like BlobDataset.
    """
    def __init__(self, root: str = "data", train: bool = True, download: bool = True):
        ds = MNIST(root, train=train, download=download)
        self.X = ds.data.unsqueeze(1).float() / 127.5 - 1.0
        self.y = ds.targets.long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
