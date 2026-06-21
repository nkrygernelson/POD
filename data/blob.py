import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import make_blobs
from torch.utils.data import Dataset

class BlobDataset(Dataset):
    def __init__(self, n_samples:int, n_clusters:int, cluster_std: float, random_state:int):
        X, y = make_blobs(n_samples=n_samples, n_features=2,
                        centers=n_clusters, cluster_std=cluster_std,
                        random_state=random_state)
        
        X = (X - X.mean(0)) / X.std(0)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]
       

