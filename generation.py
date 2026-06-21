from diffusion.diffusionprocess import DiffusionProcess
from data.blob import BlobDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#Basic idea
#How do we know how many steps are involved?
'''
sample from the prior and use euler mauryama or euler to to evolve the sample backwards until it is denoised (?)
If we use the pd_ode we solve it by euler integration.

the pf ode is differnt in differnt parameterizations

'''
def load_model(path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    network = instantiate(ckpt["model_cfg"])      # rebuilds the architecture
    network.load_state_dict(ckpt["model_state_dict"])
    network.to(device).eval()
    return network, ckpt

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    schedule = instantiate(cfg.schedule)

    model, _ = load_model(cfg.generation.trained_model_path, device=cfg.device)
    #Have to think about how to manage differences in saved config and live config.
    
    pmtrz = instantiate(cfg.parameterization, schedule=schedule)
    solver = instantiate(cfg.solver, pfode_func=pmtrz.pfode, N = cfg.solver.N, model=model)
    #I should probably load the model here based on info from config
    #temporary solution
    n_samples = 2000
    x0 = torch.randn(n_samples, cfg.input_dim)
    model, _ = load_model(cfg.generation.trained_model_path, device=cfg.device)
    x_final = solver.sample(x0=x0)

    data = BlobDataset(n_samples=cfg.dataset.n_samples, n_clusters=cfg.dataset.n_clusters, 
                       cluster_std=cfg.dataset.cluster_std, random_state=cfg.dataset.random_state)
    plt.scatter(data.X[:,0], data.X[:,1])
    plt.scatter(x_final[:,0], x_final[:,1])
    plt.show()
    return x_final


if __name__ == "__main__":  
    x = main()



    

    