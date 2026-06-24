from diffusion.diffusionprocess import DiffusionProcess
from data.blob import BlobDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from diffusion.viz.viz_functions import plot_transport
import os

#Basic idea
#How do we know how many steps are involved?
'''
sample from the prior and use euler mauryama or euler to to evolve the sample backwards until it is denoised (?)
If we use the pd_ode we solve it by euler integration.

the pf ode is differnt in differnt parameterizations

'''
def load_model(path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    network = instantiate(ckpt["model_cfg"])
    network.load_state_dict(ckpt["model_state_dict"])
    network.to(device).eval()
    return network, ckpt

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = cfg.device
    run_dir = HydraConfig.get().runtime.output_dir
    schedule = instantiate(cfg.schedule)
    tp = instantiate(cfg.time_param, schedule=schedule)
    model, _ = load_model(cfg.generation.trained_model_path, device=cfg.device)
    #Have to think about how to manage differences in saved config and live config.
    
    pmtrz = instantiate(cfg.parameterization, schedule=schedule)
    solver = instantiate(cfg.solver, time_param=tp, schedule =schedule , parameterization=pmtrz, N = cfg.solver.N, model=model)
    #I should probably load the model here based on info from config
    #temporary solution
    #TODO make this assumed prior stuff cleaner
    n_samples = 2000
    prior = instantiate(cfg.prior)
    x0 = prior.sample(n_samples, device=cfg.device)

    if cfg.viz:
        x_final, traj = solver.sample(x0=x0, return_trajectory=True)
        out_dir = os.path.join(run_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)
        plot_transport(traj.cpu().numpy(), out_path=os.path.join(out_dir, "transport.png"))
    else:
        x_final = solver.sample(x0=x0)

    data = BlobDataset(n_samples=cfg.dataset.n_samples, n_clusters=cfg.dataset.n_clusters, 
                       cluster_std=cfg.dataset.cluster_std, random_state=cfg.dataset.random_state)
    plt.scatter(data.X[:,0], data.X[:,1])
    plt.scatter(x_final[:,0], x_final[:,1])
    plt.show()
    return x_final


if __name__ == "__main__":  
    x = main()



    

    