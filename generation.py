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
from diffusion.viz.viz_functions import plot_transport, plot_density_comparison
from diffusion.utils.singularity import plot_singularity_report, ptype_of
import os
import matplotlib.pyplot as plt

# Basic idea
# How do we know how many steps are involved?
"""
sample from the prior and use euler mauryama or euler to to evolve the sample backwards until it is denoised (?)
If we use the pd_ode we solve it by euler integration.

the pf ode is differnt in differnt parameterizations

"""


def load_model(path, schedule, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    network = instantiate(ckpt["model_cfg"])
    network.load_state_dict(ckpt["model_state_dict"])
    network.to(device).eval()
    trained_pmtrz = instantiate(ckpt["param_cfg"], schedule=schedule, _recursive_=False)
    trained_tp = instantiate(ckpt["time_param_cfg"], schedule=schedule)
    return network, trained_pmtrz, trained_tp, ckpt


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = cfg.device
    run_dir = HydraConfig.get().runtime.output_dir
    schedule = instantiate(cfg.schedule)
    tp = instantiate(cfg.time_param, schedule=schedule)
    model, trained_pmtrz, trained_tp, ckpt = load_model(
        cfg.generation.trained_model_path, schedule, device=device
    )
    #we need the param used to train to be the same as the one we use to generate.
    #Want to avoid training a v_pred model and using the weights for score_pred.
    pmtrz = trained_pmtrz
    tp = trained_tp 
    solver = instantiate(
        cfg.solver,
        time_param=tp,
        schedule=schedule,
        parameterization=pmtrz,
        N=cfg.solver.N,
        model=model,
    )
    #we should warn if the param loaded from pt is different than what is in config:
    #check if param is diff
    live_param = cfg.parameterization._target_
    loaded_param = ckpt['param_cfg']['_target_']
    if  loaded_param != live_param:
        print(f"WARNING: Loaded param differnt than live_param\n"\
              f"live_param._target_:  {live_param}\n"
              f"loaded_param._target_: {loaded_param}")
    
    #now check time_param
    live_time_param = cfg.time_param._target_
    loaded_time_param = ckpt['time_param_cfg']['_target_']
    
    if loaded_time_param != live_time_param:
        print(f"WARNING: Loaded time_param different than live time_param\n"
            f"live:   {live_time_param}\n"
            f"loaded: {loaded_time_param}")

    # TODO make this assumed prior stuff cleaner
    n_samples = cfg.generation.num_samples
    prior = instantiate(cfg.prior)
    x0 = prior.sample(n_samples, device=cfg.device)

    if cfg.viz:
        x_final, traj = solver.sample(x0=x0, return_trajectory=True)
        out_dir = os.path.join(run_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)
        plot_transport(
            traj.cpu().numpy(), out_path=os.path.join(out_dir, "transport.png")
        )
        plot_singularity_report(
            schedule,
            out_path=os.path.join(out_dir, "singularity.png"),
            grid=tp.grid(int(cfg.solver.N)).cpu().numpy(),
            highlight=ptype_of(pmtrz),
        )
    else:
        x_final = solver.sample(x0=x0)

    data = instantiate(cfg.dataset)
    out_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    plot_density_comparison(
        data.X.cpu().numpy(),
        x_final.detach().cpu().numpy(),
        out_path=os.path.join(out_dir, "comparison.png"),
    )
    #temp hardcode
    print(cfg.dataset)
    if "blob" in cfg.dataset._target_.lower():
        X = data.X.cpu().numpy()
        G = x_final.detach().cpu().numpy()
        plt.scatter(X[:,0], X[:,1], s=5, label="True")
        plt.scatter(G[:,0], G[:,1], s=5, label="Gen")
        plt.show()
    return x_final


if __name__ == "__main__":
    x = main()
