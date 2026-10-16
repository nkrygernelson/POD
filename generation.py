from diffusion.diffusionprocess import DiffusionProcess
from models.guidance import GuidedModel
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
from diffusion.viz.viz_functions import plot_transport, plot_density_comparison, plot_image_grid
from diffusion.utils.singularity import plot_singularity_report, ptype_of
from diffusion.training.load_save_ckpt import load_model
import os
import matplotlib.pyplot as plt



@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = cfg.device
    run_dir = HydraConfig.get().runtime.output_dir
    schedule = instantiate(cfg.schedule)
    eps = cfg.time_param.eps # the buffer around the edges of the grid
    tp = instantiate(cfg.time_param, eps = eps, schedule=schedule)
    model, trained_pmtrz, trained_tp, ckpt = load_model(
        cfg.generation.trained_model_path, schedule, device=device
    )
    #we need the param used to train to be the same as the one we use to generate.
    #For example, we want to avoid training a v_pred model and using the weights for score_pred.
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

    # Classifier-free guidance: wrap the (conditional) model so each solver step
    # combines a conditional and unconditional pass. Requires a CFG-trained model.
    guidance_scale = cfg.generation.get("guidance_scale", None)
    if guidance_scale is not None:
        num_classes = getattr(model, "num_classes", None)
        if num_classes is None:
            raise ValueError("guidance_scale set but the loaded model is unconditional "
                             "(num_classes is None). Train with train_proc=cfg first.")
        target_class = cfg.generation.get("target_class", None)
        if target_class is not None:
            labels = torch.full((n_samples,), int(target_class),
                                dtype=torch.long, device=cfg.device)
        else:
            labels = torch.arange(n_samples, device=cfg.device) % num_classes
        solver.model = GuidedModel(model, labels, guidance_scale, num_classes).to(cfg.device)

    if cfg.viz:
        x_final, traj = solver.sample(x0=x0, return_trajectory=True)
        out_dir = os.path.join(run_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)
        if x0.dim() == 2:
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

    out_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    #images get a sample grid, 2D data the density comparison
    if x_final.dim() > 2:
        plot_image_grid(x_final.detach().cpu(), out_path=os.path.join(out_dir, "samples.png"))
        return x_final
    data = instantiate(cfg.dataset)
    plot_density_comparison(
        data.X.cpu().numpy(),
        x_final.detach().cpu().numpy(),
        out_path=os.path.join(out_dir, "comparison.png"),
    )
    #temp hardcode
    if "blob" in cfg.dataset._target_.lower():
        X = data.X.cpu().numpy()
        G = x_final.detach().cpu().numpy()
        plt.scatter(X[:,0], X[:,1], s=5, label="True")
        plt.scatter(G[:,0], G[:,1], s=5, label="Gen")
        plt.show()
    return x_final


if __name__ == "__main__":
    x = main()
