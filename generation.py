from diffusion.diffusionprocess import DiffusionProcess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from diffusion.sampling import generate, write_samples
from diffusion.utils.singularity import plot_singularity_report, ptype_of
from diffusion.training.load_save_ckpt import load_model
import os



@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = cfg.device
    use_ema = cfg.generation.get("use_ema", False)
    run_dir = HydraConfig.get().runtime.output_dir
    schedule = instantiate(cfg.schedule)
    eps = cfg.time_param.eps # the buffer around the edges of the grid
    tp = instantiate(cfg.time_param, eps = eps, schedule=schedule)
    model, trained_pmtrz, trained_tp, ckpt = load_model(
        cfg.generation.trained_model_path, schedule, device=device, use_ema=use_ema
    )
    #we need the param used to train to be the same as the one we use to generate.
    #For example, we want to avoid training a v_pred model and using the weights for score_pred.
    pmtrz = trained_pmtrz
    tp = trained_tp 
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


    # trajectories are only kept when we plot (they can be large for images)
    x_final, labels, traj = generate(cfg, model, pmtrz, tp, schedule, return_trajectory=cfg.viz)

    if cfg.viz:
        out_dir = os.path.join(run_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)
        plot_singularity_report(
            schedule,
            out_path=os.path.join(out_dir, "singularity.png"),
            grid=tp.grid(int(cfg.solver.N)).cpu().numpy(),
            highlight=ptype_of(pmtrz),
        )

    # dataset-specific saving and plotting (cfg.sample_io)
    dataset = instantiate(cfg.dataset) if cfg.viz else None
    write_samples(cfg, x_final, labels, os.path.join(run_dir, "samples", "final"), "final",
                  dataset=dataset, trajectory=traj)
    return x_final


if __name__ == "__main__":
    x = main()
