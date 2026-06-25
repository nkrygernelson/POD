"""
from hydra we configure we select the schedule from the diffusion.schedules file. We get alpha and sigma from this.
We use alpha and sigma to define the scheduler.
The model is defined from hydra config and loaded
The schedule gives us the training objective and we at this stage also have the model form hydra so we can train.
Training params are handled by hydra here
"""

from diffusion.diffusionprocess import DiffusionProcess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hydra
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from hydra.core.hydra_config import HydraConfig
from diffusion.utils.schedulerwrapper import SchedulerWrapper
from diffusion.viz.viz_functions import viz_training, plot_prior
from diffusion.utils.singularity import plot_singularity_report, ptype_of
import os
import hashlib
from pathlib import Path


def save_checkpoint(path, epoch, network, optimizer, cfg):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
            "param_cfg": OmegaConf.to_container(cfg.parameterization, resolve=True),
            "schedule_cfg": OmegaConf.to_container(cfg.schedule, resolve=True),
            "time_param_cfg": OmegaConf.to_container(cfg.time_param, resolve=True),
        },
        path,
    )

def fit_one(network, trainproc, train_loader, val_loader, cfg, device):
    optimizer = optim.Adam(network.parameters(), lr=cfg.trainer.lr)
    lr_scheduler = SchedulerWrapper(instantiate(cfg.lr_scheduler, optimizer=optimizer))
    losses_train, losses_val = [], []

    for epoch in range(cfg.trainer.epochs):
        network.train()
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)
            loss, _ = trainproc.training_loss(inputs)   # <-- the only seam
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        network.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                v, _ = trainproc.training_loss(inputs)
                val_loss += v.item()
        val_loss /= len(val_loader)

        lr_scheduler.step(val_loss)
        lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{cfg.trainer.epochs}], "
              f"train: {running_loss/len(train_loader):.4f}, "
              f"val: {val_loss:.4f}, lr: {lr:.2e}")
        losses_train.append(running_loss / len(train_loader))
        losses_val.append(val_loss)

    return network, optimizer, losses_train, losses_val

def run_reflow():
    return None

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = cfg.device
    dataset = instantiate(cfg.dataset)
    n_val = int(len(dataset) * (1 - cfg.trainer.train_split))

    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set, batch_size=cfg.trainer.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=cfg.trainer.batch_size, shuffle=False)

    sched = instantiate(cfg.schedule)
    tp = instantiate(cfg.time_param, schedule=sched)
    t_distribution = instantiate(cfg.t_distribution, time_param=tp, device=device)
    weight_func = get_method(cfg.weight_func._target_)
    pmtrz = hydra.utils.instantiate(
        cfg.parameterization, schedule=sched, _recursive_=False
    )
    network = instantiate(cfg.model)
    #TODO fix wiring this
    trainproc = instantiate(cfg.train_proc)

    DP = DiffusionProcess(
        schedule=sched,
        parameterization=pmtrz,
        weight_func=weight_func,
        network=network,
        t_distribution=t_distribution,
        time_param=tp,
    )

    epochs = cfg.trainer.epochs
    device = cfg.device
    run_dir = HydraConfig.get().runtime.output_dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    optimizer = optim.Adam(network.parameters(), lr=cfg.trainer.lr)
    lr_scheduler_base = instantiate(cfg.lr_scheduler, optimizer=optimizer)
    lr_scheduler = SchedulerWrapper(lr_scheduler_base)
    network, optimizer, losses_train, losses_val = fit_one(network, trainproc, train_loader, val_loader, cfg, device)

    if cfg.trainer.save_final:
        hc = HydraConfig.get()
        cfg_hash = hashlib.sha1(
            OmegaConf.to_yaml(cfg, resolve=True).encode()
        ).hexdigest()[:4]
        # TODO: Dont hardcode the name to the save. Uncomment the lines
        # name = f"{hc.job.name}_{cfg_hash}"
        name = "train_best"
        save_dir = Path.cwd() / "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(
            os.path.join(save_dir, f"{name}.pt"), epochs, network, optimizer, cfg
        )
    if cfg.viz:
        out_dir = os.path.join(run_dir, "figures")
        os.makedirs(out_dir, exist_ok=True)

        epoch_axis = list(range(epochs))
        viz_training(
            epochs=epoch_axis,
            train_loss=losses_train,
            val_loss=losses_val,
            out_path=os.path.join(out_dir, "loss.png"),
        )
        num_samples = min(1000, len(val_set))
        X = torch.stack([val_set[i] for i in range(num_samples)]).to(device)

        noise = torch.randn_like(X)
        t1 = torch.ones(X.shape[0], 1, device=device)
        mapped = DP.forward_sample(X, t1, noise)  # (num_samples, 2)

        prior = instantiate(cfg.prior)
        assumed = prior.sample(num_samples, device=device)

        plot_prior(
            assumed_samples=assumed.cpu().numpy(),
            mapped_samples=mapped.cpu().numpy(),
            out_path=os.path.join(out_dir, "prior_check.png"),
        )

        plot_singularity_report(
            sched,
            out_path=os.path.join(out_dir, "singularity.png"),
            grid=tp.grid(int(cfg.solver.N)).cpu().numpy(),
            highlight=ptype_of(pmtrz),
        )


if __name__ == "__main__":
    main()
