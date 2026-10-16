import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
def save_checkpoint(path, epoch, network, optimizer, cfg, ema = None):
    ckpt = {
                "epoch": epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
                "param_cfg": OmegaConf.to_container(cfg.parameterization, resolve=True),
                "schedule_cfg": OmegaConf.to_container(cfg.schedule, resolve=True),
                "time_param_cfg": OmegaConf.to_container(cfg.time_param, resolve=True),
            }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.module.state_dict()
    torch.save(
        ckpt,
        path,
    )

def load_model(path, schedule, device="cuda", use_ema=True):
    ckpt = torch.load(path, map_location=device)
    network = instantiate(ckpt["model_cfg"])
    #fall back to the raw weights for checkpoints saved without EMA
    key = "ema_state_dict" if use_ema and "ema_state_dict" in ckpt else "model_state_dict"
    network.load_state_dict(ckpt[key])
    network.to(device).eval()
    trained_pmtrz = instantiate(ckpt["param_cfg"], schedule=schedule, _recursive_=False)
    trained_tp = instantiate(ckpt["time_param_cfg"], schedule=schedule)
    return network, trained_pmtrz, trained_tp, ckpt
