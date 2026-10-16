"""
Sampling shared by generation.py and the periodic samples during training.
prior -> (optional CFG wrapper) -> solver, all driven by cfg.generation and cfg.solver.
"""
import os
import torch
from hydra.utils import instantiate, get_method
from models.guidance import GuidedModel


def wgenerate(cfg, model, pmtrz, tp, schedule, return_trajectory=False):
    device = cfg.device
    n_samples = cfg.generation.num_samples
    solver = instantiate(
        cfg.solver,
        time_param=tp,
        schedule=schedule,
        parameterization=pmtrz,
        N=cfg.solver.N,
        model=model,
    )
    prior = instantiate(cfg.prior)
    x0 = prior.sample(n_samples, device=device)

    # Classifier-free guidance: wrap the (conditional) model so each solver step
    # combines a conditional and unconditional pass. Requires a CFG-trained model.
    labels = None
    guidance_scale = cfg.generation.get("guidance_scale", None)
    if guidance_scale is not None:
        num_classes = getattr(model, "num_classes", None)
        if num_classes is None:
            raise ValueError("guidance_scale set but the loaded model is unconditional "
                             "(num_classes is None). Train with train_proc=cfg first.")
        target_class = cfg.generation.get("target_class", None)
        if target_class is not None:
            labels = torch.full((n_samples,), int(target_class),
                                dtype=torch.long, device=device)
        else:
            labels = torch.arange(n_samples, device=device) % num_classes
        solver.model = GuidedModel(model, labels, guidance_scale, num_classes).to(device)

    traj = None
    if return_trajectory:
        x_final, traj = solver.sample(x0=x0, return_trajectory=True)
        traj = traj.detach().cpu()
    else:
        x_final = solver.sample(x0=x0)
    if labels is not None:
        labels = labels.cpu()
    return x_final.detach().cpu(), labels, traj


def write_samples(cfg, samples, labels, out_dir, tag, dataset=None, trajectory=None):
    """Save and plot samples with the dataset's own functions (cfg.sample_io)."""
    os.makedirs(out_dir, exist_ok=True)
    get_method(cfg.sample_io.save)(samples, out_dir, tag, labels=labels)
    if cfg.viz:
        get_method(cfg.sample_io.plot)(samples, out_dir, tag, labels=labels,
                                       dataset=dataset, trajectory=trajectory)
