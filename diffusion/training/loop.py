import torch
from torch import optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from hydra.utils import instantiate
from diffusion.training.lr_schedulerwrapper import SchedulerWrapper
from diffusion.training.load_save_ckpt import save_checkpoint
import os
from tqdm import tqdm

def to_device_helper(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


def fit_one(network, trainproc, train_loader, val_loader, cfg, device, ckpt_dir=None, sample_fn=None):
    optimizer = optim.Adam(network.parameters(), lr=cfg.trainer.lr)
    lr_scheduler = SchedulerWrapper(instantiate(cfg.lr_scheduler, optimizer=optimizer))
    losses_train, losses_val = [], []
    ema_beta = cfg.trainer.get("ema_beta", None)
    ema = None
    if ema_beta is not None:
        ema = AveragedModel(network, device, multi_avg_fn=get_ema_multi_avg_fn(ema_beta), use_buffers=True)
    for epoch in range(cfg.trainer.epochs):
        network.train()
        running_loss = 0.0
        #maxinterval must not be below mininterval, otherwise tqdm's monitor thread refreshes anyway
        interval = cfg.trainer.get("pbar_mininterval", 0.1)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}",
                    mininterval=interval, maxinterval=max(10.0, interval))
        for batch in pbar:
            batch = to_device_helper(batch, device)
            loss, _ = trainproc.training_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update_parameters(network)
            running_loss += loss.item()

        network.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device_helper(batch, device)
                val_loss, _ = trainproc.training_loss(batch)
                running_val_loss += val_loss.item()
        running_val_loss /= len(val_loader)

        lr_scheduler.step(running_val_loss)
        lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{cfg.trainer.epochs}], "
              f"train: {running_loss/len(train_loader):.4f}, "
              f"val: {running_val_loss:.4f}, lr: {lr:.2e}")
        losses_train.append(running_loss / len(train_loader))
        losses_val.append(running_val_loss)

        #periodic checkpoints so a killed run is not lost
        save_every = cfg.trainer.get("save_every", None)
        if ckpt_dir and save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"),
                            epoch + 1, network, optimizer, cfg, ema=ema)

        #periodic samples, from the EMA weights when we have them
        sample_every = cfg.trainer.get("sample_every", None)
        if sample_fn is not None and sample_every and (epoch + 1) % sample_every == 0:
            sample_fn(ema.module if ema is not None else network, epoch + 1)

    return network, optimizer, losses_train, losses_val, ema

