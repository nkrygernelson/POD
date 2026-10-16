import torch
from torch import optim
from hydra.utils import instantiate
from diffusion.training.lr_schedulerwrapper import SchedulerWrapper
from diffusion.training.load_save_ckpt import save_checkpoint
import os
from tqdm import tqdm

def to_device_helper(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


def fit_one(network, trainproc, train_loader, val_loader, cfg, device, ckpt_dir=None):
    optimizer = optim.Adam(network.parameters(), lr=cfg.trainer.lr)
    lr_scheduler = SchedulerWrapper(instantiate(cfg.lr_scheduler, optimizer=optimizer))
    losses_train, losses_val = [], []

    for epoch in range(cfg.trainer.epochs):
        network.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = to_device_helper(batch, device)
            loss, _ = trainproc.training_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
                            epoch + 1, network, optimizer, cfg)

    return network, optimizer, losses_train, losses_val

