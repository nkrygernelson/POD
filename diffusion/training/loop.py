import torch
from torch import optim
from hydra.utils import instantiate
from diffusion.training.lr_schedulerwrapper import SchedulerWrapper

def to_device_helper(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


def fit_one(network, trainproc, train_loader, val_loader, cfg, device):
    optimizer = optim.Adam(network.parameters(), lr=cfg.trainer.lr)
    lr_scheduler = SchedulerWrapper(instantiate(cfg.lr_scheduler, optimizer=optimizer))
    losses_train, losses_val = [], []

    for epoch in range(cfg.trainer.epochs):
        network.train()
        running_loss = 0.0
        for batch in train_loader:
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

    return network, optimizer, losses_train, losses_val

