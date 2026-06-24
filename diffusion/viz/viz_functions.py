import matplotlib.pyplot as plt
import numpy as np
def viz_training(epochs, val_loss, train_loss, out_path):
    fig, ax = plt.subplots()
    ax.plot(epochs, val_loss, label="Validation Loss")
    ax.plot(epochs, train_loss, label="Training loss" )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    
def plot_prior(assumed_samples, mapped_samples, out_path):
    """(N,2) arrays."""
    fig, ax = plt.subplots()
    ax.scatter(mapped_samples[:, 0], mapped_samples[:, 1], label="mapped", alpha=0.5, s=5)
    ax.scatter(assumed_samples[:, 0], assumed_samples[:, 1], label="assumed", alpha=0.5, s=5)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_transport(trajectory, out_path, n_paths=200, fname="transport.png"):
    if trajectory.shape[-1] != 2:
        raise NotImplementedError("Transport only supports 2D data for now")

    B = trajectory.shape[1]
    n_paths = min(n_paths, B)
    idx = np.random.choice(B, size=n_paths, replace=False)
    traj = trajectory[:, idx, :]

    start = traj[0]
    end = traj[-1]

    fig, ax = plt.subplots()
    for k in range(n_paths):
        ax.plot(traj[:, k, 0], traj[:, k, 1], color="gray", alpha=0.15, linewidth=0.5)

    ax.scatter(start[:, 0], start[:, 1], s=8, alpha=0.5, color="tab:red", label="start (t=1, noise)")
    ax.scatter(end[:, 0], end[:, 1], s=8, alpha=0.5, color="tab:blue", label="end (t=0, data)")

    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)