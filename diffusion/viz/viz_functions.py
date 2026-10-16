import matplotlib
matplotlib.use("Agg")              # non-interactive backend: fast file output, no GUI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_density_comparison(true_pts, gen_pts, out_path, bins=300,
                            range_pct=(0.5, 99.5), smooth_sigma=1.5,
                            gamma=0.5, cmap="magma", dpi=200):
    """High-res 2D density comparison: true (left) vs generated (right).

    Both panels share bins / extent / colour-scale so the densities are directly
    comparable. The extent is an outlier-robust percentile range of the true
    data (so a few exploded generated points don't blow up the view, and NaNs
    are simply dropped). Optional Gaussian smoothing (`smooth_sigma`, in bins)
    turns sparse sample sets into a clean density; `gamma` < 1 (PowerNorm) lifts
    faint low-density structure such as the tree's thin branches.
    """
    true_pts = np.asarray(true_pts, dtype=float)
    gen_pts = np.asarray(gen_pts, dtype=float)

    lo = np.percentile(true_pts, range_pct[0], axis=0)
    hi = np.percentile(true_pts, range_pct[1], axis=0)
    rng = [[lo[0], hi[0]], [lo[1], hi[1]]]

    def density(pts):
        H, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
        if smooth_sigma:
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=smooth_sigma)
            except Exception:
                pass
        return H, xe, ye

    Ht, xe, ye = density(true_pts)
    Hg, _, _ = density(gen_pts)
    vmax = max(Ht.max(), Hg.max(), 1e-12)
    norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, H, title in zip(axes, (Ht, Hg), ("true", "generated")):
        ax.imshow(H.T, origin="lower", extent=extent, cmap=cmap, norm=norm,
                  aspect="equal", interpolation="bilinear")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_image_grid(images, out_path, nrow=10):
    """(N, C, H, W) images in [-1, 1] saved as a grid."""
    from torchvision.utils import save_image
    save_image((images.clamp(-1, 1) + 1) / 2, out_path, nrow=nrow)
