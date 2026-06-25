"""Singularity / conversion-conditioning report for a (schedule, parameterization).

Recovering (x_0, eps) from a network prediction divides by the determinant

    det_P(t) = alpha*b - sigma*a

of the per-parameterization coefficients (a, b), where model_out = a*x_0 + b*eps.
The conversion amplification is therefore 1/|det|, and each parameterization's
prediction/target magnitude is max(|a|, |b|). This module plots both over t
(optionally with the solver grid overlaid) so you can see where each
parameterization blows up — det -> 0 (conversion) and/or |coeff| -> inf (target).

Depends only on the schedule + parameterization, not on the model or data, so it
is a static sanity check: run it once per (schedule, param), e.g. each generation.
"""
import numpy as np
import matplotlib.pyplot as plt

# Map the prediction classes to a coefficient key (robust to the ptype/pytpe typo).
_CLASS_TO_PTYPE = {
    "CleanPred": "clean_pred",
    "NoisePred": "noise_pred",
    "ScorePred": "score_pred",
    "VPred": "v_pred",
}
PARAMS = ["clean_pred", "noise_pred", "score_pred", "v_pred"]


def ptype_of(parameterization):
    """Coefficient key for an instantiated parameterization (or None if unknown)."""
    return _CLASS_TO_PTYPE.get(type(parameterization).__name__)


def _coeffs(ptype, alpha, sigma, ap, sp):
    """(a, b) such that model_out = a*x_0 + b*eps, as arrays over the t-grid."""
    if ptype == "clean_pred":
        return np.ones_like(alpha), np.zeros_like(alpha)
    if ptype == "noise_pred":
        return np.zeros_like(alpha), np.ones_like(alpha)
    if ptype == "score_pred":
        return np.zeros_like(alpha), -1.0 / sigma
    if ptype == "v_pred":
        return ap, sp
    raise ValueError(f"unknown ptype: {ptype}")


def singularity_report(schedule, t_min=1e-3, t_max=1.0 - 1e-3, n=512, params=PARAMS):
    """Return {ptype: {t, det, amplification (1/|det|), coeff_mag (max|a|,|b|)}}."""
    t = np.linspace(t_min, t_max, n)
    alpha = np.asarray(schedule.alpha(t), dtype=float)
    sigma = np.asarray(schedule.sigma(t), dtype=float)
    ap = np.asarray(schedule.alpha_prime(t), dtype=float) * np.ones_like(t)
    sp = np.asarray(schedule.sigma_prime(t), dtype=float) * np.ones_like(t)

    out = {}
    for p in params:
        a, b = _coeffs(p, alpha, sigma, ap, sp)
        det = alpha * b - sigma * a
        out[p] = {
            "t": t,
            "det": det,
            "amplification": 1.0 / np.maximum(np.abs(det), 1e-30),
            "coeff_mag": np.maximum(np.abs(a), np.abs(b)),
        }
    return out


def plot_singularity_report(schedule, out_path, t_min=1e-3, t_max=1.0 - 1e-3,
                            n=512, params=PARAMS, grid=None, highlight=None):
    """Plot 1/|det| and max(|a|,|b|) vs t for each parameterization.

    grid: optional array of solver t-grid points to overlay as a rug (shows
          whether the sampler actually evaluates the singular region).
    highlight: ptype string to draw thicker (the run's parameterization).
    Returns the underlying report dict.
    """
    rep = singularity_report(schedule, t_min, t_max, n, params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for p in params:
        lw = 2.6 if p == highlight else 1.2
        ax1.plot(rep[p]["t"], rep[p]["amplification"], label=p, linewidth=lw)
        ax2.plot(rep[p]["t"], rep[p]["coeff_mag"], label=p, linewidth=lw)

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    if grid is not None:
        g = np.asarray(grid).ravel()
        for ax in (ax1, ax2):
            y0 = ax.get_ylim()[0]
            ax.plot(g, np.full_like(g, y0), "|", color="k", alpha=0.3, markersize=6)
        ax1.plot([], [], "|", color="k", alpha=0.6, label="solver grid")

    ax1.set_ylabel("1/|det|   (conversion amplification)")
    ax2.set_ylabel("max(|a|,|b|)   (prediction / target scale)")
    ax2.set_xlabel("t   (0 = clean, 1 = noised)")
    title = "Singularity report" + (f"  —  {highlight}" if highlight else "")
    ax1.set_title(title)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return rep
