# POD — Documentation

POD is a small, research-oriented **continuous-time diffusion framework for 2D toy
datasets**. Its purpose is to make the moving parts of diffusion modelling explicit and
swappable so you can **compare** them under one roof:

- the four standard **parameterizations** (clean `x₀`, noise `ε`, score, velocity `v`),
- several **samplers** (Euler PF-ODE, DDIM, DEIS, Euler–Maruyama SDE),
- different **time parameterizations** (physical `t`, log-SNR) and **timestep
  distributions**,
- and several **training procedures** (plain diffusion, reflow, classifier-free
  guidance).

Everything is wired through **Hydra** config groups, so a run is just a composition of
choices you can edit in `configs/base.yaml` or override on the command line.

## The one rule to remember

**Time convention: `t = 0` is clean data, `t = 1` is pure noise.** The forward
(noising) process is

```
x_t = alpha(t)·x₀ + sigma(t)·ε
```

and **sampling integrates backwards, `t = 1 → t = 0`**.

## Mental model

A training/sampling run is assembled from these pieces:

| Piece | Role | Config group |
|---|---|---|
| **schedule** | `alpha(t)`, `sigma(t)` (+ derivatives, `gamma`) — defines the forward process | `schedule/` |
| **parameterization** | *what the network predicts* and how to convert that to `x₀`/`ε`/score/`v` and the sampling drift | `parameterization/` |
| **time_param** | bijection `t ↔ u` (the coordinate fed to the network and used for the solver grid) | `time_param/` |
| **t_distribution** | which timesteps training samples land on | `t_distribution/` |
| **DiffusionProcess** | ties the above + network together → the training loss | (built in code) |
| **train_proc** | *how* we train (standard / reflow / CFG); owns the per-batch loss | `train_proc/` |
| **solver** | integrates backwards to generate samples | `solver/` |
| **model** | the neural network `forward(x, t, cond=None)` | `model/` |
| **dataset / prior / weight_func / lr_scheduler** | data, the `t=1` prior, loss weighting, LR schedule | respective groups |

## Quickstart

```bash
# train with the default config, then generate from the checkpoint
uv run python main.py

# or run the two stages separately
uv run python train.py
uv run python generation.py

# override anything from the CLI (Hydra)
uv run python train.py parameterization=v_pred solver=euler time_param=identity
```

`main.py` simply runs `train.py` then `generation.py`, forwarding all CLI args to both.
Outputs land in `outputs/<timestamp>/` (Hydra run dir): `.hydra/` (resolved config),
`checkpoints/`, and `figures/`. The "best" model is also copied to
`trained_models/train_best.pt`, which is what `generation.py` loads by default.

## Environment

- **Python** `>=3.12,<3.13`. **Package manager: uv** — never pip/requirements.txt; add
  deps with `uv add <pkg>` (they go in `pyproject.toml`, locked in `uv.lock`).
- **torch 2.4.1**, matplotlib, scikit-learn, hydra-core + OmegaConf, click, tqdm.
- Default `device: cpu` (these are tiny 2D problems).

## Where to read next

- **[architecture.md](architecture.md)** — the runtime data flow, every core
  abstraction in detail, the `t` vs `u` rule, and how checkpoints reconcile.
- **[components.md](components.md)** — reference for each pluggable component
  (schedules, parameterizations, time, t-distributions, solvers, models, data).
- **[training-and-generation.md](training-and-generation.md)** — the three training
  procedures, the sampling/generation flow, guided sampling, recipes, and known traps.
- **[configuration.md](configuration.md)** — the Hydra config tree and how to override.

> A companion file `vpred_converters.md` (repo root) is a deeper math note on the
> parameterization conversions, the determinant/Wronskian, numerical stability, and
> log-SNR. `STRUCTURE.md` is an older LLM-oriented reference and may lag this `docs/`.
