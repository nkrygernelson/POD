# Architecture & data flow

This document explains how a run is assembled, the runtime data flow for training and
sampling, the core abstractions, and two cross-cutting rules that everything depends on
(the `t` vs `u` convention and checkpoint reconciliation).

## Repository layout

```
POD/
├── main.py                 # launcher: train.py -> generation.py (forwards CLI args)
├── train.py                # TRAIN entrypoint (@hydra.main)
├── generation.py           # SAMPLE entrypoint (@hydra.main)
├── configs/                # Hydra config tree (see configuration.md)
├── diffusion/              # core library
│   ├── diffusionprocess.py # DiffusionProcess: forward_sample + training_loss + predict_*
│   ├── schedules/linear.py # alpha/sigma (+ derivatives, gamma)
│   ├── parameterizations/  # paramv2.py (ACTIVE), paramv1.py (legacy), det_check.py
│   ├── time/               # time.py (TimeParam/Identity/LogSNRTime), uniform.py (t-dists)
│   ├── solvers/            # euler.py, ddim.py, deis.py, euler_mauryama.py
│   ├── dists_funcs/        # standard_normal.py (prior), weights.py (loss weights)
│   ├── training/           # loop.py (fit_one), load_save_ckpt.py, lr_schedulerwrapper.py,
│   │   └── train_procs/    #   standard.py, cfg.py, reflow.py
│   ├── utils/singularity.py# conditioning diagnostics (1/|det| vs t)
│   └── viz/viz_functions.py# loss/prior/transport/density plots (Agg backend)
├── models/                 # beefier.py (ACTIVE), model.py (legacy), guidance.py (CFG wrapper)
├── data/                   # blob.py (labelled), tree.py (unlabelled NVIDIA "tree")
├── trained_models/         # train_best.pt + reflow_k.pt checkpoints
└── outputs/                # Hydra run dirs (gitignored): .hydra/, checkpoints/, figures/
```

## Core abstractions

### Schedule — `diffusion/schedules/linear.py`
Static methods `alpha(t)`, `sigma(t)`, `alpha_prime(t)`, `sigma_prime(t)`, `gamma(t)`.
`Linear`: `alpha = 1 − t + 1e-5`, `sigma = t + 1e-5`, `alpha' = −1`, `sigma' = 1`,
`gamma = 0.1` (constant; only the SDE solver uses `gamma`). The `1e-5` floors make the
schedule **near-singular at the endpoints** (`alpha(1) ≈ 1e-5`, `sigma(0) ≈ 1e-5`) — the
source of most sampling instability. This is why grids and training timesteps are
buffered away from the ends (see the `eps` buffer below).

### Parameterization — `diffusion/parameterizations/paramv2.py` (ACTIVE)
Every prediction is written as a linear combination `model_out = a(t)·x₀ + b(t)·ε`:

| class | a | b | native converter |
|-------|---|---|-----------------|
| `CleanPred` | 1 | 0 | `to_x_0 = m` |
| `NoisePred` | 0 | 1 | `to_noise = m` |
| `ScorePred` | 0 | −1/σ | `to_score = m` |
| `VPred` | α′ | σ′ | `to_v = m` |

The base class provides:
- `training_target(x₀, eps, t) = a·x₀ + b·ε`.
- `to_x0_eps(model_out, x_t, t)` — the **single inversion**: `det = α·b − σ·a`,
  `x₀ = (b·x_t − σ·m)/det`, `ε = (−a·x_t + α·m)/det`. Everything divides by `det`, which
  → ∞ where `det → 0` (the singularity reporter monitors this). **The inversion is affine
  in `model_out`**, which is what makes CFG guidance valid in model-output space.
- `to_x_0 / to_noise / to_score / to_v` derived from the inversion; each subclass
  overrides only its own trivial converter.
- `pfode(x, t, m) = to_v` (PF-ODE drift = predicted velocity).
- `rev_sde(x, t, m, gamma) = v − ½·gamma²·score` (reverse-SDE drift).

> ⚠️ **Argument-order trap.** The converters take `(m, x, t)` = `(model_out, x_t, t)`,
> but `pfode`/`rev_sde` take `(x, t, m)`. Mixing these up silently returns the wrong
> quantity (e.g. for `NoisePred`, `to_noise(x, t, m)` returns `x_t` instead of the
> predicted noise). `paramv1.py` is the legacy per-class version, not used.

### Time parameterization — `diffusion/time/time.py`
`TimeParam(schedule, eps=0)` exposes the `t ↔ u` bijection and the **single source of
truth for the usable support**:
- `u_bounds(eps=None)` → the working-coordinate range, buffered inward by `eps` (falls
  back to `self.eps`). It shrinks inward regardless of whether `u` runs low→high
  (LogSNR) or high→low (Identity).
- `grid(N, device, eps=None)` = `u_to_t(linspace(*u_bounds(eps), N+1))` — the solver grid.

Implementations:
- `Identity` — `u = t`, bounds `[1.0, 0.0]` (descending; index 0 = noise, last = data).
- `LogSNRTime(schedule, u_min=-10, u_max=10)` — `u = λ = 2·(log α − log σ)`, inverse
  `t = sigmoid(−u/2)`. `u_min=-10 → t≈0.993` (noise), `u_max=10 → t≈0.007` (data).

**Design point:** the schedule, converters, and weights are all written in **physical
`t`**. `time_param` only changes (a) the coordinate fed to the network and (b) the solver
grid spacing. It is *not* a full reparameterization of the schedule. See the `t` vs `u`
rule below.

### t-distribution — `diffusion/time/uniform.py`
Training timestep samplers. Both **derive their support from `time_param.u_bounds(eps)`**
so training and the sampling grid share the same buffered support. Each returns a
`(t, u)` pair.
- `Uniform_u(device, time_param, eps=None)` — uniform in `u` over the buffered bounds,
  then `t = u_to_t(u)`.
- `Uniform_t(device, time_param, eps=None)` — uniform in `t` over the *t-image* of those
  bounds, then `u = t_to_u(t)`.

`eps` defaults to `time_param.eps` (one shared knob); pass it explicitly only to use a
deliberately different training buffer.

### DiffusionProcess — `diffusion/diffusionprocess.py`
Built from `schedule, parameterization, weight_func, time_param, t_distribution,
network`. The training loss:

```python
def training_loss(self, x_0, noise=None, cond=None):
    t, u   = self.t_distribution.sample((B, 1))   # (t physical, u network coord)
    if noise is None: noise = torch.randn_like(x_0)   # curated noise hook (reflow)
    x_t    = alpha(t)·x_0 + sigma(t)·noise            # forward_sample, in t
    target = parameterization.training_target(x_0, noise, t)   # in t
    weight = weight_func(t)                            # in t
    pred   = network(x_t, u, cond=cond)                # network sees u (+ optional label)
    loss   = mean(weight · (pred − target)²)
    return loss, clean_loss   # clean_loss = diagnostic via to_x_0
```

Two hooks are important:
- `noise=` lets a caller inject **curated noise** instead of sampling it (used by reflow).
- `cond=` is an optional class label passed to the network (used by CFG; ignored by
  unconditional models).

It also exposes `predict_noise/x_0/score/v(x_t, t, cond=None)` helpers (convert `t→u` for
the net, keep the converter in `t`).

## Training data flow — `train.py` + `diffusion/training/loop.py`

1. Instantiate `dataset`; `random_split` into train/val loaders.
2. Instantiate `schedule → time_param(schedule) → t_distribution(time_param, device)`,
   `weight_func` (via `get_method`, it's a function not a class), `parameterization`
   (`_recursive_=False`, `schedule` injected), and `network`.
3. **Dispatch on `cfg.train_proc.kind`:**
   - `"reflow"` → strip labels from the datasets, call `run_reflow(...)`, return early
     (reflow runs its own multi-round loop + per-round checkpoints).
   - otherwise → build a `DiffusionProcess`, then
     `trainproc = instantiate(cfg.train_proc, diffusion_process=DP, _recursive_=False)`,
     then `fit_one(...)`.
4. `fit_one` (in `training/loop.py`): Adam + `SchedulerWrapper`, epoch loop calling
   `trainproc.training_loss(batch) -> (loss, aux)`; the batch is moved to device by a
   helper that handles a bare tensor **or** a tuple/list (labels / reflow pairs).
   **Validation loss is averaged over the whole val set** (`running_val_loss`), and that
   average drives the LR scheduler and the logged/plotted curve.
5. If `save_final`: write `trained_models/train_best.pt`. If `viz`: write
   `loss.png`, `prior_check.png`, `singularity.png`.

See **[training-and-generation.md](training-and-generation.md)** for the train_procs
themselves.

## Sampling data flow — `generation.py`

1. `load_model(ckpt, schedule)` rebuilds the network from `model_cfg` **and** the
   parameterization + time_param from their saved configs (not the live config).
2. Build `solver = instantiate(cfg.solver, time_param, schedule, parameterization, N,
   model)`. Warn if the live `cfg` parameterization/time_param disagree with the
   checkpoint's.
3. `x0 = prior.sample(num_samples)`.
4. **Optional guidance:** if `generation.guidance_scale` is set and the model is
   conditional, wrap `solver.model` in a `GuidedModel` (cond + uncond pass combined by
   the scale). See guided sampling in training-and-generation.md.
5. `solver.sample(x0, return_trajectory=cfg.viz)` integrates `t=1 → t=0`.
6. Save `transport.png`, `singularity.png`, and `comparison.png` (true vs generated
   density). A trailing debug block calls `plt.show()` for blob (crashes headless — see
   traps).

## The `t` vs `u` rule (do not break)

- Schedule math, `training_target`, every `to_*` converter, `pfode`, `rev_sde`, and
  `weight_func` operate on **physical `t`**.
- The **network** is fed the working coordinate **`u`** (`u = t` for Identity).
- Conversions `t ↔ u` happen only at origins: the `t_distribution` emits `(t, u)`; each
  solver / `predict_*` calls `t_to_u` once at the network boundary.

## Checkpoint reconciliation

`save_checkpoint` (in `training/load_save_ckpt.py`) stores the state dicts **plus four
configs**: `model_cfg`, `param_cfg`, `schedule_cfg`, `time_param_cfg`. On load,
`load_model` rebuilds the network, parameterization, and time_param from the checkpoint,
and `generation.py` **warns** if the live config's parameterization/time_param differ.

This matters because the network is conditioned on `u` and trained against a specific
parameterization — generating with a *different* time_param or parameterization than was
used for training silently produces garbage. Two consequences:
- **Changing the model class/architecture (e.g. adding `num_classes`) invalidates old
  checkpoints** — `generation.py` rebuilds from the saved `_target_`/args, so retrain.
- The parameterization/time_param are reconciled from the checkpoint, so you don't have
  to remember them at generation time — but heed the mismatch warnings.
