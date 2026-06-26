# Training procedures, generation & recipes

This covers the three **training procedures** (`train_proc`), how generation/sampling
works, **classifier-free guidance**, end-to-end recipes, and known traps.

## Training procedures (`diffusion/training/train_procs/`)

A **train_proc wraps a `DiffusionProcess`** and exposes one method:

```python
def training_loss(self, batch) -> (loss, aux)
```

The proc **owns batch unpacking** (a batch may be a bare `x`, a `(x, label)` tuple, or a
`(x0, x1)` reflow pair). `fit_one` stays generic. Construction injects the DP:
`instantiate(cfg.train_proc, diffusion_process=DP, _recursive_=False)`. Each config
carries a `kind:` field that `train.py` dispatches on.

### `standard` — `StandardProc`
Plain diffusion loss. Drops any label, calls `DP.training_loss(x0)` (fresh noise sampled
inside). This is the default.

### `cfg` — `CFGProc` (classifier-free guidance)
Trains a single model to be both conditional and unconditional. Unpacks `(x0, y)`; with
probability `p_uncond` (default 0.1) replaces the label with the **null token** (index
`num_classes`); calls `DP.training_loss(x0, cond=cond)`. Requires:
- a **labelled dataset** (`blob`), and
- a **conditional model** (`beefier` with `num_classes` set).

Guidance is applied at *sampling* time (below).

### `reflow` — `ReflowProc` + `run_reflow` + `make_reflow_pairs`
Reflow straightens the probability-flow trajectories by iteratively retraining a *fresh*
model on coupled `(x0, x1)` pairs.

- **`ReflowProc`** unpacks an `(x0, x1)` pair and calls `DP.training_loss(x0, noise=x1)` —
  the curated `x1` replaces sampled noise.
- **`make_reflow_pairs`** builds a `TensorDataset(x0, x1)` by integrating the teacher's
  PF-ODE **backwards** (data `t=0` → noise `t=1`) with a deterministic `EulerSolver` over
  the **reversed** grid `torch.flip(tp.grid(N))`. Flipping the grid flips every step's
  `dt`, so the same `pfode` drift runs data→noise. (Euler is used because the stochastic
  Euler–Maruyama solver is not invertible.)
- **`run_reflow`** orchestrates the rounds: **round 0** trains (or loads via
  `cfg.train_proc.teacher_path`) a *standard* teacher; **rounds 1..n_steps** generate
  fresh pairs from the current teacher, train a fresh student on them, save
  `reflow_k.pt`, and promote the student to teacher.

Because reflow runs its own loop and per-round checkpoints, `train.py` strips labels from
the datasets, calls `run_reflow`, and returns early (no single-model save/viz).

## The shared training loop — `diffusion/training/loop.py::fit_one`

Adam + `SchedulerWrapper`, an epoch loop that calls `trainproc.training_loss(batch)`.
Notable: the **validation loss is averaged over the entire val set** (`running_val_loss`),
and that average is what drives the LR scheduler and is logged/plotted — not a single
minibatch. Returns `(network, optimizer, losses_train, losses_val)`.

## Generation & sampling — `generation.py`

1. `load_model` rebuilds the network from `model_cfg` and the parameterization +
   time_param from their saved configs (warns on mismatch vs the live config).
2. Build the solver via `instantiate(cfg.solver, time_param, schedule, parameterization,
   N, model)`.
3. Sample the prior `x0 = prior.sample(num_samples)`.
4. Integrate `t=1 → t=0` with `solver.sample(...)`.
5. Save `transport.png` (trajectories), `singularity.png`, `comparison.png` (true vs
   generated density).

### Guided sampling (CFG inference)
If `generation.guidance_scale` is set **and** the loaded model is conditional
(`num_classes` not None), `generation.py` wraps `solver.model` in a
`models.guidance.GuidedModel`:

```
m = m_uncond + w·(m_cond − m_uncond)
```

- Labels come from `generation.target_class` (all samples that class) or, if unset, an
  even split across classes (`arange(n) % num_classes`).
- Guidance scale `w`: `0` = unconditional, `1` = plain conditional, `>1` = amplified
  guidance (tighter concentration; too high over-saturates).
- This requires a **CFG-trained** checkpoint — guiding a model that only ever saw the
  null token (e.g. trained with `standard`) is meaningless.
- `num_classes` at generation comes from the checkpoint, not the CLI.

## Recipes

```bash
# Standard training + generation (default config)
uv run python train.py train_proc=standard
uv run python generation.py

# Reflow (round-0 teacher + n_steps refinement rounds; saves checkpoints/reflow_k.pt)
uv run python train.py train_proc=reflow train_proc.n_steps=2

# Classifier-free guidance: train a conditional model on a labelled dataset...
uv run python train.py train_proc=cfg model=beefier dataset=blob num_classes=3 dataset.n_clusters=3
# ...then generate one class with guidance
uv run python generation.py generation.guidance_scale=3 generation.target_class=0
# ...or an even split across all classes
uv run python generation.py generation.guidance_scale=3
# unconditional generation: leave guidance_scale unset/null

# Compare parameterizations / solvers / time params
uv run python train.py parameterization=v_pred
uv run python generation.py solver=euler            # or ddim / deis / euler_mauryama
uv run python train.py time_param=identity
```

## Known traps

- **Converter argument order.** Converters are `(m, x, t)`; `pfode`/`rev_sde` are
  `(x, t, m)`. Swapping them returns the wrong quantity silently (this was a real DEIS
  bug). When writing a solver, copy an existing one's call pattern.
- **CFG needs labels + a conditional model.** `tree` has no per-sample labels (use
  `blob`); the legacy `simple` model is unconditional (use `beefier` with `num_classes`).
  The default `beefier` config sets `num_classes`, but under `standard` training it only
  ever sees the null token, so it isn't actually class-conditional.
- **Stale-checkpoint / mismatch.** Generating with a different parameterization or
  time_param than was trained produces garbage; `load_model` reconciles from the
  checkpoint and warns. **Changing the model architecture invalidates old checkpoints —
  retrain.**
- **`generation.py` trailing `plt.show()`** (the blob debug scatter) opens an interactive
  window and **crashes in headless/no-Tk environments** — after `comparison.png` is
  already saved. Prefix `MPLBACKEND=Agg` if you hit this.
- **Endpoint singularities.** The linear schedule is near-singular at `t=0,1`. Use
  `time_param=logsnr` (or buffered `identity`), enough `solver_steps`, and check
  `singularity.png` for where a parameterization's `1/|det|` blows up.
- **Reflow validation.** Reflow trains on `(x0, x1)` pairs, so its loop builds pair
  loaders internally; the real-data val loader isn't reused for the reflow loss.
