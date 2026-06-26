# Component reference

Each pluggable component is a class/function selected by a Hydra config group. This is a
reference for what's available and the contract each must satisfy. See
[configuration.md](configuration.md) for how the groups compose, and
[architecture.md](architecture.md) for the data flow.

## Schedules — `configs/schedule/`, `diffusion/schedules/`
Defines the forward process `x_t = alpha(t)·x₀ + sigma(t)·ε`.

| name | class | notes |
|---|---|---|
| `linear` | `Linear` | `alpha=1−t+1e-5`, `sigma=t+1e-5`, `alpha'=−1`, `sigma'=1`, `gamma=0.1` |

**Contract:** static methods `alpha, sigma, alpha_prime, sigma_prime` (and `gamma` if
you'll use the SDE solver). All math is in physical `t`. The `1e-5` endpoint floors make
it near-singular at `t=0,1` — keep enough solver steps and use the `eps` buffer / log-SNR.

## Parameterizations — `configs/parameterization/`, `diffusion/parameterizations/paramv2.py`
Says what the network predicts and converts between quantities.

| name | class | predicts |
|---|---|---|
| `clean_pred` | `CleanPred` | `x₀` |
| `noise_pred` | `NoisePred` | `ε` |
| `score_pred` | `ScorePred` | score = `−ε/σ` |
| `v_pred` | `VPred` | velocity `v = α'·x₀ + σ'·ε` |

**Contract:** subclass `Parameterization`, implement `coeffs(t) → (a, b)`, override the
one native converter to return `m`. Conversions, `pfode`, `rev_sde` come free from the
base. Converter signature is `(m, x, t)`; `pfode`/`rev_sde` are `(x, t, m)`.

Use the **singularity report** (`figures/singularity.png`) to see where each
parameterization's `1/|det|` blows up for a given schedule + grid.

## Time parameterizations — `configs/time_param/`, `diffusion/time/time.py`
The `t ↔ u` bijection; `u` is the coordinate fed to the network and used for the grid.

| name | class | `u` | config keys |
|---|---|---|---|
| `identity` | `Identity` | `u = t` | `eps` |
| `logsnr` | `LogSNRTime` | `u = 2(log α − log σ)` | `u_min`, `u_max`, `eps` |

**Contract:** subclass `TimeParam`, set `u_min/u_max`, implement `u_to_t`/`t_to_u`. The
base provides `u_bounds(eps)` (buffered support) and `grid(N, device, eps)`. `eps`
defaults to `self.eps` and buffers the support inward away from the singular ends.

## t-distributions — `configs/t_distribution/`, `diffusion/time/uniform.py`
Chooses which timesteps training samples land on; returns `(t, u)`.

| name | class | samples uniformly in | support |
|---|---|---|---|
| `uniform_u` | `Uniform_u` | `u` | `time_param.u_bounds(eps)` |
| (code only) | `Uniform_t` | `t` | t-image of `time_param.u_bounds(eps)` |

Both **share the solver grid's buffered support** via `u_bounds`. `eps` defaults to
`time_param.eps` (one shared knob; the config omits it on purpose). **Contract:** class
`(device, time_param, eps=None)` with `.sample(shape) -> (t, u)`.

## Solvers — `configs/solver/`, `diffusion/solvers/`
Integrate backwards (`t=1 → t=0`) to generate. All take `(model, parameterization,
time_param, N, schedule=...)`, build their grid via `time_param.grid(N, device)`, feed
`u = t_to_u(t)` to the model, and accept a `time_grid=` override (used by reflow to
integrate *forwards*) and an `eps=` buffer.

| name | class | kind | update |
|---|---|---|---|
| `euler` | `EulerSolver` | deterministic PF-ODE | `x += dt · pfode(x, t, m)` |
| `ddim` | `DDIMSolver` | deterministic | `x = α(t_next)·x₀★ + σ(t_next)·ε★` from `to_x_0`/`to_noise` |
| `deis` | `DEISSolver` | deterministic, exponential multistep (order `M`) | integrating factor `E = α(t_i)/α(t_{i-1})` + Lagrange-weighted buffer of past `ε` predictions (weights via Gauss–Legendre) |
| `euler_mauryama` | `EulerMauryamaSolver` | stochastic reverse-SDE | `x += rev_sde·dt + gamma·√|dt|·noise` (no noise on last step) |

Notes:
- **DEIS** works in **noise (`ε`) space** and keeps a buffer of the last `M+1` noise
  predictions; `M=3` by default. It is the most code-heavy solver; the multistep weights
  `C_{i,j}` are precomputed numerically. It needs `schedule` (for `E` and `g²`).
- **Euler–Maruyama** is the only stochastic one and needs `schedule.gamma(t)`; it is
  **not invertible**, so reflow uses Euler for backward integration.

**Contract:** `sample(x0, return_trajectory=False, time_grid=None, ...)` returning the
final `x` (or `(x, trajectory)`); grid via `time_param.grid`; feed `u` to the model; use
`t` for converters/schedule.

## Models — `configs/model/`, `models/`
`nn.Module` with `forward(x, t, cond=None)` where `x:(B,2)`, `t:(B,1)` (the `t` arg is
actually the network coordinate `u`).

| name | class | notes |
|---|---|---|
| `beefier` | `models.beefier.SimpleModel` | **ACTIVE.** Fourier time-embed + Fourier features on `x` + residual SiLU blocks; optional class conditioning |
| `simple` | `models.model.SimpleModel` | legacy plain MLP |

**Conditioning:** `beefier` takes `num_classes`. If set, it builds
`nn.Embedding(num_classes + 1, time_emb_dim)` where index `num_classes` is the
**null/unconditional token**, added to the time embedding. `cond=None` resolves to the
null token, so a conditional-capable model still runs unconditionally (this is the case
under the `standard` proc even though the default `beefier` config sets `num_classes`).
`num_classes=None` is byte-for-byte the original unconditional model.

**`models/guidance.py::GuidedModel`** is a sampling-only wrapper (not a config target):
it holds the base model, a `(B,)` label buffer, a guidance scale `w`, and the null
index, and its `forward(x, t)` returns `m_uncond + w·(m_cond − m_uncond)`. Because solvers
call `self.model(x, t)`, it drops in as `solver.model` with no solver changes.

## Datasets — `configs/dataset/`, `data/`
`torch.utils.data.Dataset` exposing `self.X` as float32 `(N, 2)`.

| name | class | `__getitem__` returns | labels? |
|---|---|---|---|
| `blob` | `BlobDataset` | `(X[i], y[i])` | **yes** (`n_clusters` classes) — CFG-ready |
| `tree` | `TreeDataset` | `X[i]` | no (NVIDIA 2D fractal "tree" mixture) |

CFG needs per-sample labels, so it targets `blob`. The training loop and `StandardProc`
handle either shape (they drop labels when not needed).

## Prior — `configs/prior/`, `diffusion/dists_funcs/standard_normal.py`
`StandardNormal(dim).sample(n, device)` — the `t=1` sampling prior.

## Weight functions — `configs/weight_func/`, `diffusion/dists_funcs/weights.py`
Loss weights, resolved with `get_method` and called as `weight_func(t)`.

| name | function | value |
|---|---|---|
| `ones` | `one_func` | 1 |
| `linear` | `linear_weight` | `t²` |
| (code only) | `exponential` | `exp(t)` |

## LR schedulers — `configs/lr_scheduler/`, wrapped by `SchedulerWrapper`
Standard torch schedulers (`cosine`, `plateau`, `step`, `exponential`).
`diffusion/training/lr_schedulerwrapper.py::SchedulerWrapper` unifies `.step()` (passes
the val loss to `ReduceLROnPlateau`, ignores it otherwise) and `.get_last_lr()`.

## Diagnostics & viz
- `diffusion/utils/singularity.py` — `plot_singularity_report(schedule, out_path, grid,
  highlight)` plots `1/|det|` (conversion amplification) and `max(|a|,|b|)` (target
  scale) per parameterization vs `t`, with the solver grid overlaid → `singularity.png`.
- `diffusion/viz/viz_functions.py` (Agg backend) — `viz_training` (loss curves),
  `plot_prior` (assumed prior vs data pushed to `t=1`), `plot_transport` (noise→data
  trajectories), `plot_density_comparison` (true vs generated 2D-histogram heatmaps).
