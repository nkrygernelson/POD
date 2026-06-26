# Configuration (Hydra)

Every run is a composition of config groups under `configs/`, instantiated by their
`_target_`. The root is `configs/base.yaml`; you switch options by editing its
`defaults:` or overriding on the CLI.

## Root: `configs/base.yaml`

Current `defaults`:

```yaml
schedule: linear      solver: deis          generation: default
t_distribution: uniform_u                   time_param: logsnr
model: beefier        dataset: tree         lr_scheduler: cosine
parameterization: noise_pred                weight_func: ones
prior: standard_normal                      trainer: default
train_proc: standard
```

Top-level keys:

| key | default | meaning |
|---|---|---|
| `solver_steps` | 40 | `N`, interpolated into every `solver/*.yaml` |
| `seed` | 0 | RNG seed |
| `device` | cpu | torch device |
| `input_dim` / `output_dim` | 2 / 2 | data dimensionality |
| `num_classes` | 3 | class count; interpolated into `model` and `train_proc=cfg` |
| `viz` | True | write figures during train/generation |
| `run_name` | timestamp | Hydra run dir = `outputs/${run_name}` |

`num_classes` is the single knob shared by the model's conditioning embedding
(`beefier.yaml: num_classes: ${num_classes}`) and `CFGProc`. Set it to `null` for a
purely unconditional setup.

## Overriding

Edit `base.yaml`, or override per-run on the CLI (forwarded to both stages by `main.py`):

```bash
uv run python train.py parameterization=v_pred solver=euler time_param=identity
uv run python train.py train_proc=cfg dataset=blob num_classes=3 trainer.epochs=50
uv run python generation.py solver=ddim generation.guidance_scale=3 generation.target_class=1
```

Dotted paths reach into a group (`trainer.epochs=50`, `solver.M=4`,
`train_proc.n_steps=2`).

## Group reference

| group | options | key fields |
|---|---|---|
| `schedule` | `linear` | — |
| `parameterization` | `clean_pred`, `noise_pred`, `score_pred`, `v_pred` | — (schedule injected in code) |
| `time_param` | `identity`, `logsnr` | `eps`; `logsnr` adds `u_min`, `u_max` |
| `t_distribution` | `uniform_u` | none (eps shared via `time_param.eps`; `time_param`/`device` injected) |
| `solver` | `ddim`, `euler`, `deis`, `euler_mauryama` | `N: ${solver_steps}`; `deis` adds `M` |
| `model` | `beefier`, `simple` | `beefier`: `hidden_dim, n_blocks, time_emb_dim, x_fourier_*, time_fourier_scale, num_classes` |
| `dataset` | `blob` (labelled), `tree` | `blob`: `n_samples, n_clusters, cluster_std, random_state`; `tree`: `n_samples, classes, random_state` |
| `prior` | `standard_normal` | `dim: ${input_dim}` |
| `weight_func` | `ones`, `linear` | — (a function, resolved with `get_method`) |
| `lr_scheduler` | `cosine`, `plateau`, `step`, `exponential` | torch scheduler args |
| `trainer` | `default` | `train_split, batch_size, epochs, lr, save_final, save_every` |
| `train_proc` | `standard`, `reflow`, `cfg` | all have `kind`; `reflow`: `n_steps, teacher_path`; `cfg`: `p_uncond, num_classes` |
| `generation` | `default` | `trained_model_path, num_samples, guidance_scale, target_class` |

## Instantiation conventions (gotchas)

Some objects need a runtime dependency injected at `instantiate` time rather than from
YAML:

- `time_param`: `instantiate(cfg.time_param, schedule=sched)`.
- `t_distribution`: `instantiate(cfg.t_distribution, time_param=tp, device=device)`.
- `parameterization`: `instantiate(cfg.parameterization, schedule=sched,
  _recursive_=False)`.
- `train_proc`: `instantiate(cfg.train_proc, diffusion_process=DP, _recursive_=False)`.
- `weight_func`: resolved with `get_method(cfg.weight_func._target_)` — it's a plain
  function, not a class.
- `solver` (generation): `instantiate(cfg.solver, time_param=, schedule=,
  parameterization=, N=, model=)`.

`_recursive_=False` is used where a config field would otherwise be wrongly
auto-instantiated (e.g. the injected `schedule`/`diffusion_process`).

## Adding a new component

The recipe is always: **write the class/function → add a YAML under the right
`configs/<group>/` with a `_target_` → select it in `base.yaml` or via CLI.** See the
per-component contracts in [components.md](components.md). Remember to keep schedule math
in `t`, feed the network `u`, and convert only at the origins.
