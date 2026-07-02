# Configuration reference

All behaviour is driven by a single YAML file validated into the typed
`Config` tree (`src/diffusionlab/config.py`). Loading is **strict**: unknown
keys raise an error naming the section and listing the valid keys, so typos
fail immediately instead of silently using defaults.

```bash
diffusionlab validate-config --config configs/cifar10.yaml   # check without running
```

Any field can be overridden on the command line with dotted paths; values
are YAML-parsed (`true`, `0.1`, `[1, 2, 4]` all work):

```bash
diffusionlab train --config configs/cifar10.yaml \
  --set training.max_steps=1000 --set model.channel_multipliers=[1,2,4]
```

## Top level

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `run_name` | str | `diffusion-run` | Non-empty, no `/`; artifacts land in `<output_dir>/<run_name>` |

## `model` -- UNet architecture

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `in_channels` | int | 3 | Must equal the dataset's channels (1 for MNIST, 3 for CIFAR/synthetic); also the output channel count |
| `base_channels` | int | 128 | Width of level 0; must be divisible by `num_groups` |
| `channel_multipliers` | list[int] | `[1,2,2,2]` | One entry per resolution level; level i has `base_channels * multipliers[i]` channels; each extra level halves resolution once |
| `num_res_blocks` | int | 2 | Residual blocks per encoder level (decoder uses one more per level) |
| `attention_levels` | list[int] | `[1]` | 0-based level indices that get self-attention after each ResBlock; the bottleneck always has attention |
| `num_heads` | int | 4 | Must divide the channel count at every attention site |
| `dropout` | float | 0.1 | In `[0, 1)`; applied inside residual blocks |
| `num_groups` | int | 8 | GroupNorm groups; must divide `base_channels` |

Parameter-count intuition: CIFAR config (`128, [1,2,2,2]`) is ~57M params;
MNIST config (`64, [1,2,4]`) is ~9M; the smoke config is ~0.2M.

## `diffusion` -- process and sampler

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `num_train_timesteps` | int | 1000 | Chain length T; >= 2 |
| `beta_schedule` | str | `cosine` | `linear` \| `scaled_linear` \| `cosine` |
| `beta_start` / `beta_end` | float | 1e-4 / 0.02 | `(scaled_)linear` only; `0 < start < end < 1` |
| `prediction_type` | str | `epsilon` | `epsilon` \| `sample` \| `v_prediction` |
| `clip_sample` | bool | true | Clip predicted x0 to [-1,1] during sampling (recommended for images) |
| `variance_type` | str | `fixed_small` | DDPM only: `fixed_small` (posterior) \| `fixed_large` (beta_t) |
| `sampler` | str | `ddim` | Default sampler for generation; `ddpm` \| `ddim` |
| `num_inference_steps` | int | 50 | `[1, T]`; **must equal T when `sampler=ddpm`** |
| `ddim_eta` | float | 0.0 | >= 0; 0 = deterministic, 1 = DDPM-equivalent variance |
| `timestep_spacing` | str | `leading` | DDIM subsequence choice: `leading` \| `trailing` (better at very few steps) |

## `data` -- dataset and loading

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `dataset` | str | `cifar10` | `mnist` \| `fashion_mnist` \| `cifar10` \| `synthetic` (procedural, no download) |
| `data_dir` | str | `./data` | torchvision download/cache root |
| `image_size` | int | 32 | >= 4 and divisible by `2^(len(channel_multipliers)-1)` |
| `batch_size` | int | 128 | Per optimizer step, before gradient accumulation |
| `num_workers` | int | 4 | DataLoader workers; use 0 in tests/containers with tiny data |
| `horizontal_flip` | bool | true | Disable for digits |
| `download` | bool | true | Allow torchvision downloads |
| `synthetic_size` | int | 1024 | Sample count of the synthetic dataset |

## `optim` -- optimizer and EMA

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `lr` | float | 2e-4 | Peak AdamW LR (after warmup) |
| `weight_decay` | float | 0.0 | AdamW decoupled decay |
| `betas` | [float,float] | `[0.9, 0.999]` | Each in [0,1) |
| `warmup_steps` | int | 500 | Linear warmup; 0 disables |
| `grad_clip_norm` | float | 1.0 | Global-norm clip; 0 disables (norm still logged) |
| `ema_decay` | float | 0.9999 | In [0,1); use ~0.999 for runs under ~50k steps |

## `training` -- loop control

| Key | Type | Default | Constraints / notes |
| --- | --- | --- | --- |
| `max_steps` | int | 100000 | Total optimizer steps (resume continues toward this) |
| `gradient_accumulation_steps` | int | 1 | Micro-batches per step; effective batch = `batch_size * this` |
| `mixed_precision` | str | `none` | `none` \| `fp16` (CUDA only) \| `bf16` |
| `log_interval` | int | 50 | Steps between metric records |
| `sample_interval` | int | 2000 | Steps between EMA preview grids |
| `checkpoint_interval` | int | 10000 | Steps between numbered checkpoints (+ rolling `last.pt`) |
| `num_sample_images` | int | 64 | Images per preview grid |
| `output_dir` | str | `runs` | Root for run directories |
| `seed` | int | 42 | Seeds Python, torch CPU and CUDA |
| `device` | str | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |

## Cross-field rules enforced by `Config.validate`

1. `model.in_channels` must match the dataset's channel count.
2. `data.image_size` must be divisible by `2^(levels-1)` so every
   downsampling halving is exact.
3. `model.base_channels % model.num_groups == 0` (GroupNorm needs it; all
   deeper widths are multiples of `base_channels`, so they inherit it).
4. Every attention site's channels must be divisible by `num_heads`
   (includes the bottleneck at `base * multipliers[-1]`).
5. `sampler=ddpm` requires `num_inference_steps == num_train_timesteps`.
6. `attention_levels` must be valid level indices.

## Run directory layout

```
runs/<run_name>/
  config.yaml        exact snapshot of the effective config
  train.log          human-readable log (copy of console output)
  metrics.jsonl      one JSON object per log_interval (step, loss, lr, grad_norm, seconds_per_step)
  checkpoints/
    step_00010000.pt numbered checkpoints
    last.pt          rolling latest (use this to resume)
  samples/
    step_00010000.png EMA preview grids (same latent seeds every time)
```
