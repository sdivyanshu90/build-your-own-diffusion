# API reference

The public Python API is everything exported from `diffusionlab`
(`from diffusionlab import ...`). Full parameter documentation lives in the
docstrings (all public classes/functions are documented); this page is the
orientation map with the signatures you will actually type.

## Package exports

```python
from diffusionlab import (
    # configuration
    Config, ConfigError,
    ModelConfig, DiffusionConfig, DataConfig, OptimConfig, TrainingConfig,
    # model
    UNet,
    # schedulers
    BaseScheduler, DDPMScheduler, DDIMScheduler, SchedulerOutput, build_scheduler,
    # training
    Trainer, ExponentialMovingAverage,
    # inference
    DiffusionPipeline,
    __version__,
)
```

## Configuration (`diffusionlab.config`)

```python
Config.from_yaml(path, overrides=None) -> Config      # strict; raises ConfigError
Config.from_dict(data: dict) -> Config                # strict; validates
config.to_dict() -> dict                              # YAML/JSON-safe primitives
config.save_yaml(path)                                # run-directory snapshots
config.validate()                                     # raises ConfigError with field name
```

Overrides are `"dotted.path=value"` strings with YAML-parsed values, e.g.
`Config.from_yaml("c.yaml", overrides=["training.max_steps=100"])`.

## Model (`diffusionlab.models`)

```python
model = UNet(config.model)                       # nn.Module
pred = model(x, timesteps)                       # x: (B,C,H,W); t: (B,) or scalar tensor
```

Output shape == input shape; meaning (epsilon / x0 / v) is defined by
`diffusion.prediction_type`. Building blocks (`ResidualBlock`,
`AttentionBlock`, `Downsample`, `Upsample`, `SinusoidalPositionalEmbedding`,
`TimestepEmbedSequential`) are exported for reuse and experimentation.

## Schedulers (`diffusionlab.schedulers`)

```python
scheduler = build_scheduler(config.diffusion)                # or sampler="ddpm"/"ddim"

# Training-side (t may be a (B,) tensor):
x_t    = scheduler.add_noise(x0, noise, t)
target = scheduler.training_target(x0, noise, t)
v      = scheduler.get_velocity(x0, noise, t)

# Sampling-side (t is a python int from scheduler.timesteps):
scheduler.set_timesteps(num_inference_steps)
for t in scheduler.timesteps.tolist():                       # descending
    out = scheduler.step(model_output, t, x, generator=g)    # SchedulerOutput
    x = out.prev_sample                                      # out.pred_original_sample also available

# Conversions:
x0_hat  = scheduler.predict_original_sample(model_output, t, x_t, clip=None)
eps_hat = scheduler.predict_epsilon(x_t, t, x0_hat)
```

`DDPMScheduler.set_timesteps` requires the full chain length;
`DDIMScheduler` accepts any `1 <= n <= T` and honours
`ddim_eta`/`timestep_spacing`. Schedule tensors (`betas`, `alphas_cumprod`,
`sqrt_alphas_cumprod`, ...) are float32 CPU attributes.

## Training (`diffusionlab.training`)

```python
trainer = Trainer(config, resume_from=None)   # builds model/optimizer/data; creates run_dir
trainer.train() -> dict                       # runs to training.max_steps; returns last metrics
trainer.save_checkpoint() -> Path             # atomic; also refreshes last.pt
trainer.close()                               # release metrics file handle
trainer.step                                  # completed optimizer steps
trainer.run_dir / trainer.checkpoint_dir / trainer.sample_dir
```

```python
ema = ExponentialMovingAverage(model, decay=0.9999)
ema.update(model)          # after each optimizer step
ema.store(model); ema.copy_to(model)   # swap EMA weights in ...
ema.restore(model)                     # ... and back
ema.state_dict() / ema.load_state_dict(state)  # validates names AND shapes
```

## Inference (`diffusionlab.pipeline`)

```python
pipe = DiffusionPipeline.from_checkpoint(
    "runs/x/checkpoints/last.pt",
    device="auto",          # "auto" | "cpu" | "cuda" | "cuda:N"
    use_ema=True,           # EMA weights sample better; False for the online weights
    sampler=None,           # override "ddpm"/"ddim"
)
images = pipe.sample(
    num_images=16,
    num_inference_steps=None,   # default: config's num_inference_steps
    generator=torch.Generator(pipe.device.type).manual_seed(0),
    progress=False,
)                                # (N, C, H, W) float in [-1, 1], on pipe.device
```

`load_checkpoint(path) -> dict` is the safe (`weights_only=True`) low-level
loader with required-key validation.

## Utilities (`diffusionlab.utils`)

```python
seed_everything(seed)                        # python + torch + cuda
device = get_device("auto")                  # raises if CUDA requested but absent
n = count_parameters(model)                  # trainable_only=True by default
img8 = to_uint8(images)                      # [-1,1] float -> [0,255] uint8
save_image_grid(images, "grid.png", nrow=8)  # creates parents; returns Path
setup_logging(level, log_file=None)          # idempotent "diffusionlab" logger tree
JsonlMetricsWriter(path).log({...})          # append-only, flushed per record
```

## CLI

```
diffusionlab --version
diffusionlab train           --config FILE [--set K=V ...] [--resume CKPT]
diffusionlab sample          --checkpoint CKPT [--output PNG] [--num-images N]
                             [--steps N] [--sampler ddpm|ddim] [--seed S]
                             [--grid-cols N] [--device D] [--no-ema]
diffusionlab validate-config --config FILE [--set K=V ...]
```

Exit codes: `0` success, `2` configuration/input error, `130` interrupted.
`python -m diffusionlab ...` is equivalent to the console script.

## Stability policy

Everything importable from the top-level `diffusionlab` package plus the CLI
surface is covered by semantic versioning; checkpoint files carry
`format_version` for forward-compatible evolution. Module-internal helpers
prefixed with `_` are not API.
