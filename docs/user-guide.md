# User guide

How to install, train, monitor, resume, and sample. For every config field
see [configuration.md](configuration.md); for operating long runs in
production see [operations.md](operations.md).

## Installation

Requirements: Python >= 3.10. A CUDA GPU is strongly recommended for real
datasets but nothing requires one.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime only
pip install -e ".[dev]"     # + test/lint tooling
```

For GPU wheels matching your CUDA version, install torch first following
[pytorch.org](https://pytorch.org/get-started/locally/), then `pip install -e .`.

Verify the install:

```bash
diffusionlab --version
diffusionlab train --config configs/smoke.yaml   # < 1 minute, CPU OK
```

## Training

### First real run: MNIST

```bash
diffusionlab train --config configs/mnist.yaml
```

This downloads MNIST into `./data`, trains a ~9M-parameter UNet for 30k
steps, and writes everything under `runs/mnist-ddpm/`. Watch the preview
grids in `runs/mnist-ddpm/samples/` -- the *same* latent noise is denoised
for every grid, so you can judge progress by watching identical slots
sharpen over time. Digit-like shapes usually appear within a few thousand
steps on GPU.

### CIFAR-10

```bash
diffusionlab train --config configs/cifar10.yaml \
  --set training.mixed_precision=bf16     # if your GPU supports it
```

Budget guidance: recognisable objects by ~50k steps; the config's 200k steps
gives good quality; the literature uses 800k for benchmark FID numbers.

### Fitting into GPU memory

Keep the *effective* batch (`batch_size x gradient_accumulation_steps`)
constant while lowering per-step memory:

```bash
diffusionlab train --config configs/cifar10.yaml \
  --set data.batch_size=32 \
  --set training.gradient_accumulation_steps=4
```

Other levers, in order of preference: `mixed_precision` (`bf16` if
supported, else `fp16`), smaller `base_channels`, fewer
`attention_levels`.

### Monitoring a run

- **Console / `train.log`**: step, loss, LR, gradient norm.
- **`metrics.jsonl`**: machine-readable; e.g.
  ```bash
  jq -r '[.step, .loss] | @tsv' runs/cifar10-ddpm/metrics.jsonl
  ```
  or in Python: `pd.read_json("metrics.jsonl", lines=True).plot(x="step", y="loss")`.
- **`samples/*.png`**: the qualitative ground truth. Loss curves flatten
  early in diffusion training; the grids keep improving long after -- trust
  the grids.

What healthy training looks like: loss drops fast from ~1.0 and then creeps
down very slowly (most of the improvement hides in low-noise timesteps);
gradient norm settles around or below the clip value.

### Resuming and extending

`last.pt` contains model, EMA, optimizer, scaler, step counter, and RNG
states:

```bash
# Continue an interrupted run
diffusionlab train --config configs/cifar10.yaml \
  --resume runs/cifar10-ddpm/checkpoints/last.pt

# Extend a finished run to 300k steps
diffusionlab train --config configs/cifar10.yaml \
  --set training.max_steps=300000 \
  --resume runs/cifar10-ddpm/checkpoints/last.pt
```

Hyperparameters come from the config you pass (so you *can* change LR or
intervals on resume); weights and optimizer state come from the checkpoint.
A config mismatch with the checkpoint's embedded config logs a warning.

## Sampling

```bash
diffusionlab sample \
  --checkpoint runs/cifar10-ddpm/checkpoints/last.pt \
  --num-images 64 --grid-cols 8 \
  --steps 100 \
  --seed 0 \
  --output cifar_samples.png
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--steps N` | DDIM inference steps; 50-100 is a good quality/speed range, 250+ for best quality |
| `--sampler ddpm` | Full-chain ancestral sampling (requires `--steps` = T); marginally different texture, ~10-20x slower |
| `--seed S` | Reproducible generation (same seed -> same images with eta=0 DDIM) |
| `--no-ema` | Use raw online weights instead of EMA (mainly for debugging -- EMA is better) |
| `--device cpu` | Force CPU |

From Python:

```python
from diffusionlab import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_checkpoint("runs/cifar10-ddpm/checkpoints/last.pt")
imgs = pipe.sample(16, num_inference_steps=100,
                   generator=torch.Generator(pipe.device.type).manual_seed(0),
                   progress=True)          # (16, 3, 32, 32) in [-1, 1]

from diffusionlab.utils import save_image_grid, to_uint8
save_image_grid(imgs, "out.png", nrow=4)
arr = to_uint8(imgs).permute(0, 2, 3, 1).cpu().numpy()  # HWC uint8 for PIL/np
```

## Datasets

Built-ins: `mnist`, `fashion_mnist`, `cifar10` (torchvision, auto-download),
and `synthetic` (procedural shapes; instant, deterministic, great for
pipeline debugging).

**Adding your own dataset** (three small changes):

1. `data.py`: add a factory in `build_dataset` returning a `Dataset` whose
   `__getitem__` yields a `(C, H, W)` float tensor in `[-1, 1]` (wrap
   labelled datasets in `ImagesOnly`).
2. `config.py`: add the name and channel count to `DATASET_CHANNELS`.
3. Add a config YAML with matching `model.in_channels` and an `image_size`
   divisible by `2^(levels-1)`.

## Choosing hyperparameters

| Situation | Recommendation |
| --- | --- |
| New dataset, first attempt | Start from `configs/cifar10.yaml`; scale `base_channels`/levels with resolution (add one level per 2x above 32px) |
| Short runs (< 50k steps) | `ema_decay: 0.999` (0.9999 barely moves in short runs) |
| Few-step sampling matters | Train with `prediction_type: v_prediction` and sample with `timestep_spacing: trailing` |
| Loss spikes / NaN with fp16 | Switch to `bf16`, or lower `lr`, or raise `warmup_steps` |
| Overfitting a small dataset | Raise `dropout` to 0.2-0.3, keep `horizontal_flip: true` |
