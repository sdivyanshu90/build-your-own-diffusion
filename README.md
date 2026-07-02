# build-your-own-diffusion

A production-quality, from-scratch implementation of **denoising diffusion
models** in PyTorch: a timestep-conditioned **UNet** noise predictor plus
**DDPM** (Ho et al., 2020) and **DDIM** (Song et al., 2021) **schedulers**,
wrapped in a fully tested training/sampling toolkit called `diffusionlab`.

No `diffusers` dependency -- every equation is implemented, documented, and
verified here.

[![CI](https://github.com/sdivyanshu90/build-your-own-diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/sdivyanshu90/build-your-own-diffusion/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **UNet** with residual blocks, multi-head self-attention (flash-capable via
  `scaled_dot_product_attention`), GroupNorm, sinusoidal timestep embeddings,
  and zero-initialised residual branches for stable deep training.
- **Schedulers**: DDPM ancestral sampling and DDIM accelerated sampling
  (deterministic or stochastic via `eta`), with `linear`, `scaled_linear`,
  and `cosine` beta schedules computed in float64.
- **Three prediction parameterisations**: `epsilon`, `sample` (x0), and
  `v_prediction`.
- **Trainer**: step-based loop with EMA weights, mixed precision (fp16/bf16),
  gradient accumulation and clipping, LR warmup, JSONL metrics, in-training
  sample grids, and atomic, resumable, **pickle-free checkpoints**
  (`torch.load(weights_only=True)` works -- untrusted checkpoints cannot
  execute code).
- **Typed YAML configuration** with strict unknown-key detection and dotted
  CLI overrides.
- **175+ tests** asserting mathematical identities (not just shapes) at ~98%
  branch coverage, all CPU-friendly with zero dataset downloads.

## Quickstart

```bash
git clone https://github.com/sdivyanshu90/build-your-own-diffusion
cd build-your-own-diffusion
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 60-second smoke run on the built-in synthetic dataset (CPU is fine)
diffusionlab train --config configs/smoke.yaml

# Real experiment: MNIST
diffusionlab train --config configs/mnist.yaml

# Generate images from any checkpoint
diffusionlab sample \
  --checkpoint runs/mnist-ddpm/checkpoints/last.pt \
  --num-images 64 --steps 50 --seed 0 --output samples.png
```

Any config value can be overridden from the command line:

```bash
diffusionlab train --config configs/cifar10.yaml \
  --set training.max_steps=100000 \
  --set training.mixed_precision=bf16 \
  --set data.batch_size=64 \
  --set training.gradient_accumulation_steps=2
```

Resume an interrupted (or finished) run:

```bash
diffusionlab train --config configs/cifar10.yaml \
  --set training.max_steps=300000 \
  --resume runs/cifar10-ddpm/checkpoints/last.pt
```

## Python API

```python
import torch
from diffusionlab import Config, Trainer, DiffusionPipeline

# Train
config = Config.from_yaml("configs/mnist.yaml")
trainer = Trainer(config)
trainer.train()

# Sample
pipeline = DiffusionPipeline.from_checkpoint(
    "runs/mnist-ddpm/checkpoints/last.pt", device="auto", use_ema=True
)
images = pipeline.sample(16, num_inference_steps=50,
                         generator=torch.Generator().manual_seed(0))
# images: (16, 1, 32, 32) float tensor in [-1, 1]
```

## How it works (30 seconds)

Diffusion models learn to *reverse* a gradual noising process. The forward
process turns an image `x0` into pure noise over `T` steps in closed form:

```
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

The UNet is trained on a single MSE objective -- predict the noise `eps` from
`(x_t, t)`. Sampling starts from pure noise and repeatedly applies the
learned denoiser: DDPM takes all `T` stochastic steps; DDIM takes a short
deterministic subsequence (e.g. 50 steps) of the same trained model. The full
derivations live in [docs/mathematics.md](docs/mathematics.md).

## Repository layout

```
configs/                 Ready-to-run experiment configs (mnist, cifar10, smoke)
src/diffusionlab/
  config.py              Typed, validated, YAML-backed configuration
  data.py                Datasets (torchvision + synthetic) and dataloaders
  models/                UNet, residual/attention blocks, time embeddings
  schedulers/            Beta schedules, DDPM and DDIM samplers
  training/              EMA and the step-based Trainer
  pipeline.py            Checkpoint -> images sampling pipeline
  cli.py                 The `diffusionlab` command
tests/                   175+ tests (math identities, e2e train->sample, CLI)
docs/                    Architecture, math, configuration, operations, security
.github/workflows/       CI: lint, type-check, tests+coverage, smoke train
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | System design, diagrams, data flow, design rationale |
| [docs/mathematics.md](docs/mathematics.md) | DDPM/DDIM derivations mapped to code |
| [docs/configuration.md](docs/configuration.md) | Every config field, defaults, constraints |
| [docs/user-guide.md](docs/user-guide.md) | Training, sampling, resuming, datasets, performance tuning |
| [docs/operations.md](docs/operations.md) | Runbook: monitoring, troubleshooting, backup/DR, upgrades |
| [docs/security.md](docs/security.md) | Threat model, safe checkpoint loading, supply chain |
| [docs/api-reference.md](docs/api-reference.md) | Python API and CLI reference |
| [docs/development.md](docs/development.md) | Dev setup, testing strategy, CI/CD, release process |

## Development

```bash
make setup       # editable install + pre-commit hooks
make lint        # ruff check + format check
make typecheck   # mypy
make test        # pytest with coverage gate (>= 95%)
make smoke       # tiny end-to-end training run
```

## License

MIT -- see [LICENSE](LICENSE).
