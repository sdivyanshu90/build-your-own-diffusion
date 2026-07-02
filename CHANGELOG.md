# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-02

### Added

- Timestep-conditioned UNet (`diffusionlab.models`): residual blocks with
  zero-initialised branches, multi-head self-attention via
  `scaled_dot_product_attention`, GroupNorm/SiLU, sinusoidal time
  embeddings, configurable depth/width/attention placement.
- Schedulers (`diffusionlab.schedulers`): DDPM ancestral sampler and DDIM
  accelerated sampler (`eta`, leading/trailing spacing); `linear`,
  `scaled_linear`, and `cosine` beta schedules computed in float64;
  `epsilon` / `sample` / `v_prediction` parameterisations.
- Step-based `Trainer`: EMA weights, fp16/bf16 mixed precision, gradient
  accumulation and clipping, LR warmup, JSONL metrics, fixed-latent sample
  grids, atomic pickle-free checkpoints with full resume (including RNG
  state).
- `DiffusionPipeline` for checkpoint-to-images generation with
  `torch.load(weights_only=True)` safety.
- Typed, strictly-validated YAML configuration with dotted CLI overrides.
- Datasets: MNIST, Fashion-MNIST, CIFAR-10 (torchvision) and a procedural
  download-free `synthetic` dataset.
- CLI: `diffusionlab train | sample | validate-config`.
- Test suite: 175+ tests asserting mathematical identities and end-to-end
  behaviour at ~98% branch coverage (95% enforced).
- Tooling: ruff, mypy, pre-commit, Makefile, GitHub Actions CI (lint,
  types, tests on 3.10/3.12, CLI smoke train+sample), Dockerfile and
  docker-compose.
- Documentation: architecture (with diagrams), full mathematical
  derivations, configuration reference, user guide, operations runbook,
  security notes, API reference, development guide.
