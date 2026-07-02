# Operations runbook

Operating `diffusionlab` training jobs in production: deployment, monitoring,
alerting, backup/recovery, upgrades, and a troubleshooting playbook.

## 1. Deployment

### Docker

```bash
docker build -t diffusionlab .
# GPU training (requires nvidia-container-toolkit)
docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs \
  diffusionlab train --config configs/cifar10.yaml
# CPU-only image
docker build --build-arg BASE_IMAGE=python:3.11-slim -t diffusionlab:cpu .
```

Or with compose (`docker-compose.yml` defines `train`, `smoke`, and `sample`
services with the volumes and GPU reservation prewired):

```bash
docker compose run --rm smoke     # verify the image end-to-end
docker compose run --rm train     # long training job
```

### Job-runner guidance (Kubernetes/Batch/Slurm)

Run training as a **Job** (not a Deployment -- it terminates). Checklist:

- Mount persistent volumes at `/app/data` (datasets) and `/app/runs`
  (artifacts). Everything else in the container is disposable.
- Request 1 GPU, and CPU >= `data.num_workers + 2`.
- Set `restartPolicy: OnFailure` and make the container command resume
  automatically:
  `diffusionlab train --config ... --resume runs/<name>/checkpoints/last.pt`
  (if the file does not exist yet, the CLI exits 2; start the very first run
  without `--resume`, or wrap with a shell check:
  `[ -f runs/x/checkpoints/last.pt ] && ARGS="--resume ..."`).
- Logs go to stdout (twelve-factor); collect with your standard log stack.

## 2. Monitoring and alerting

### Signals

| Signal | Source | Healthy | Alert when |
| --- | --- | --- | --- |
| Process liveness | job runner | running/completed | restarts > 3/hour |
| `loss` | `metrics.jsonl` | falls fast, then slow monotone-ish drift | NaN (process exits non-zero), or > 2x its trailing-1h median |
| `grad_norm` | `metrics.jsonl` | O(clip value) | sustained 10x growth (divergence precursor) |
| `seconds_per_step` | `metrics.jsonl` | flat | > 1.5x baseline (data stall, thermal throttling, contention) |
| Checkpoint freshness | `checkpoints/last.pt` mtime | < 2 x checkpoint_interval x s/step | stale (job wedged) |
| Sample grids | `samples/*.png` | progressive sharpening | pure-noise grids late in training |
| Disk | volume metrics | headroom > 3 checkpoints | otherwise (a 57M-param checkpoint is ~700MB with optimizer state) |

`metrics.jsonl` is append-only JSON-lines precisely so a sidecar/agent can
tail it into Prometheus/CloudWatch/etc. without library integration:

```bash
tail -f runs/cifar10-ddpm/metrics.jsonl | jq -r '"loss=\(.loss) step=\(.step)"'
```

### Exit codes (CLI)

| Code | Meaning |
| --- | --- |
| 0 | success |
| 2 | configuration/input error (bad config, missing file, bad checkpoint) |
| 130 | interrupted (SIGINT) |
| other non-zero | unhandled failure (e.g. non-finite loss RuntimeError, OOM) -- see traceback in logs |

## 3. Backup and disaster recovery

A run is fully reconstructible from **`config.yaml` + any checkpoint**;
treat those as the crown jewels and everything else as cache.

- **Backup cadence:** sync `runs/<name>/checkpoints/` and
  `runs/<name>/config.yaml` to object storage on the checkpoint cadence,
  e.g. a cron alongside the job:
  `aws s3 sync runs/ s3://bucket/diffusion-runs/ --exclude "*" --include "*/checkpoints/*" --include "*/config.yaml"`.
- **Retention:** keep `last.pt` + every Nth numbered checkpoint. Numbered
  checkpoints let you roll back a run that later diverged.
- **Recovery drill:** on a fresh machine,
  `pip install -e . && diffusionlab train --config config.yaml --resume last.pt`
  must resume at the recorded step (this exact path is covered by the test
  suite).
- **Corruption safety:** checkpoints are written atomically
  (`tmp` + `os.replace`), so a crash never truncates `step_*.pt`; at worst
  `last.pt` is one interval old.

## 4. Upgrades and maintenance

- Checkpoints carry `format_version: 1`. Any future change to checkpoint
  layout must bump it and add a loader branch; loaders reject missing keys
  today.
- Upgrading torch: run `make test` first -- the suite pins the behavioural
  contract (schedule math, AMP paths, `weights_only` loading), so most
  incompatibilities surface immediately.
- Dependency updates: Dependabot-style bumps should ride through CI (lint +
  types + tests + smoke train) before merge.
- The config schema is validated strictly, so renamed/removed fields fail
  loudly at load time; keep `configs/*.yaml` in the same PR as schema
  changes.

## 5. Troubleshooting playbook

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `RuntimeError: non-finite loss at step N` | LR too high; fp16 overflow; corrupt data | Lower `optim.lr`, raise `warmup_steps`, prefer `bf16` over `fp16`; check the data pipeline outputs are in [-1,1] |
| CUDA out of memory | batch too large for GPU | Lower `data.batch_size`, compensate with `gradient_accumulation_steps`; enable mixed precision |
| `ConfigError: unknown key(s) [...]` | typo in YAML/override | The message lists valid keys for that section |
| `ConfigError: ... must be divisible by ...` | image size vs UNet depth, groups, or heads mismatch | See cross-field rules in configuration.md |
| `DDPM requires num_inference_steps == num_train_timesteps` | subsampled DDPM chain | Use `--sampler ddim` or `--steps <T>` |
| Samples are pure noise late in training | too few steps for model/data scale; LR spike killed the run | Compare grids over time; roll back to an earlier numbered checkpoint and resume with lower LR |
| Samples are blurry | too few inference steps; online instead of EMA weights | Raise `--steps`; ensure EMA weights are used (default) |
| First GPU sampling step extremely slow | CUDA kernel autotune/compile on first call (esp. WSL2) | Expected one-time cost; subsequent steps are fast |
| `device 'cuda' requested but CUDA is not available` | wrong image/host, missing `--gpus all` | Use the CUDA base image + NVIDIA runtime, or set `training.device=cpu` |
| Dataloader hangs in containers | too many workers vs shm/CPU limit | Set `data.num_workers=0` (tiny data) or raise `--shm-size` |
| Resume warning: config differs from checkpoint | intentional (e.g. extended max_steps) or accidental drift | Diff your config against the run's `config.yaml` snapshot |
| Downloads blocked in restricted env | dataset auto-download disabled by policy | Pre-stage `data/` volume; set `data.download=false` |

## 6. Capacity planning quick numbers

- CIFAR-10 config (~57M params): ~1.4GB GPU memory for weights+EMA+Adam in
  fp32 before activations; batch 128 at 32px fits comfortably in 8-12GB
  with mixed precision.
- Checkpoint size ~= 12 bytes/param (fp32 weights + EMA + 2 Adam moments)
  -> ~700MB for the CIFAR config, ~110MB for MNIST.
- Sampling: cost = `num_images x steps` UNet forwards; batch them (a single
  `sample(64, steps=100)` call is one 64-wide batch per step).
