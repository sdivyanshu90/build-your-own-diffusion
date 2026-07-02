# Security

The security posture of an ML training library differs from a web service:
there is no network listener, no authentication surface, no database, and no
untrusted multi-tenant input at runtime. The realistic threat model is
**malicious artifacts** (checkpoints, configs, datasets) and **supply
chain**. This document covers both, plus the residual hardening applied.

## Threat model

| Asset | Threat | Mitigation |
| --- | --- | --- |
| Host running `sample`/`train` | Arbitrary code execution via a malicious checkpoint | Pickle-free checkpoints (below) |
| Host | Malicious YAML config | `yaml.safe_load` everywhere -- no object construction from YAML |
| Training result | Poisoned dataset | Out of scope for the library; pin dataset sources and checksums in your pipeline |
| Artifacts | Tampering in storage | Store checkpoints with object-store integrity (ETag/KMS); optionally record SHA-256 alongside |
| Build | Compromised dependency | Small pinned dependency set, lockable via pip-tools/uv; CI installs from pinned wheels index |

## Checkpoint safety (the important one)

`torch.load` with default settings unpickles arbitrary Python objects --
loading an untrusted checkpoint is code execution. `diffusionlab` closes
this by construction:

- The trainer serialises **only primitives and tensors**: state dicts, the
  config as a plain nested dict (never a dataclass instance), RNG states as
  tensors/ints. Nothing in a checkpoint requires pickle semantics.
- Every load site uses `torch.load(..., weights_only=True)`
  (`pipeline.load_checkpoint`), which restricts deserialisation to a safe
  allowlist. A checkpoint that needs more **fails to load** rather than
  executing.
- Required keys are validated after load, and EMA state is validated for
  name *and shape* agreement before use.
- The regression test `test_checkpoint_loads_with_weights_only` pins this
  contract; a change that reintroduces pickled objects breaks CI.

## Input validation

- Configs: strict schema (unknown keys rejected), typed coercion, and
  cross-field validation with descriptive errors (`config.py`).
- CLI: argparse-typed flags; all user-facing errors map to exit code 2
  without tracebacks-as-UX.
- Tensors: shape/divisibility checks at the UNet boundary; scheduler `step`
  rejects timesteps outside the prepared sequence.

## Secrets and configuration hygiene

The library needs **no secrets**: no API keys, no tokens, no credentials.
Consequently:

- Nothing reads environment variables for auth; there is no `.env` handling
  to get wrong. If you wrap this in infrastructure that has secrets (e.g.
  S3 backup credentials), supply them through your platform's secret store,
  never through the config YAML (which is snapshotted into every run
  directory and checkpoint).
- Logs and metrics contain hyperparameters and loss values only.

## Supply chain

- Runtime dependencies are four, all mainstream: `torch`, `torchvision`,
  `PyYAML`, `tqdm`. Review surface is deliberately small (no `diffusers`,
  no config framework).
- CI runs on pinned GitHub Actions (major-version tags) and installs torch
  from the official CPU wheel index.
- `pip-audit`/Dependabot integration is recommended on the repository; the
  pyproject bounds are compatible ranges, and a lockfile
  (`pip compile pyproject.toml`) can be added for byte-reproducible envs.
- Docker: images build from the official `pytorch/pytorch` (or
  `python:slim`) base; the app runs as a non-root user.

## Residual notes

- `save_image_grid` writes only to paths given by the operator; the library
  never fetches URLs and makes no network calls except torchvision dataset
  downloads, which can be disabled (`data.download=false`).
- The synthetic dataset makes the *entire* test suite network-free, so CI
  policy can block egress.
- OWASP web categories (XSS/CSRF/SQLi/SSRF) do not apply -- there is no web
  surface, no SQL, and no outbound fetch of user-supplied URLs. If you build
  a generation API on top of `DiffusionPipeline`, apply your service
  framework's standard controls (authn/z, rate limiting, input validation
  of `num_images`/`steps` bounds) at that layer.
