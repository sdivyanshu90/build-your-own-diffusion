# Development guide

## Setup

```bash
git clone https://github.com/sdivyanshu90/build-your-own-diffusion
cd build-your-own-diffusion
python3 -m venv .venv && source .venv/bin/activate
make setup            # pip install -e ".[dev]" + pre-commit install
```

Day-to-day loop:

```bash
make lint             # ruff check + ruff format --check
make format           # apply formatting + safe autofixes
make typecheck        # mypy (src, strict-ish)
make test             # pytest + coverage gate (>= 95%)
make smoke            # 20-step end-to-end training run (synthetic data)
make check            # lint + typecheck + test (what CI runs)
```

## Testing strategy

The suite (in `tests/`) is designed around three principles:

1. **Assert mathematical identities, not just shapes.** Examples: DDIM's
   `eta=1` variance must equal the DDPM posterior variance exactly for all
   t; feeding the true training target through `predict_original_sample`
   must recover `x_0` for every prediction type; the cosine schedule's
   cumulative product must reproduce its defining `alpha_bar(t)` formula.
   A sign or off-by-one error in the schedule math cannot pass.
2. **Everything runs on CPU in seconds with zero downloads.** The
   procedural `synthetic` dataset and a ~0.2M-param UNet make full
   train -> checkpoint -> resume -> sample cycles cheap; torchvision dataset
   code is exercised through stub factories.
3. **Behavioural end-to-end coverage.** The CLI is tested through
   `main(argv)` (train, resume, sample, error paths and exit codes); the
   trainer's artifacts (metrics JSONL, config snapshot, atomic checkpoints,
   EMA divergence from online weights) are asserted from disk; a small
   overfitting test proves gradients and timestep conditioning actually
   learn.

Layout: one test module per source area (`test_config`, `test_schedules`,
`test_schedulers`, `test_models`, `test_ema`, `test_data`, `test_utils`,
`test_trainer`, `test_pipeline`, `test_cli`, `test_edge_paths`). Shared
fixtures live in `tests/conftest.py`; the session-scoped `trained_run`
fixture trains once and is shared by trainer/pipeline/CLI tests.

Coverage: ~98% branch coverage, enforced at 95% (`fail_under` in
pyproject). `__main__.py` is exercised via a subprocess test and excluded
from the (in-process) coverage measurement.

Run subsets:

```bash
pytest tests/test_schedulers.py -q          # one area
pytest -k ddim -q                           # by keyword
pytest -q -m "not slow"                     # marker reserved for future long tests
```

## Code style and conventions

- Formatting/linting: ruff (line length 100, py310 target); run via
  pre-commit or `make format`.
- Types: full annotations on all public functions; mypy runs on `src` with
  `disallow_untyped_defs`. (`warn_return_any` is off because torch stubs
  type `nn.Module.__call__` as `Any`.)
- Docstrings: Google-style on every public class/function; module
  docstrings explain *design rationale*, not just contents.
- Model-space convention: images are `[-1, 1]` floats everywhere inside the
  library; conversion happens only at the edges (`utils/image.py`, data
  transforms).
- Errors: user-facing configuration problems raise `ConfigError` (exit code
  2 in the CLI); internal invariant violations use assert/RuntimeError.

## CI/CD

`.github/workflows/ci.yml` runs on pushes and PRs to `main`:

1. **lint** -- ruff check + format check.
2. **typecheck** -- mypy.
3. **test** (Python 3.10 and 3.12) -- pytest with the 95% coverage gate,
   using CPU-only torch wheels; coverage XML uploaded as an artifact.
4. **smoke** -- installs the package and runs the real CLI end-to-end:
   `train` on `configs/smoke.yaml`, then `sample` from the produced
   checkpoint (catches packaging/entry-point breakage unit tests can miss).

Suggested release flow: bump `__version__` and `pyproject.toml` version
together, update `CHANGELOG.md`, tag `vX.Y.Z`; `python -m build` produces
the wheel/sdist.

## Extension points

| Goal | Where |
| --- | --- |
| New sampler (Euler, DPM-Solver, ...) | Subclass `BaseScheduler` (implement `set_timesteps` + `step`), register in `_SCHEDULER_REGISTRY`, add the name to config validation |
| New beta schedule | Add a function to `schedulers/schedules.py`, register in `_SCHEDULE_REGISTRY`, extend `_BETA_SCHEDULES` in config.py |
| New dataset | See "Adding your own dataset" in user-guide.md |
| Metrics sink (wandb/TensorBoard) | Wrap or replace `JsonlMetricsWriter` in `utils/logging.py`; the trainer only calls `.log(dict)` |
| Class-conditional generation | Extend `UNet.forward` with a label embedding added to the time embedding; thread labels through `Trainer._compute_loss` |
| Multi-GPU | Wrap `Trainer.model` in DDP + shard the dataloader; isolated to `trainer.py` |
