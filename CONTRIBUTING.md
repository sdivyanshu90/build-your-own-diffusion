# Contributing

Thanks for considering a contribution! This project aims to stay a
*readable, verified* reference implementation, so correctness and clarity
outrank feature count.

## Workflow

1. Fork/branch from `main`.
2. `make setup` (editable install + pre-commit hooks).
3. Make your change, including tests and doc updates in the same PR.
4. `make check` must pass (identical to CI: ruff, mypy, pytest with the
   95% coverage gate).
5. Open a PR with a description of *why*, not just *what*.

## Ground rules

- **Every mathematical claim gets a test.** New samplers/schedules must
  assert identities or reference values, not just output shapes (see
  `tests/test_schedulers.py` for the pattern).
- **Tests stay fast and offline.** CPU-only, seconds not minutes, no
  dataset downloads (use `synthetic` or stub factories).
- **Config changes are strict-schema changes.** Update
  `config.py` validation, `docs/configuration.md`, and the example YAMLs
  together.
- **Checkpoint compatibility matters.** Anything that changes the
  checkpoint layout must bump `format_version` and keep
  `torch.load(weights_only=True)` working.
- Full type annotations and Google-style docstrings on public API.

## Good first contributions

- Additional samplers (Euler ancestral, DPM-Solver++, Heun) behind the
  existing `BaseScheduler` protocol.
- A `sigmoid` beta schedule.
- Class-conditional training (label embedding added to the time embedding).
- A wandb/TensorBoard metrics sink alongside `JsonlMetricsWriter`.

## Reporting bugs

Open an issue with: package version, `python --version`,
`torch.__version__`, the exact command/config (attach the run's
`config.yaml`), and the full traceback. For sample-quality issues, attach a
couple of `samples/step_*.png` grids and the `metrics.jsonl`.
