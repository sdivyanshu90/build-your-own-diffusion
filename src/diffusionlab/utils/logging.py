"""Logging strategy: human logs via :mod:`logging`, metrics via JSONL.

Two channels with different consumers:

- The standard :mod:`logging` tree rooted at ``diffusionlab`` carries
  human-readable progress and warnings (console and, during training, a
  per-run ``train.log`` file).
- :class:`JsonlMetricsWriter` appends one JSON object per metric event to a
  per-run ``metrics.jsonl`` -- machine-readable, append-only, trivially
  ingested by pandas, jq, or a metrics forwarder in production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import TracebackType
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure the ``diffusionlab`` logger tree.

    Idempotent: existing handlers are replaced, so calling this from both
    the CLI and a notebook never duplicates output.

    Args:
        level: Minimum level for all handlers.
        log_file: Optional file that receives a copy of every record.

    Returns:
        The root ``diffusionlab`` logger.
    """
    logger = logging.getLogger("diffusionlab")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class JsonlMetricsWriter:
    """Append-only JSON-lines metrics sink with immediate flushing.

    Each :meth:`log` call writes one line, so a crashed run loses at most the
    final in-flight record and the file is always valid line-by-line JSON.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        """Append one metric record (must be JSON-serialisable)."""
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlMetricsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
