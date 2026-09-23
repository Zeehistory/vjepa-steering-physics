"""Logging utilities: stdlib logger setup + a lightweight JSONL metrics writer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("vjepa")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class JSONLLogger:
    """Append-only JSONL metrics log (one record per line)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def log(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
