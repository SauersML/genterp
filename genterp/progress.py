"""Structured progress logging for long-running Genterp workflows."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class ProgressLogger:
    """Print elapsed-time progress lines with completed/total unit counts."""

    name: str
    total_units: int | None = None
    completed_units: int = 0
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    started_at: float = field(default_factory=time.monotonic)

    def log(self, action: str, detail: str | None = None) -> None:
        elapsed = time.monotonic() - self.started_at
        progress = self.progress_text()
        suffix = f" | {detail}" if detail else ""
        print(f"[{self.name} t+{elapsed:8.1f}s {progress}] {action}{suffix}", file=self.stream, flush=True)

    def start_unit(self, action: str, detail: str | None = None) -> None:
        self.log(f"START {action}", detail)

    def finish_unit(self, action: str, detail: str | None = None, units: int = 1) -> None:
        self.completed_units += units
        self.log(f"DONE  {action}", detail)

    def set_progress(self, completed_units: int, total_units: int | None = None) -> None:
        self.completed_units = completed_units
        if total_units is not None:
            self.total_units = total_units

    def progress_text(self) -> str:
        total = "?" if self.total_units is None else f"{self.total_units:,}"
        return f"units={self.completed_units:,}/{total}"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
