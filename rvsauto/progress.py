"""Terminal progress bar for long batch screen / redock runs."""

from __future__ import annotations

import logging
import sys
from typing import Optional


class _TqdmStreamHandler(logging.Handler):
    """Route log records through ``tqdm.write`` so they do not break the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def use_tqdm_safe_console_logging() -> None:
    """Replace stdout/stderr ``StreamHandler``s with tqdm-safe handlers."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(_TqdmStreamHandler())


class PipelineProgress:
    """Single global progress bar with percent complete and ETA (via tqdm)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._total = 0
        self._bar = None

    def begin(self, total: int) -> None:
        self._total = max(0, int(total))
        if not self.enabled or self._total == 0:
            return
        from tqdm import tqdm

        self._bar = tqdm(
            total=self._total,
            unit="task",
            dynamic_ncols=True,
            file=sys.stderr,
            desc="RVSAuto",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    def add_tasks(self, extra: int) -> None:
        """Increase total work units (e.g. when docking job count is known)."""
        if extra <= 0:
            return
        self._total += extra
        if self._bar is not None:
            self._bar.total = self._total
            self._bar.refresh()

    def step(self, n: int = 1, phase: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if self._bar is None:
            return
        if phase:
            self._bar.set_description(f"RVSAuto | {phase}")
        if n:
            self._bar.update(n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
