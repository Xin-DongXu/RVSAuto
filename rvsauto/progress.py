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


def silence_console_logging() -> None:
    """Keep file logging; remove console handlers so only the progress bar shows."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            root.removeHandler(handler)
            handler.close()


class PipelineProgress:
    """One progress bar per pipeline phase (each phase is 0–100% on its own)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._total = 0
        self._phase = "RVSAuto"
        self._bar = None

    def begin_phase(self, phase: str, total: int) -> None:
        """Close the previous phase bar (if any) and start a new one."""
        self.close()
        self._phase = phase or "RVSAuto"
        self._total = max(0, int(total))
        if not self.enabled or self._total == 0:
            return
        from tqdm import tqdm

        self._bar = tqdm(
            total=self._total,
            unit="task",
            dynamic_ncols=True,
            file=sys.stderr,
            desc=f"RVSAuto | {self._phase}",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

    def begin(self, total: int) -> None:
        """Backward-compatible alias for a single unlabeled phase."""
        self.begin_phase("RVSAuto", total)

    def add_tasks(self, extra: int) -> None:
        """Increase the *current* phase total (rarely needed with per-phase bars)."""
        if extra <= 0:
            return
        self._total += extra
        if self._bar is not None:
            self._bar.total = self._total
            self._bar.refresh()
        elif self.enabled and self._total > 0:
            self.begin_phase(self._phase, self._total)

    def step(self, n: int = 1, phase: Optional[str] = None) -> None:
        if not self.enabled:
            return
        if phase and phase != self._phase:
            # Description-only update if caller still passes a phase label.
            self._phase = phase
            if self._bar is not None:
                self._bar.set_description(f"RVSAuto | {self._phase}")
        if self._bar is None:
            return
        if n:
            self._bar.update(n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
