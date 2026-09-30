"""Tests for terminal progress helper."""

import logging

from rvsauto.common import quiet_subprocesses_enabled, set_quiet_subprocesses
from rvsauto.progress import PipelineProgress, silence_console_logging


def test_disabled_progress_is_noop():
    p = PipelineProgress(enabled=False)
    p.begin_phase("test", 10)
    p.step(5)
    p.add_tasks(3)
    p.close()


def test_begin_phase_replaces_bar():
    p = PipelineProgress(enabled=False)
    p.begin_phase("A", 10)
    p.step(10)
    p.begin_phase("B", 5)
    p.step(5)
    p.close()


def test_silence_console_logging_keeps_file_handler(tmp_path):
    log_file = tmp_path / "run.log"
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    stream_h = logging.StreamHandler()
    root.addHandler(file_h)
    root.addHandler(stream_h)
    root.setLevel(logging.INFO)

    silence_console_logging()
    remaining = root.handlers
    assert len(remaining) == 1
    assert isinstance(remaining[0], logging.FileHandler)

    logging.info("file-only")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    assert "file-only" in log_file.read_text(encoding="utf-8")


def test_set_quiet_subprocesses_toggle():
    prev = quiet_subprocesses_enabled()
    try:
        set_quiet_subprocesses(True)
        assert quiet_subprocesses_enabled() is True
        set_quiet_subprocesses(False)
        assert quiet_subprocesses_enabled() is False
    finally:
        set_quiet_subprocesses(prev)
