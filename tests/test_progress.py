"""Tests for terminal progress helper."""

from rvsauto.progress import PipelineProgress


def test_disabled_progress_is_noop():
    p = PipelineProgress(enabled=False)
    p.begin(10)
    p.step(5, phase="test")
    p.add_tasks(3)
    p.close()
