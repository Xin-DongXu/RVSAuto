"""Tests for ``rvsauto`` top-level CLI dispatch."""

import pytest

from rvsauto.__main__ import main


def test_main_no_args_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "screen" in out and "redock" in out


def test_main_version():
    assert main(["--version"]) == 0


def test_main_screen_delegates_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["screen", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--pdb_dir" in out


def test_main_redock_delegates_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["redock", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--input_dir" in out


def test_main_unknown_command():
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code == 2
