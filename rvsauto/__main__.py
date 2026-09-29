"""``python -m rvsauto`` and the ``rvsauto`` console command."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvsauto",
        description=(
            "RVSAuto: batch UniDock virtual screening and self-redocking pipelines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rvsauto screen --pdb_dir receptors --ligand_dir ligands --output_dir out\n"
            "  rvsauto redock --input_dir complexes --output_dir redock_out\n"
            "\n"
            "Aliases: rvsauto-unidock (screen), rvsauto-redock (redock)."
        ),
    )
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"rvsauto {__version__}",
    )
    sub = p.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
        required=False,
    )
    sub.add_parser(
        "screen",
        help="Virtual screening with P2Rank or AF2BIND pockets (UniDock).",
        add_help=False,
    )
    sub.add_parser(
        "redock",
        help="Self-redocking and heavy-atom RMSD (UniDock).",
        add_help=False,
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list: List[str] = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        build_parser().print_help()
        return 0

    if args_list[0] in ("-h", "--help") and len(args_list) == 1:
        build_parser().print_help()
        return 0

    command = args_list[0]
    rest = args_list[1:]

    if command == "screen":
        from .cli_unidock import main as screen_main

        return screen_main(rest)

    if command == "redock":
        from .cli_redock import main as redock_main

        return redock_main(rest)

    # --version at top level (no subcommand)
    if command in ("-V", "--version") and len(args_list) == 1:
        print(f"rvsauto {__version__}")
        return 0

    parser = build_parser()
    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
