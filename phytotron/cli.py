"""The ``phytotron`` command: ``run``, ``quickstart`` and ``fetch-weather``.

    phytotron quickstart --controller incumbent
    phytotron run --run-dir runs/full --workers 18
    phytotron fetch-weather --lat 52.37 --lon 4.90 --years 2014-2025 --location Amsterdam_OM

Every subcommand takes ``--help``. The math libraries are pinned to one thread before
anything else is imported, which MPC results need to be reproducible.
"""

from __future__ import annotations

import argparse
import sys

from phytotron import __version__
from phytotron.reproducibility import pin_threads

COMMANDS = ("run", "quickstart", "fetch-weather")


def build_parser() -> argparse.ArgumentParser:
    from phytotron.commands import fetch_weather, quickstart, run_benchmark

    parser = argparse.ArgumentParser(prog="phytotron", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"phytotron {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run_benchmark.build_parser(
        sub.add_parser("run", help="run the benchmark grid", description=run_benchmark.__doc__)
    )
    quickstart.build_parser(
        sub.add_parser(
            "quickstart", help="one controller, one season", description=quickstart.__doc__
        )
    )
    fetch_weather.build_parser(
        sub.add_parser(
            "fetch-weather", help="download weather years", description=fetch_weather.__doc__
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    pin_threads()  # before numpy loads
    args = build_parser().parse_args(argv)
    from phytotron.commands import fetch_weather, quickstart, run_benchmark

    if args.command == "run":
        return run_benchmark.run(args)
    if args.command == "quickstart":
        quickstart.run(args)
        return 0
    fetch_weather.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
