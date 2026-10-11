"""Same as ``phytotron run``; kept so the documented commands keep working.

Run ``phytotron run --help`` or ``python examples/run_benchmark.py --help``.
"""

import sys

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads

from phytotron.commands.run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
