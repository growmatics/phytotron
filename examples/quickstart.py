"""Same as ``phytotron quickstart``; kept so the documented commands keep working.

Run ``phytotron quickstart --help`` or ``python examples/quickstart.py --help``.
"""

import sys

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads

from phytotron.commands.quickstart import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
