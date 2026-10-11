"""Same as ``phytotron fetch-weather``; kept so the documented commands keep working.

Run ``phytotron fetch-weather --help`` or ``python examples/fetch_weather.py --help``.
"""

import sys

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads

from phytotron.commands.fetch_weather import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
