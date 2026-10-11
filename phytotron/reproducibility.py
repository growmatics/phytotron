"""Thread pinning for reproducible results.

MPC results depend on how many threads the math libraries use. Multithreaded sums
round differently, and over the thousands of IPOPT solves in a season that moves the
solver onto a different path: the 2025 MPC season of Growmatics' reference run of 24 July
2026 (``docs/validation.md``, section 1) gives -1.745 EUR/m2 with the
default thread count and exactly -1.607615 (the recorded value) with one thread. The
incumbent is not affected. See ``docs/validation.md``.

The environment variables only take effect if they are set before numpy (or casadi)
is loaded, so call :func:`pin_threads` first thing in a script::

    from phytotron.reproducibility import pin_threads

    pin_threads()

    import numpy as np  # noqa: E402
"""

from __future__ import annotations

import os
import sys
import warnings

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_EARLY_MODULES = ("numpy", "casadi")


def pin_threads(n: int = 1) -> None:
    """Set the math-library thread variables to ``n``; warn if it is already too late."""
    for var in THREAD_VARS:
        os.environ[var] = str(n)
    loaded = [m for m in _EARLY_MODULES if m in sys.modules]
    if loaded:
        warnings.warn(
            f"pin_threads() called after {', '.join(loaded)} was imported; the thread "
            "setting may not take effect. Call it before any other import.",
            RuntimeWarning,
            stacklevel=2,
        )


def thread_settings() -> dict[str, str | None]:
    """The current thread variables, for the run record."""
    return {var: os.environ.get(var) for var in THREAD_VARS}


def threads_pinned() -> bool:
    """True if every thread variable is set to 1."""
    return all(value == "1" for value in thread_settings().values())
