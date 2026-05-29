"""MorphoFavela entry-point scripts.

This package collects the executable entry points for the MorphoFavela pipeline.
Each module exposes a `main()` function suitable for invocation either
as `python -m scripts.<name>`, via the console-script aliases registered
in `pyproject.toml` (`mf-svf`, `mf-morphometry`, etc.), or by direct
script invocation `python scripts/<name>.py`.

The actual analysis logic lives in the `src/` library — these wrappers
are thin argparse-driven adapters.
"""
