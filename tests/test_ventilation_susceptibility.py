"""2-D ventilation-susceptibility bivariate palette (structural guards)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ventilation_susceptibility import (  # noqa: E402
    CLASS_COLOR,
    REGIMES,
)


def _luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_palette_covers_all_six_classes():
    assert set(CLASS_COLOR) == {(r, d) for r in REGIMES for d in ("shallow", "deep")}


def test_deep_is_darker_than_shallow_per_regime():
    for r in REGIMES:
        assert _luminance(CLASS_COLOR[(r, "deep")]) < _luminance(CLASS_COLOR[(r, "shallow")])


def test_worst_class_is_the_darkest():
    # skimming|deep (the doubly-constrained worst case) must read darkest
    worst = _luminance(CLASS_COLOR[("skimming", "deep")])
    assert worst == min(_luminance(c) for c in CLASS_COLOR.values())
