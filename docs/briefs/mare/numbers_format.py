"""Formatting rule for mare_numbers.json values: 3 significant figures for
floats/percentages, thousands separators for integer counts, text passed
through. Shared by build_brief.py and tests/test_mare_brief.py.
"""
from __future__ import annotations


def _sig3(value: float) -> str:
    if value == 0:
        return "0"
    s = f"{value:.3g}"
    if "e" in s or "E" in s:
        # magnitudes in this brief never need scientific notation;
        # fall back to explicit 3-sig-fig decimal rounding.
        import math

        digits = 3 - int(math.floor(math.log10(abs(value)))) - 1
        s = f"{round(value, digits)}"
    return s


def format_entry(entry: dict) -> str:
    kind = entry["kind"]
    value = entry["value"]
    if kind == "text":
        return str(value)
    if kind == "int":
        return f"{int(round(value)):,}"
    if kind in ("float", "percent"):
        return _sig3(float(value))
    raise ValueError(f"unknown kind: {kind!r}")
