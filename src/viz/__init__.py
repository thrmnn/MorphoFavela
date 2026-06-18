"""Cross-script visualization helpers (presentation/paper preset switching).

Distinct from `src/visualization/`, which holds figure-generation routines
for specific artefacts. This subpackage holds *style-layer* helpers that
any figure script can opt into via a `--preset {paper,presentation}` flag.
"""

from src.viz.presentation_style import (
    add_scale_bar,
    apply,
    apply_to_map_axes,
    short_label,
)

__all__ = [
    "add_scale_bar",
    "apply",
    "apply_to_map_axes",
    "short_label",
]
