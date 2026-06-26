"""Figure 7 — Vidigal street-level solar envelope (winter / annual / summer).

Canonical site-7 figure for the technical report's §5.4. The same
plotting machinery serves the four other sites via
``figS_solar_envelope.py`` — this wrapper just pins the Vidigal-
specific filename stem, the §5.3 cross-reference in the suptitle,
and the legacy ``fig07a/b/c_solar_*.png`` standalone-panel names.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figS_solar_envelope import render_envelope

SUPTITLE = (
    "Vidigal — street-level solar envelope.  Same points, same scale, three reference cases.\n"
    "(a) §5.3 winter dissociation panel; (b) the figure year-round comfort & PV studies "
    "should use; (c) S-facing alleys recover in summer."
)


def main() -> None:
    render_envelope(
        site="vidigal",
        stem="fig07_solar_envelope_vidigal",
        suptitle=SUPTITLE,
        save_standalones=True,
        standalone_stems={
            "winter": "fig07a_solar_winter",
            "annual": "fig07b_solar_annual",
            "summer": "fig07c_solar_summer",
        },
    )


if __name__ == "__main__":
    main()
