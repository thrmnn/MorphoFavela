"""Output-asset directory resolution for the brisa deck figure scripts.

The figures are consumed by the external brisa paper/slides repo, which on the
author's machine lives at ``/home/theo/brisa_paper/artifacts/slides/assets``.
That absolute path was previously hardcoded in every figure script, so a fresh
clone (or any other host) could not run them.

Resolution order:
1. ``$BRISA_ASSETS_DIR`` — point this at the external brisa repo's asset dir.
2. Repo-relative fallback ``<repo>/outputs/brisa_deck_assets`` so scripts run
   out-of-the-box without the external repo present.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

BRISA_ASSETS_DIR = Path(
    os.environ.get("BRISA_ASSETS_DIR", _REPO_ROOT / "outputs" / "brisa_deck_assets")
)
