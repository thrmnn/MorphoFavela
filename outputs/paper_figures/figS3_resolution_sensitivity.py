#!/usr/bin/env python3
"""Figure S3: Resolution sensitivity (placeholder).

Shows information loss from 10m → 20m grid aggregation.

# TODO: requires 20m morphometric grids to be computed.
# Once available, this script will:
# 1. Load 10m and 20m grids for Vidigal
# 2. Compare indicator distributions (violin plots)
# 3. Compute spatial correlation between resolutions
# 4. Show maps of the difference
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import *


def main():
    apply_style()

    # Check if 20m grids exist
    grid_20m_path = PROJECT_ROOT / "outputs" / "vidigal" / "morphometrics" / "grid" / "grid_metrics_20m.gpkg"

    if not grid_20m_path.exists():
        print("  SKIPPED: 20m morphometric grids not yet computed.")
        print("  To generate, run:")
        print("    python scripts/run_morphometric_audit.py --area vidigal --cell-size 20")
        print("  Then re-run this script.")
        return

    # TODO: implement comparison when data is available
    print("  20m grid found — implement comparison here.")


if __name__ == "__main__":
    main()
