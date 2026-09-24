"""Small PNG contact sheet for the PI's review: OM2 with a few variables
along the route (map colored by SVF + line plots vs distance).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


def build_contact_sheet(points_df, out_path: Path, route_id: str = "OM2", version: str = "") -> Path:
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.4)

        ax_map = fig.add_subplot(gs[0])
        sc = ax_map.scatter(points_df["x"], points_df["y"], c=points_df["sky_view_factor"], cmap="viridis", s=4)
        # adjustable="datalim" (not the default "box") lets the axes box stay
        # the gridspec's full width while only the data limits shrink to
        # honour equal aspect — the default squeezed the map into the
        # centre of a 10-inch-wide axis and left a blank panel either side
        # (panel must-fix 8, 2026-09-24).
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.set_title(f"{route_id} route, coloured by sky_view_factor")
        ax_map.set_xlabel("Easting (m, EPSG:31983)")
        ax_map.set_ylabel("Northing (m)")
        fig.colorbar(sc, ax=ax_map, label="sky_view_factor", shrink=0.8)

        panels = [
            ("height_width_ratio", "H/W ratio"),
            ("plan_density_lambda_p", "plan density (lambda_p)"),
            ("ventilation_frontal_area_proxy", "ventilation frontal-area PROXY"),
        ]
        for i, (col, label) in enumerate(panels, start=1):
            ax = fig.add_subplot(gs[i])
            if col in points_df.columns:
                ax.plot(points_df["distance_along_m"], points_df[col], lw=0.8)
            ax.set_ylabel(label, fontsize=8)
            if i == len(panels):
                ax.set_xlabel("distance along route (m)")

        version_suffix = f" {version}" if version else ""
        fig.suptitle(f"{route_id} morphology contact sheet —{version_suffix} (airborne, 2019 source)", fontsize=11)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path
