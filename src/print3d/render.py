"""Headless preview renders for the 3D-print models.

Two products: a hillshaded plan + low-angle axonometric of a draped surface (for
the full-site DSM models), and a shaded axonometric of a patch mesh. Both write a
single PNG so the dashboard can show what will come off the printer without a
slicer screenshot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

TERRAIN = "#d8cdbf"   # bare ground
BUILT = "#9c6f52"     # massing
TEAL = "#0f766e"


def _downsample(arr: np.ndarray, target: int = 140) -> np.ndarray:
    step = max(1, max(arr.shape) // target)
    return arr[::step, ::step]


def render_site(dsm, title: str, subtitle: str, out_png: Path) -> Path:
    """Hillshade plan (left) + axonometric massing (right) of a site DSM."""
    surf, ground, height = dsm.surface, dsm.ground, dsm.height
    fig = plt.figure(figsize=(11, 5.2))
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.92, subtitle, fontsize=9, color="#555", ha="left")

    # plan: hillshade of the surface, buildings tinted over bare ground
    ax = fig.add_subplot(1, 2, 1)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(surf, cmap=plt.cm.gray, vert_exag=2.0, blend_mode="soft")
    ax.imshow(rgb, origin="upper")
    built = np.ma.masked_where(height <= 0, height)
    ax.imshow(built, origin="upper", cmap="copper", alpha=0.55)
    ax.set_title("hillshade plan · built massing tinted", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # axonometric surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    X, Y, Z = _downsample(dsm.X), _downsample(dsm.Y), _downsample(surf)
    Hd = _downsample(height)
    built_rgb = matplotlib.colors.to_rgba(BUILT)
    ground_rgb = matplotlib.colors.to_rgba(TERRAIN)
    facecolors = np.where(
        (Hd > 0.5)[..., None], np.array(built_rgb), np.array(ground_rgb)
    )
    ax2.plot_surface(
        X, Y, Z, facecolors=facecolors,
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True,
    )
    ax2.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z) * 3, 1)))
    ax2.view_init(elev=38, azim=-60)
    ax2.set_axis_off()
    ax2.set_title("draped surface (relief ×3)", fontsize=9, y=0.96)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def render_patch_mesh(mesh, title: str, subtitle: str, out_png: Path) -> Path:
    """Axonometric of a patch print mesh, shaded by height."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v, f = mesh.vertices, mesh.faces
    tri = v[f]
    z = tri[:, :, 2].mean(axis=1)
    norm = (z - z.min()) / (np.ptp(z) + 1e-9)
    colors = plt.cm.copper(0.25 + 0.7 * norm)

    fig = plt.figure(figsize=(6.4, 6.0))
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.92, subtitle, fontsize=9, color="#555", ha="left")
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tri, facecolors=colors, edgecolors="none")
    ax.add_collection3d(coll)
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_box_aspect((np.ptp(v[:, 0]), np.ptp(v[:, 1]), max(np.ptp(v[:, 2]), 1)))
    ax.view_init(elev=32, azim=-55)
    ax.set_axis_off()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png
