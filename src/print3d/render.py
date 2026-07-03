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
FRAME = "#efe9df"     # base frame / mat
WATER = "#5b86a6"     # sea / lagoon
TEAL = "#0f766e"


def _downsample(arr: np.ndarray, target: int = 160):
    step = max(1, max(arr.shape) // target)
    return arr[::step, ::step], step


def render_site(dsm, title: str, subtitle: str, out_png: Path, exaggerate: float = 1.0) -> Path:
    """Hillshade plan + two honest-aspect axonometric angles of the composed site
    artifact — framed base, engraved nameplate, recessed water — so the reviewer
    can read the fabric before downloading the STL."""
    comp = dsm.composed
    surf = comp.surf
    fig = plt.figure(figsize=(15, 5.2))
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.93, subtitle, fontsize=9, color="#555", ha="left")

    # plan: hillshade of the composed surface; water + massing tinted, nameplate shows
    ax = fig.add_subplot(1, 3, 1)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(surf, cmap=plt.cm.gray, vert_exag=2.0, blend_mode="soft")
    ax.imshow(rgb, origin="upper", aspect="equal")
    built = np.ma.masked_where(~comp.built, np.ones_like(surf))
    ax.imshow(built, origin="upper", cmap=matplotlib.colors.ListedColormap([BUILT]),
              alpha=0.5, aspect="equal")
    wat = np.ma.masked_where(~comp.water, np.ones_like(surf))
    ax.imshow(wat, origin="upper", cmap=matplotlib.colors.ListedColormap([WATER]),
              alpha=0.6, aspect="equal")
    ax.set_title("plan · framed base, engraved nameplate, water", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # two axonometric angles — TRUE aspect ratio (label if exaggerated)
    (X, _), (Y, _), (Z, step) = (_downsample(comp.X), _downsample(comp.Y), _downsample(surf))
    built_d, water_d, frame_d = comp.built[::step, ::step], comp.water[::step, ::step], comp.frame[::step, ::step]
    fc = np.empty(Z.shape + (4,))
    fc[:] = matplotlib.colors.to_rgba(TERRAIN)
    fc[frame_d] = matplotlib.colors.to_rgba(FRAME)
    fc[built_d] = matplotlib.colors.to_rgba(BUILT)
    fc[water_d] = matplotlib.colors.to_rgba(WATER)
    aspect_note = "true aspect" if exaggerate == 1 else f"relief ×{exaggerate:g}"
    for k, azim in enumerate((-58, 128)):
        ax2 = fig.add_subplot(1, 3, 2 + k, projection="3d")
        ax2.plot_surface(X, Y, Z * exaggerate, facecolors=fc, rstride=1, cstride=1,
                         linewidth=0, antialiased=False, shade=True)
        ax2.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z) * exaggerate, 1e-6)))
        ax2.view_init(elev=32, azim=azim)
        ax2.set_axis_off()
        ax2.set_title(f"axonometric · {aspect_note}" + (" · reverse" if k else ""),
                      fontsize=9, y=0.96)

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
