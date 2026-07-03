"""Arrange the 5 full-site models on one Ender-3 bed and slice a single G-code.

    python scripts/build_print_plate.py

Shelf-packs the five site STLs onto a 220x220 mm bed (centred, 8 mm gaps),
exports one combined plate STL, renders a top-down bed-layout PNG, and — if a
CuraEngine binary is available — slices the plate to G-code (PLA, 0.2 mm) and
reports the print-time / filament estimate. Output → outputs/_hub/print_plate/.

CuraEngine + Cura 4.4 definitions are a local, non-repo install; point at them
with $CURAENGINE and $CURA_DEFS (defaults under ~/.local/opt/cura).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.print3d.model import ROOT, SITE_LABELS

BED = 220.0                # Ender-3 bed, mm
GAP = 8.0                  # gap between models, mm
MARGIN = 8.0               # keep-clear from bed edge, mm
CAMPAIGN = ["rocinha", "maré", "riodaspedras", "vidigal", "complexo_do_alemao"]

CURAENGINE = Path(os.environ.get("CURAENGINE", Path.home() / ".local/opt/cura/root/usr/bin/CuraEngine"))
CURA_LIB = CURAENGINE.parent.parent / "lib" / "x86_64-linux-gnu"
CURA_DEFS = Path(os.environ.get("CURA_DEFS", Path.home() / ".local/opt/cura/defs"))

# PLA / 0.2 mm settings, tuned for a legible terrain model (no supports needed —
# the framed base + draped terrain has no steep overhangs).
CURA_SETTINGS = {
    "layer_height": 0.2, "layer_height_0": 0.24, "line_width": 0.4,
    "wall_line_count": 2, "top_layers": 4, "bottom_layers": 4,
    "infill_sparse_density": 12, "infill_pattern": "grid",
    "material_print_temperature": 200, "material_print_temperature_layer_0": 205,
    "material_bed_temperature": 60, "material_bed_temperature_layer_0": 60,
    "speed_print": 50, "speed_travel": 120, "speed_layer_0": 20, "speed_wall_0": 30,
    "retraction_enable": "true", "retraction_amount": 5, "retraction_speed": 45,
    "adhesion_type": "skirt", "skirt_line_count": 2, "skirt_gap": 3,
    "support_enable": "false", "cool_fan_enabled": "true",
    "machine_width": 220, "machine_depth": 220, "machine_height": 250,
}


def _load_sites():
    items = []
    for site in CAMPAIGN:
        stls = sorted((ROOT / "outputs" / site / "print").glob(f"{site}_site_1to*.stl"))
        if stls:
            m = trimesh.load(stls[-1], process=False)
            items.append((site, m))
    return items


def shelf_pack(items):
    """Shelf/row packer: place footprints left→right, wrap to a new shelf when the
    row would exceed the usable bed. Returns [(site, mesh, cx, cy)] with the whole
    block centred on the bed. Tallest-first keeps shelves tight."""
    usable = BED - 2 * MARGIN
    order = sorted(items, key=lambda it: it[1].extents[1], reverse=True)
    placed, x, y, shelf_h, rows = [], 0.0, 0.0, 0.0, []
    row = []
    for site, m in order:
        w, d = m.extents[0], m.extents[1]
        if row and x + w > usable:
            rows.append((row, y, shelf_h)); y += shelf_h + GAP; x = 0.0; shelf_h = 0.0; row = []
        row.append((site, m, x + w / 2, w, d)); x += w + GAP; shelf_h = max(shelf_h, d)
    if row:
        rows.append((row, y, shelf_h))
    total_h = y + shelf_h
    total_w = max((sum(r[3] for r in rr[0]) + GAP * (len(rr[0]) - 1)) for rr in rows)
    ox = (BED - total_w) / 2
    oy = (BED - total_h) / 2
    for row, ry, sh in rows:
        rx = ox + (total_w - (sum(r[3] for r in row) + GAP * (len(row) - 1))) / 2
        cur = rx
        for site, m, _, w, d in row:
            placed.append((site, m, cur + w / 2, oy + ry + sh / 2))
            cur += w + GAP
    return placed


def build_plate(placed):
    parts = []
    layout = []
    for site, m, cx, cy in placed:
        mm = m.copy()
        b = mm.bounds
        mm.apply_translation([cx - (b[0][0] + b[1][0]) / 2,
                              cy - (b[0][1] + b[1][1]) / 2,
                              -b[0][2]])  # centre in slot, sit on bed (z=0)
        parts.append(mm)
        e = mm.extents
        layout.append((site, cx, cy, e[0], e[1], e[2]))
    plate = trimesh.util.concatenate(parts)
    # CuraEngine adds a bed-centre offset (machine_center_is_zero=false), so feed
    # it an origin-centred plate; the layout PNG keeps the human bed coordinates.
    b = plate.bounds
    plate.apply_translation([-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, 0])
    return plate, layout


def render_layout(layout, out_png, time_s=None, filament_g=None):
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    ax.add_patch(Rectangle((0, 0), BED, BED, fill=False, ec="#333", lw=2))
    ax.add_patch(Rectangle((MARGIN, MARGIN), BED - 2 * MARGIN, BED - 2 * MARGIN,
                           fill=False, ec="#bbb", ls="--", lw=0.8))
    for site, cx, cy, w, d, h in layout:
        ax.add_patch(Rectangle((cx - w / 2, cy - d / 2), w, d, facecolor="#cbb79a",
                               edgecolor="#7a5a3a", lw=1.2, alpha=0.9))
        ax.text(cx, cy, f"{SITE_LABELS.get(site, site)}\n{w:.0f}×{d:.0f}×{h:.0f} mm",
                ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(-5, BED + 5); ax.set_ylim(-5, BED + 5); ax.set_aspect("equal")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    title = "Ender-3 plate · 5 favela sites · 220×220 mm"
    if time_s:
        title += f"\nsliced: {time_s/3600:.1f} h · {filament_g:.0f} g PLA (0.2 mm)"
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def slice_plate(plate_stl, out_gcode):
    if not CURAENGINE.exists():
        print(f"  CuraEngine not found at {CURAENGINE} — skipping slice (plate STL still written).")
        return None
    ender3 = CURA_DEFS / "creality_ender3.def.json"
    cmd = [str(CURAENGINE), "slice", "-v", "-j", str(ender3), "-o", str(out_gcode)]
    for k, v in CURA_SETTINGS.items():
        cmd += ["-s", f"{k}={v}"]
    cmd += ["-l", str(plate_stl)]
    env = dict(os.environ, LD_LIBRARY_PATH=str(CURA_LIB),
               CURA_ENGINE_SEARCH_PATH=str(CURA_DEFS))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    log = r.stdout + r.stderr
    if not out_gcode.exists():
        print("  SLICE FAILED:\n" + "\n".join(log.splitlines()[-15:]))
        return None
    # standalone CuraEngine's header ;TIME/;Filament are unreliable — read the
    # real total from the last ;TIME_ELAPSED and filament from the max E move.
    import re
    time_s, e_max = 0.0, 0.0
    for ln in out_gcode.read_text(errors="ignore").splitlines():
        if ln.startswith(";TIME_ELAPSED:"):
            time_s = float(ln.split(":")[1])
        elif ln.startswith(("G0", "G1")) and (m := re.search(r"E(-?[\d.]+)", ln)):
            e_max = max(e_max, float(m.group(1)))
    fil_g = e_max * np.pi * (1.75 / 2) ** 2 * 1.24 / 1000  # PLA 1.24 g/cc, 1.75 mm
    return time_s, fil_g


def main():
    items = _load_sites()
    if len(items) < 5:
        print(f"  only {len(items)} site STLs found — run build_site_model.py first")
    placed = shelf_pack(items)
    plate, layout = build_plate(placed)
    out = ROOT / "outputs" / "_hub" / "print_plate"
    out.mkdir(parents=True, exist_ok=True)
    plate_stl = out / "favelas_plate_ender3.stl"
    plate.export(plate_stl)
    ext = plate.extents
    print(f"  plate: {len(layout)} sites · footprint {ext[0]:.0f}×{ext[1]:.0f} mm · "
          f"tallest {ext[2]:.0f} mm · {len(plate.faces):,} tris")
    print(f"  -> {plate_stl.relative_to(ROOT)}")

    gcode = out / "favelas_plate_ender3.gcode"
    res = slice_plate(plate_stl, gcode)
    time_s = fil_g = None
    if res:
        time_s, fil_g = res
        mb = gcode.stat().st_size / 1e6
        print(f"  sliced -> {gcode.relative_to(ROOT)} ({mb:.1f} MB)")
        print(f"  ESTIMATE: {time_s/3600:.1f} h  ·  {fil_g:.0f} g PLA")
    png = render_layout(layout, out / "favelas_plate_layout.png", time_s,
                        fil_g if time_s else None)
    print(f"  -> {png.relative_to(ROOT)}")

    import json
    (out / "favelas_plate.json").write_text(json.dumps({
        "printer": "Creality Ender-3", "bed_mm": BED, "nozzle_mm": 0.4,
        "layer_mm": CURA_SETTINGS["layer_height"], "material": "PLA",
        "footprint_mm": [round(ext[0], 1), round(ext[1], 1)], "tallest_mm": round(ext[2], 1),
        "n_sites": len(layout), "triangles": int(len(plate.faces)),
        "print_time_h": round(time_s / 3600, 1) if time_s else None,
        "filament_g": round(fil_g) if time_s else None,
        "sites": [{"site": s, "w": round(w, 1), "d": round(d, 1), "h": round(h, 1)}
                  for s, _, _, w, d, h in layout],
    }, indent=2))


if __name__ == "__main__":
    main()
