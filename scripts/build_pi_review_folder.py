"""Assemble one folder the PI can review every figure from, in one place.

Release class is read from each run's own manifest, never typed here: the
citywide and zoom families are withheld (red line L1), so the folder is a
*viewing* surface only — it never writes into shared/figures or papers/.
Seeing an artefact and releasing it are different acts (2026-09-17).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
THUMB_W = 1100


def _latest(glob: str) -> Path:
    hits = sorted(d for d in (ROOT / "runs").glob(glob) if d.is_dir())
    if not hits:
        raise FileNotFoundError(f"no run matching runs/{glob}")
    return hits[-1]


def _manifest_classes(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "figure_manifest.json"
    if not path.exists():
        return {}
    figs = json.loads(path.read_text()).get("figures", {})
    return {
        Path(f["png_path"]).name: {
            "release_class": f.get("release_class"),
            "red_line": f.get("red_line"),
            "pixel_m": (f.get("aggregation") or {}).get("pixel_m"),
            "n_cells": (f.get("aggregation") or {}).get("n_cells_aggregated"),
        }
        for f in figs.values() if f.get("status") == "produced" and f.get("png_path")
    }


SECTIONS = [
    ("01_p1_solar_figures", "P1 solar figures (f1-f4)",
     "The four figures staged for the paper. Your promotion ruling is the only thing between these and shared/figures.",
     lambda: _latest("wp07_figures_*")),
    ("02_citywide_maps", "Citywide maps (f5, f5b, f6)",
     "Sky-view and irradiation across the whole 8.4 M-cell domain. Withheld under red line L1 — yours to read, not to circulate.",
     lambda: _latest("wp07_map_*")),
    ("03_zoom_favelas", "Per-favela zoom extracts",
     "Each study favela at the run's 5 m sampling pitch, sharing the citywide colour limits. Ipanema is absent: no bairro boundary exists on disk.",
     lambda: _latest("wp07_zoom_*")),
]

EXTRA = {
    "04_morphotypes": (
        "Morphotypes and morphotopes",
        "The cross-site signature work the weekly deck draws on.",
        [ROOT / "outputs/cross_site/signature/figures_v2" / n for n in (
            "morphotype_schematics.png", "maps_morphotypes.png", "morphotope_maps.png",
            "morphotope_maps_repartition.png", "morphotope_profile.png",
            "morphotope_recurrence.png", "morphotope_stability.png",
            "fingerprint_heatmap.png", "dendrogram.png", "composition_by_site.png",
            "k_selection_rigor.png", "experience_dotplots.png",
        )] + [ROOT / "outputs/cross_site/presentation_figures/fig_morpho_violins.png"],
    ),
    "05_folha_de_rua": (
        "Folha de Rua site sheets",
        "One A3 sheet per site: grid, terrain, density, then sky view and sunlight.",
        sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*_A3.png"))
        + sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*.pdf")),
    ),
    "06_weekly_deck": (
        "Weekly update deck (W39)",
        "Tomorrow's deck and its contact sheet.",
        [Path.home() / "SCL/SCR/brisaverse/slides/brisa_wk39_update.pdf",
         Path.home() / "SCL/SCR/brisaverse/slides/contact_wk39_update.png"],
    ),
}


def _copy(src: Path, dest_dir: Path, meta: dict, entries: list, section: str) -> None:
    if not src.exists():
        entries.append({"section": section, "file": src.name, "status": "MISSING", "source": str(src)})
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)
    row = {"section": section, "file": src.name, "status": "ok",
           "source": str(src.relative_to(ROOT)) if ROOT in src.parents else str(src)}
    row.update({k: v for k, v in meta.items() if v is not None})
    if src.suffix.lower() == ".png":
        with Image.open(src) as im:
            row["pixels"] = list(im.size)
            thumb = im.copy()
            if im.width > THUMB_W:
                thumb.thumbnail((THUMB_W, THUMB_W * 4))
            tdir = dest_dir / "_thumbs"
            tdir.mkdir(exist_ok=True)
            thumb.convert("RGB").save(tdir / (src.stem + ".jpg"), quality=86)
        row["thumb"] = f"_thumbs/{src.stem}.jpg"
    row["bytes"] = src.stat().st_size
    entries.append(row)


def build(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    sections: list[dict] = []

    for slug, title, blurb, resolve in SECTIONS:
        run_dir = resolve()
        classes = _manifest_classes(run_dir)
        for name in sorted(classes):
            _copy(run_dir / name, out_root / slug, classes[name], entries, slug)
        sections.append({"slug": slug, "title": title, "blurb": blurb,
                         "provenance": str(run_dir.relative_to(ROOT))})

    for slug, (title, blurb, paths) in EXTRA.items():
        for src in paths:
            _copy(src, out_root / slug, {}, entries, slug)
        sections.append({"slug": slug, "title": title, "blurb": blurb,
                         "provenance": "existing outputs/ and slides/ products"})

    manifest = {
        "_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_pi_review_folder.py",
        "sections": sections,
        "files": entries,
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out_root / "index.html").write_text(_render(manifest))
    return manifest


def _render(m: dict) -> str:
    by_section: dict[str, list] = {}
    for e in m["files"]:
        by_section.setdefault(e["section"], []).append(e)
    parts = ["""<!doctype html><meta charset="utf-8"><title>Figure review</title>
<style>
:root{--bg:#faf8f5;--ink:#1c1a17;--dim:#6b6560;--line:#ddd6cd;--warn:#b6482b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:0 20px 80px}
header{padding:32px 0 8px;border-bottom:1px solid var(--line);margin-bottom:24px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:40px 0 2px}
.blurb{color:var(--dim);margin:0 0 4px;max-width:62ch}
.prov{color:var(--dim);font-size:12.5px;font-family:ui-monospace,monospace;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}
figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff}
figure img{width:100%;display:block;background:#fff}
figcaption{padding:9px 11px;font-size:13px}
.name{font-family:ui-monospace,monospace;word-break:break-all}
.meta{color:var(--dim);font-size:12px;margin-top:3px}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:99px;
border:1px solid var(--line);margin-right:5px}
.withheld{color:var(--warn);border-color:var(--warn)}
a{color:inherit}.nolink{padding:9px 11px;font-size:13px}
</style>
<header><h1>Figure review</h1>
<p class="blurb">Every figure this cycle produced, in one place. Cards marked
<span class="tag withheld">withheld · L1</span> are yours to read; they do not travel into
the paper or shared figures without your own tap.</p></header>"""]
    for s in m["sections"]:
        rows = by_section.get(s["slug"], [])
        if not rows:
            continue
        parts.append(f'<h2>{s["title"]}</h2><p class="blurb">{s["blurb"]}</p>'
                     f'<p class="prov">{s["provenance"]}</p><div class="grid">')
        for e in rows:
            if e["status"] == "MISSING":
                parts.append(f'<figure><div class="nolink">missing: <span class="name">{e["file"]}</span></div></figure>')
                continue
            tags = ""
            if e.get("release_class") == "withheld":
                tags += f'<span class="tag withheld">withheld · {e.get("red_line","L1")}</span>'
            elif e.get("release_class"):
                tags += f'<span class="tag">{e["release_class"]}</span>'
            bits = []
            if e.get("pixels"):
                bits.append(f'{e["pixels"][0]}×{e["pixels"][1]} px')
            if e.get("pixel_m"):
                bits.append(f'{e["pixel_m"]:g} m pitch')
            if e.get("n_cells"):
                bits.append(f'{e["n_cells"]:,} cells')
            bits.append(f'{e["bytes"]/1e6:.1f} MB')
            img = (f'<a href="{s["slug"]}/{e["file"]}"><img src="{s["slug"]}/{e["thumb"]}" loading="lazy" alt=""></a>'
                   if e.get("thumb") else "")
            link = f'<a href="{s["slug"]}/{e["file"]}">{e["file"]}</a>'
            parts.append(f'<figure>{img}<figcaption>{tags}<div class="name">{link}</div>'
                         f'<div class="meta">{" · ".join(bits)}</div></figcaption></figure>')
        parts.append("</div>")
    return "\n".join(parts)


# The PI browses this from a file manager, where a symlink shows up as a single
# file rather than a folder. The repo copy is what the hub mirrors; this one is
# a hard-linked twin, so it is a real directory that costs no extra disk.
WEEKLY_RM = Path.home() / "SCL" / "SCR" / "weekly RM"


def mirror_to_weekly_rm(built: Path, date_slug: str) -> Path:
    dest = WEEKLY_RM / date_slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built, dest, copy_function=os.link)
    return dest


if __name__ == "__main__":
    import sys
    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "_review" / date_slug
    man = build(out)
    twin = mirror_to_weekly_rm(out, out.name)
    print(f"also at {twin}")
    ok = sum(1 for e in man["files"] if e["status"] == "ok")
    missing = [e["file"] for e in man["files"] if e["status"] == "MISSING"]
    print(f"{out}: {ok} files in {len(man['sections'])} sections")
    if missing:
        print("MISSING:", ", ".join(missing))
