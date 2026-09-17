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

Image.MAX_IMAGE_PIXELS = None  # the citywide pair is 21298x6211 by design

ROOT = Path(__file__).resolve().parents[1]
THUMB_W = 1100


def _image_dir(run_dir: Path) -> Path:
    """Where a run keeps its PNGs. Most write them beside the manifest; some use
    a figures/ subdirectory. Resolve it rather than assuming either."""
    return run_dir / "figures" if (run_dir / "figures").is_dir() else run_dir


def _latest(glob: str) -> Path | None:
    """Newest run holding images. Returns None rather than raising: a family
    still being produced should leave its section out, not break the folder."""
    hits = sorted(d for d in (ROOT / "runs").glob(glob)
                  if d.is_dir() and (d / "figure_manifest.json").is_file()
                  and list(_image_dir(d).glob("*.png")))
    return hits[-1] if hits else None


def _wp_by_figure(run_dir: Path) -> dict[str, str]:
    """Which work package each figure's numbers come from — derived, never typed:
    a figure cites ledger ids, each ledger entry names its source run, and the
    ledger's own runs_of_record maps a run back to its WP. The PI asked "what
    about WP4 and WP6 figures?" precisely because f2 and f4 never said so."""
    manifest = run_dir / "figure_manifest.json"
    if not manifest.exists():
        return {}
    man = json.loads(manifest.read_text())
    ledger_rel = man.get("ledger_source")
    if not ledger_rel:
        return {}
    ledger_path = ROOT / ledger_rel
    if not ledger_path.exists():
        return {}
    ledger = json.loads(ledger_path.read_text())
    entries = ledger.get("entries", {})
    run_to_wp = {run: wp.upper() for wp, run in
                 ledger.get("_meta", {}).get("runs_of_record", {}).items()}
    out = {}
    for fig in man.get("figures", {}).values():
        if fig.get("status") != "produced" or not fig.get("png_path"):
            continue
        wps = sorted({run_to_wp[r] for r in
                      (entries.get(i, {}).get("source", {}).get("run_id") for i in fig.get("ledger_ids_used", []))
                      if r in run_to_wp})
        if wps:
            out[Path(fig["png_path"]).name] = " + ".join(wps)
    return out


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
     "The four figures staged for the paper, each tagged with the work package its numbers come from. "
     "Your promotion ruling is the only thing between these and shared/figures. The numbers behind them, "
     "per work package, are at "
     "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp04.html\">WP04</a>, "
     "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp05.html\">WP05</a>, "
     "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_wp06.html\">WP06</a> and "
     "<a href=\"/morphofavela-dash/outputs/_hub/wp07_staged/review/_results_g3.html\">G3</a>.",
     lambda: _latest("wp07_figures_*")),
    ("02_citywide_maps", "Citywide maps (f5, f5b, f6)",
     "Sky-view and irradiation across the whole 8.4 M-cell domain. Withheld under red line L1 — yours to read, not to circulate.",
     lambda: _latest("wp07_map_*")),
    ("03_zoom_favelas", "Per-favela zoom extracts",
     "Each study favela at the run's sampling pitch, sharing the citywide colour limits. Ipanema is absent: no bairro boundary exists on disk.",
     lambda: _latest("wp07_zoom_*")),
    ("04_terrain_vs_buildings", "Terrain versus buildings",
     "How much of the sun lost to an open flat horizon is the hill, and how much is what was built on it. "
     "The maps put terrain-only beside terrain-with-buildings on one colour scale.",
     lambda: _latest("terrain_split_*")),
    ("05_method_schematics", "How the method works",
     "The obstruction surface built from terrain and building tops, the ray march that decides whether a sky "
     "patch is blocked, and the matrix step that turns visibility into irradiation.",
     lambda: _latest("wp07_method_*")),
]

EXTRA = {
    "06_morphotypes": (
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
    "07_folha_de_rua": (
        "Folha de Rua site sheets",
        "One A3 sheet per site: grid, terrain, density, then sky view and sunlight.",
        sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*_A3.png"))
        + sorted(ROOT.glob("outputs/_distribution/site_dashboards/*/folha_*.pdf")),
    ),
    "08_weekly_deck": (
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
    dest = dest_dir / src.name
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    row = {"section": section, "file": src.name, "status": "ok",
           "source": str(src.relative_to(ROOT)) if ROOT in src.parents else str(src)}
    row.update({k: v for k, v in meta.items() if v is not None})
    if src.suffix.lower() == ".png":
        try:
            with Image.open(src) as im:
                row["pixels"] = list(im.size)
                im.draft("RGB", (THUMB_W, THUMB_W))
                thumb = im.copy()
                thumb.thumbnail((THUMB_W, THUMB_W * 4))
                tdir = dest_dir / "_thumbs"
                tdir.mkdir(exist_ok=True)
                thumb.convert("RGB").save(tdir / (src.stem + ".jpg"), quality=84)
            row["thumb"] = f"_thumbs/{src.stem}.jpg"
        except Exception as exc:
            row["thumb_error"] = type(exc).__name__
    row["bytes"] = src.stat().st_size
    entries.append(row)


# Everything else that already exists under outputs/. Archived snapshots and the
# audit dashboards are excluded: they are stale copies, and showing the PI the
# same figure three times is worse than not showing it.
SWEEP_EXCLUDE = ("_review", "_thumbs", "/thumbs/", "_archive_",
                 "_distribution/audit", "_hub/wp07_staged", "_hub/thumbs")
SWEEP_SUFFIXES = (".png", ".svg", ".pdf")


def sweep_remaining(out_root: Path, already: set, entries: list) -> list:
    """Hardlink every other figure on disk, deduplicated by content. Grouped by
    the directory it came from, because that is what says which analysis it
    belongs to."""
    import hashlib
    from collections import defaultdict

    seen_hashes = set()
    groups = defaultdict(list)
    for src in sorted((ROOT / "outputs").rglob("*")):
        if src.suffix.lower() not in SWEEP_SUFFIXES or not src.is_file():
            continue
        t = str(src)
        if any(x in t for x in SWEEP_EXCLUDE) or src.name in already:
            continue
        h = hashlib.md5(src.read_bytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        groups[src.parent.relative_to(ROOT / "outputs").as_posix()].append(src)

    sections = []
    for i, (rel, srcs) in enumerate(sorted(groups.items())):
        slug = "09_other/" + rel.replace("/", "__")
        for src in srcs:
            _copy(src, out_root / slug, {}, entries, slug)
        sections.append({"slug": slug, "title": rel, "blurb": "",
                         "provenance": f"outputs/{rel}", "group": "other"})
    return sections


def build(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    sections: list[dict] = []

    for slug, title, blurb, resolve in SECTIONS:
        run_dir = resolve()
        if run_dir is None:
            continue
        classes = _manifest_classes(run_dir)
        wps = _wp_by_figure(run_dir)
        img_dir = _image_dir(run_dir)
        for name in sorted(classes):
            meta = dict(classes[name])
            if name in wps:
                meta["work_package"] = wps[name]
            _copy(img_dir / name, out_root / slug, meta, entries, slug)
        sections.append({"slug": slug, "title": title, "blurb": blurb,
                         "provenance": str(run_dir.relative_to(ROOT))})

    for slug, (title, blurb, paths) in EXTRA.items():
        for src in paths:
            _copy(src, out_root / slug, {}, entries, slug)
        sections.append({"slug": slug, "title": title, "blurb": blurb,
                         "provenance": "existing outputs/ and slides/ products"})

    # Drop section directories this build did not write. Renaming a section
    # otherwise leaves its old copy behind and the PI sees it twice.
    written = {s["slug"].split("/")[0] for s in sections}
    for child in out_root.iterdir():
        if child.is_dir() and child.name not in written:
            shutil.rmtree(child)

    already = {e["file"] for e in entries if e["status"] == "ok"}
    sections.extend(sweep_remaining(out_root, already, entries))

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
.wp{background:var(--ink);color:var(--bg);border-color:var(--ink);font-weight:600}
a{color:inherit}.nolink{padding:9px 11px;font-size:13px}
.toc{border:1px solid var(--line);border-radius:8px;padding:16px 18px;background:#fff;margin-bottom:8px}
.toc ul{list-style:none;margin:8px 0 16px;padding:0;display:grid;
grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:2px 18px}
.toc.cols ul,.toc ul.cols{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.toc li{font-size:13px;padding:2px 0}
.toc .n{color:var(--dim);font-variant-numeric:tabular-nums}
h2.divider{margin-top:56px;padding-top:22px;border-top:2px solid var(--line)}
</style>
<header><h1>Figure review</h1>
<p class="blurb">Every figure this cycle produced, in one place. Cards marked
<span class="tag withheld">withheld · L1</span> are yours to read; they do not travel into
the paper or shared figures without your own tap.</p></header>"""]
    live = [s for s in m["sections"] if by_section.get(s["slug"])]
    curated = [s for s in live if s.get("group") != "other"]
    other = [s for s in live if s.get("group") == "other"]

    def anchor(sl):
        return "s-" + sl.replace("/", "-").replace("__", "-")

    parts.append('<nav class="toc"><strong>This review</strong><ul>')
    for s in curated:
        parts.append(f'<li><a href="#{anchor(s["slug"])}">{s["title"]}</a> '
                     f'<span class="n">{len(by_section[s["slug"]])}</span></li>')
    parts.append('</ul>')
    if other:
        n = sum(len(by_section[s["slug"]]) for s in other)
        parts.append(f'<strong>Everything else on disk</strong> '
                     f'<span class="n">{n} figures in {len(other)} folders</span><ul class="cols">')
        for s in other:
            parts.append(f'<li><a href="#{anchor(s["slug"])}">{s["title"]}</a> '
                         f'<span class="n">{len(by_section[s["slug"]])}</span></li>')
        parts.append('</ul>')
    parts.append('</nav>')

    for idx, s in enumerate(curated + other):
        rows = by_section.get(s["slug"], [])
        if not rows:
            continue
        if other and s is other[0]:
            parts.append('<h2 class="divider">Everything else on disk</h2>'
                         '<p class="blurb">Every other figure under outputs/, deduplicated by content and '
                         'grouped by the folder it came from. These are earlier and ongoing analyses, not a '
                         'curated set, and some predate the current reframe. On the hub only the previews of '
                         'this part are mirrored, because the originals are 1.5 GB and the server has 12 GB '
                         'free; open the local folder to reach them at full size.</p>')
        parts.append(f'<h2 id="{anchor(s["slug"])}">{s["title"]}</h2>'
                     + (f'<p class="blurb">{s["blurb"]}</p>' if s["blurb"] else "")
                     + f'<p class="prov">{s["provenance"]}</p><div class="grid">')
        for e in rows:
            if e["status"] == "MISSING":
                parts.append(f'<figure><div class="nolink">missing: <span class="name">{e["file"]}</span></div></figure>')
                continue
            tags = ""
            if e.get("work_package"):
                tags += f'<span class="tag wp">{e["work_package"]}</span>'
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
