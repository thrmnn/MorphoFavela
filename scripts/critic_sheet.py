#!/usr/bin/env python3
"""Render review artefacts for the critic loop: PDF pages -> PNGs, and any set
of PNGs -> one numbered contact sheet. One Read of the sheet gives a grader the
whole artefact; flagged pages are then opened at full resolution.

    python3 scripts/critic_sheet.py pages <file.pdf> <out_dir> [--dpi 80]
    python3 scripts/critic_sheet.py sheet <out.png> <png> [<png> ...] [--cols 4] [--tile-w 620]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def pages(pdf: Path, out_dir: Path, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(out_dir / "page")], check=True)
    files = sorted(out_dir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[1]))
    for f in files:
        print(f)
    return files


def sheet(out: Path, pngs: list[Path], cols: int, tile_w: int) -> Path:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        scale = tile_w / im.width
        tiles.append(im.resize((tile_w, max(1, round(im.height * scale))), Image.LANCZOS))
    pad, label = 18, 34
    tile_h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (tile_h + label + pad) + pad), "#202020")
    draw = ImageDraw.Draw(canvas)
    for i, (t, p) in enumerate(zip(tiles, pngs)):
        r, c = divmod(i, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + label + pad)
        draw.text((x, y), f"{i + 1}  {p.name}", fill="#f0f0f0", font=font)
        canvas.paste(t, (x, y + label))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out, canvas.size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("pdf", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--dpi", type=int, default=80)
    s = sub.add_parser("sheet")
    s.add_argument("out", type=Path)
    s.add_argument("pngs", type=Path, nargs="+")
    s.add_argument("--cols", type=int, default=4)
    s.add_argument("--tile-w", type=int, default=620)
    a = ap.parse_args()
    if a.cmd == "pages":
        pages(a.pdf, a.out_dir, a.dpi)
    else:
        sheet(a.out, a.pngs, a.cols, a.tile_w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
