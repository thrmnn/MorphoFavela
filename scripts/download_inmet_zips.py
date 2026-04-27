#!/usr/bin/env python3
"""Robust INMET BDMEP ZIP downloader with resume + retries.

The INMET portal at `portal.inmet.gov.br/uploads/dadoshistoricos/`
drops large transfers mid-stream from a single IP. `curl 7.68` (the
WSL default) trips on missing flags (`--retry-all-errors` is 7.71+)
and `-C -` resume keeps "completing" if Content-Length matches a
truncated body. This script:

  - HEAD-checks the expected Content-Length per ZIP
  - Downloads with HTTP Range when a partial exists
  - Validates with `unzip -tq` after each attempt
  - Resets the partial if it reaches full size but is corrupt

Usage:
    python scripts/download_inmet_zips.py \\
        --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \\
        --out-dir data/inmet/raw

Pair with `scripts/extract_inmet_stations.py` to pull per-station
CSVs out of the yearly ZIPs.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from pathlib import Path

URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{}.zip"
HEADERS = {"User-Agent": "Mozilla/5.0"}
CHUNK = 1024 * 1024  # 1 MB


def expected_size(year: str) -> int:
    req = urllib.request.Request(URL.format(year), headers=HEADERS, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def download_partial(year: str, want: int, out_dir: Path) -> int:
    out = out_dir / f"{year}.zip"
    have = out.stat().st_size if out.exists() else 0
    if have >= want:
        return have
    headers = dict(HEADERS)
    if have > 0:
        headers["Range"] = f"bytes={have}-{want - 1}"
    req = urllib.request.Request(URL.format(year), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        mode = "ab" if have > 0 else "wb"
        with open(out, mode) as f:
            while True:
                buf = r.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
    return out.stat().st_size


def is_valid(year: str, out_dir: Path) -> bool:
    out = out_dir / f"{year}.zip"
    if not out.exists():
        return False
    rc = subprocess.run(
        ["unzip", "-tq", str(out)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--years", nargs="+", required=True, help="e.g. 2015 2016 ... 2024")
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Destination directory for the ZIPs"
    )
    ap.add_argument("--max-attempts", type=int, default=12)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        print(f"=== {year} ===", flush=True)
        if is_valid(year, args.out_dir):
            print(f"  {year} already valid; skipping", flush=True)
            continue
        try:
            want = expected_size(year)
        except Exception as e:
            print(f"  HEAD failed: {e}", flush=True)
            continue
        for attempt in range(1, args.max_attempts + 1):
            try:
                got = download_partial(year, want, args.out_dir)
                print(f"  attempt {attempt}: got {got:,}/{want:,}", flush=True)
                if got >= want and is_valid(year, args.out_dir):
                    print(f"  {year} OK (attempt {attempt})", flush=True)
                    break
                if not is_valid(year, args.out_dir) and got >= want:
                    print(f"  {year} corrupt at full size; resetting", flush=True)
                    (args.out_dir / f"{year}.zip").unlink(missing_ok=True)
            except Exception as e:
                print(f"  attempt {attempt} error: {e}", flush=True)
                time.sleep(20)
        else:
            print(f"  {year} FAILED after {args.max_attempts} attempts", flush=True)

    print("=== ALL DONE ===", flush=True)
    for year in args.years:
        print(
            ("OK" if is_valid(year, args.out_dir) else "BAD") + f" {year}.zip",
            flush=True,
        )


if __name__ == "__main__":
    main()
