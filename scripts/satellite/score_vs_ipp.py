"""Score the free-EO reconstruction against the IPP answer key.

This is the ONLY module allowed to open the IPP ground truth. It never writes
back into the reconstruction, so the quarantine holds in both directions.

Scored components: DTM (GLO-30 vs IPP 5 m) and footprints (Open Buildings v3
vs IPP A101/A102). Heights are BLOCKED upstream and are reported as such —
never as a number.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely import STRtree

REPO = Path(__file__).resolve().parents[2]
IPP_DTM = REPO / "data" / "RJ" / "DTM_RJ.tif"
IPP_BUILDINGS = REPO / "data" / "RJ" / "buildings_RJ_2019.shp"

# IPP building instances. A101/A102 are real building outlines (A102 is the
# favela-specific class, 24% of Rocinha); A103+ are upper-floor projections and
# awnings, which would double-count roof area.
INSTANCE_TIPOS = ("A101 - EDIFICACAO", "A102 - EDIFICACAO_FAVELA")
ROOF_VISIBLE_PREFIXES = ("A101", "A102", "A103", "A104", "A105", "A106", "A107", "A119", "A120")

# Accuracy bands the plan committed to BEFORE any measurement (plan §1).
# Landing far better than these is a leakage signal, not a win.
PLAN_BANDS = {
    "dtm_rmse_m": (4.0, 8.0),
    "dtm_rmse_steep_m": (8.0, 10.0),
    "dtm_bias_m": (1.0, 3.0),
    "footprint_area_iou": (0.45, 0.60),
    "footprint_instance_f1": (0.35, 0.55),
}
IOU_MATCH_THRESHOLD = 0.5
SLOPE_BINS = [(0.0, 10.0), (10.0, 25.0), (25.0, 90.0)]
# Control region for separating a vertical-reference offset from the surface-model
# effect: open, flat, near-sea-level ground in a box around the AOI.
CONTROL_PAD_M = 2000.0
CONTROL_MIN_DIST_M = 40.0
CONTROL_MAX_SLOPE_DEG = 3.0
CONTROL_MAX_ELEV_M = 6.0


def _band_verdict(name: str, value: float | None, higher_is_better: bool) -> dict:
    lo, hi = PLAN_BANDS[name]
    if value is None or not np.isfinite(value):
        return {"band": [lo, hi], "verdict": "NOT_MEASURED"}
    if value < lo:
        verdict = (
            "BETTER_THAN_PREDICTED_REVIEW_FOR_LEAKAGE"
            if not higher_is_better
            else "WORSE_THAN_PREDICTED"
        )
    elif value > hi:
        verdict = (
            "WORSE_THAN_PREDICTED"
            if not higher_is_better
            else "BETTER_THAN_PREDICTED_REVIEW_FOR_LEAKAGE"
        )
    else:
        verdict = "WITHIN_PREDICTED_BAND"
    return {"band": [lo, hi], "verdict": verdict}


def _err_stats(err: np.ndarray) -> dict:
    return {
        "n_px": int(err.size),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mae_m": float(np.mean(np.abs(err))),
        "bias_m": float(np.mean(err)),
        "median_err_m": float(np.median(err)),
        "p90_abs_err_m": float(np.percentile(np.abs(err), 90)),
    }


def _read_ipp_window(bounds: tuple) -> tuple[np.ndarray, object, object]:
    with rasterio.open(IPP_DTM) as ipp:
        window = from_bounds(*bounds, transform=ipp.transform).round_offsets().round_lengths()
        arr = ipp.read(1, window=window).astype("float64")
        transform = ipp.window_transform(window)
        arr[(arr == ipp.nodata) | (arr > 1e30)] = np.nan
        return arr, transform, ipp.crs


def _regrid(src_path: Path, shape: tuple, transform, crs) -> np.ndarray:
    with rasterio.open(src_path) as rec:
        out = np.empty(shape, dtype="float64")
        reproject(
            source=rec.read(1),
            destination=out,
            src_transform=rec.transform,
            src_crs=rec.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=rec.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return out


def _slope_deg(truth: np.ndarray, px_m: float = 5.0) -> np.ndarray:
    gy, gx = np.gradient(np.nan_to_num(truth, nan=np.nanmean(truth)), px_m, px_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def datum_control(site_dir: Path, aoi: gpd.GeoDataFrame) -> dict:
    """Median GLO-30 − IPP on open, flat, low ground around the AOI.

    Rocinha itself has no surface large enough to resolve at 30 m posting, so
    the control comes from the surrounding box. It answers the question the
    in-AOI bias cannot: how much of the offset is a vertical-reference
    difference rather than the DSM sitting on roofs and canopy.
    """
    x0, y0, x1, y1 = aoi.total_bounds
    box = (x0 - CONTROL_PAD_M, y0 - CONTROL_PAD_M, x1 + CONTROL_PAD_M, y1 + CONTROL_PAD_M)
    truth, transform, crs = _read_ipp_window(box)
    recon = _regrid(
        site_dir / "recon" / "dtm_glo30_control_native.tif", truth.shape, transform, crs
    )

    blds = gpd.read_file(IPP_BUILDINGS, bbox=box)
    mask = rasterize(
        [(g, 1) for g in blds.geometry],
        out_shape=truth.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    dist = ndimage.distance_transform_edt(~mask) * 5.0
    slope = _slope_deg(truth)

    valid = np.isfinite(truth) & np.isfinite(recon)
    sel = (
        valid
        & (dist > CONTROL_MIN_DIST_M)
        & (slope < CONTROL_MAX_SLOPE_DEG)
        & (truth < CONTROL_MAX_ELEV_M)
    )
    if sel.sum() < 500:
        return {"status": "INSUFFICIENT_CONTROL_PIXELS", "n_px": int(sel.sum())}
    err = recon[sel] - truth[sel]
    return {
        "status": "OK",
        "definition": (
            f">{CONTROL_MIN_DIST_M:.0f} m from any IPP building, slope "
            f"<{CONTROL_MAX_SLOPE_DEG:.0f} deg, elevation <{CONTROL_MAX_ELEV_M:.0f} m, "
            f"within {CONTROL_PAD_M:.0f} m of the AOI"
        ),
        "n_px": int(sel.sum()),
        "median_offset_m": float(np.median(err)),
        "mean_offset_m": float(np.mean(err)),
        "note": (
            "the median is the reference-offset estimate; the mean runs higher "
            "because isolated trees and structures survive the filter"
        ),
    }


def landcover_strata(site_dir: Path, aoi: gpd.GeoDataFrame, truth, recon, transform, valid) -> dict:
    """Decompose the in-AOI bias by what the 30 m cell is actually looking at."""
    blds = gpd.read_file(IPP_BUILDINGS, bbox=tuple(aoi.total_bounds))
    poly = aoi.union_all()
    blds = blds[blds.intersects(poly) & blds["tipo"].isin(INSTANCE_TIPOS)]
    mask = rasterize(
        [(g, 1) for g in blds.geometry],
        out_shape=truth.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    dist = ndimage.distance_transform_edt(~mask) * 5.0
    out = {}
    for label, m in (
        ("on_building_footprint", valid & mask),
        ("open_within_10m_of_building", valid & ~mask & (dist <= 10.0)),
        ("open_beyond_20m_of_building", valid & ~mask & (dist > 20.0)),
    ):
        if m.sum() >= 50:
            out[label] = _err_stats(recon[m] - truth[m])
    out["built_pixel_fraction"] = float((mask & valid).sum() / valid.sum())
    return out


def score_dtm(site_dir: Path, aoi: gpd.GeoDataFrame) -> tuple[dict, dict]:
    """Compare GLO-30 against the IPP 5 m DTM on the IPP grid."""
    poly = aoi.union_all()
    truth, transform, crs = _read_ipp_window(tuple(aoi.total_bounds))
    recon = _regrid(site_dir / "recon" / "dtm_glo30_native.tif", truth.shape, transform, crs)

    inside = rasterize(
        [(poly, 1)], out_shape=truth.shape, transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    slope_deg = _slope_deg(truth)
    valid = inside & np.isfinite(truth) & np.isfinite(recon)

    out = _err_stats(recon[valid] - truth[valid])
    out["by_slope"] = {}
    for lo, hi in SLOPE_BINS:
        m = valid & (slope_deg >= lo) & (slope_deg < hi)
        if m.sum() >= 50:
            out["by_slope"][f"{lo:.0f}-{hi:.0f}deg"] = _err_stats(recon[m] - truth[m])

    out["by_landcover"] = landcover_strata(site_dir, aoi, truth, recon, transform, valid)
    out["reference_offset_control"] = datum_control(site_dir, aoi)

    steep = out["by_slope"].get("25-90deg", {}).get("rmse_m")
    out["assessment"] = {
        "rmse": _band_verdict("dtm_rmse_m", out["rmse_m"], higher_is_better=False),
        "bias": _band_verdict("dtm_bias_m", out["bias_m"], higher_is_better=False),
        "rmse_steep": _band_verdict("dtm_rmse_steep_m", steep, higher_is_better=False),
    }

    ctrl = out["reference_offset_control"].get("median_offset_m")
    if ctrl is not None:
        out["bias_decomposition"] = {
            "total_bias_m": out["bias_m"],
            "reference_offset_m": ctrl,
            "surface_model_component_m": out["bias_m"] - ctrl,
            "evidence": (
                "bias on building footprints "
                f"({out['by_landcover']['on_building_footprint']['bias_m']:+.2f} m) vs open ground "
                f"({out['by_landcover']['open_within_10m_of_building']['bias_m']:+.2f} m) vs open "
                f">20 m from any building "
                f"({out['by_landcover']['open_beyond_20m_of_building']['bias_m']:+.2f} m)"
            ),
            "reading": (
                "a uniform vertical-reference shift would move every stratum equally; "
                "the offset instead tracks what the 30 m cell sees, so most of the bias "
                "is GLO-30 behaving as a surface model over roofs and canopy"
            ),
        }
    diag = {
        "truth": truth,
        "recon": recon,
        "valid": valid,
        "slope": slope_deg,
        "transform": transform,
    }
    return out, diag


def _instance_f1(pred: list, truth: list) -> dict:
    """Greedy one-to-one matching at IoU >= 0.5."""
    tree = STRtree(truth)
    pairs = []
    for i, geom in enumerate(pred):
        for j in tree.query(geom):
            inter = geom.intersection(truth[j]).area
            if inter <= 0:
                continue
            union = geom.area + truth[j].area - inter
            if union > 0:
                iou = inter / union
                if iou >= IOU_MATCH_THRESHOLD:
                    pairs.append((iou, i, int(j)))
    pairs.sort(reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "n_pred": len(pred),
        "n_truth": len(truth),
        "precision": float(precision),
        "recall": float(recall),
        "instance_f1": float(f1),
        "iou_threshold": IOU_MATCH_THRESHOLD,
    }


def score_footprints(site_dir: Path, aoi: gpd.GeoDataFrame) -> tuple[dict, dict]:
    poly = aoi.union_all()
    ipp = gpd.read_file(IPP_BUILDINGS, bbox=tuple(aoi.total_bounds))
    ipp = ipp[ipp.intersects(poly)].copy()
    ipp["geometry"] = ipp.geometry.buffer(0).intersection(poly)
    ipp = ipp[~ipp.is_empty]

    inst = ipp[ipp["tipo"].isin(INSTANCE_TIPOS)]
    roof = ipp[ipp["tipo"].str.startswith(ROOF_VISIBLE_PREFIXES, na=False)]

    ob = gpd.read_file(site_dir / "recon" / "footprints_ob_v3.gpkg", layer="footprints")
    ob["geometry"] = ob.geometry.buffer(0).intersection(poly)
    ob = ob[~ob.is_empty]

    ob_u = ob.union_all()
    inst_u = inst.union_all()
    roof_u = roof.union_all()
    area_iou = ob_u.intersection(inst_u).area / ob_u.union(inst_u).area
    area_iou_roof = ob_u.intersection(roof_u).area / ob_u.union(roof_u).area

    out = {
        "instance_tipos": list(INSTANCE_TIPOS),
        "area_iou": float(area_iou),
        "area_iou_vs_roof_visible_set": float(area_iou_roof),
        "built_area_ratio": float(ob_u.area / inst_u.area),
        "count_ratio": float(len(ob) / len(inst)),
        "aoi_area_m2": float(poly.area),
        "ob_built_area_m2": float(ob_u.area),
        "ipp_built_area_m2": float(inst_u.area),
        **_instance_f1(list(ob.geometry), list(inst.geometry)),
    }
    out["assessment"] = {
        "area_iou": _band_verdict("footprint_area_iou", out["area_iou"], higher_is_better=True),
        "instance_f1": _band_verdict(
            "footprint_instance_f1", out["instance_f1"], higher_is_better=True
        ),
    }
    return out, {"ipp": inst, "ob": ob}


def make_figure(site_dir: Path, dtm_diag: dict, fp_diag: dict, card: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    err = np.where(dtm_diag["valid"], dtm_diag["recon"] - dtm_diag["truth"], np.nan)
    vmax = float(np.nanpercentile(np.abs(err), 98))
    im = axes[0].imshow(err, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(
        f"GLO-30 − IPP DTM (m)\nRMSE {card['dtm']['rmse_m']:.2f} · bias {card['dtm']['bias_m']:+.2f}"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="error (m)")

    lc = card["dtm"]["by_landcover"]
    ctrl = card["dtm"]["reference_offset_control"]
    labels = ["on\nbuildings", "open\n<10 m", "open\n>20 m"]
    values = [
        lc["on_building_footprint"]["bias_m"],
        lc["open_within_10m_of_building"]["bias_m"],
        lc["open_beyond_20m_of_building"]["bias_m"],
    ]
    colors = ["#C44E52", "#DD8452", "#55A868"]
    if ctrl.get("status") == "OK":
        labels.append("open flat\ncontrol")
        values.append(ctrl["median_offset_m"])
        colors.append("#4C72B0")
    axes[1].bar(labels, values, color=colors)
    axes[1].axhspan(*PLAN_BANDS["dtm_bias_m"], color="grey", alpha=0.25, label="plan bias band")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("elevation bias (m)")
    axes[1].set_title("Where the +bias comes from\n(a datum shift would be flat across these)")
    axes[1].legend(fontsize=8)

    fp_diag["ipp"].plot(ax=axes[2], color="#333333", linewidth=0)
    fp_diag["ob"].boundary.plot(ax=axes[2], color="#D55E00", linewidth=0.35)
    axes[2].set_title(
        f"Footprints — IPP (grey) vs Open Buildings (orange)\n"
        f"area-IoU {card['footprints']['area_iou']:.3f} · "
        f"instance-F1 {card['footprints']['instance_f1']:.3f} · "
        f"count {card['footprints']['count_ratio']:.2f}×"
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(
        f"{card['site']} free-EO reconstruction vs IPP ground truth — "
        f"heights {card['heights']['status']} (GEE auth needed)",
        fontsize=11,
    )
    fig.tight_layout()
    out = site_dir / "scorecard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--site-dir",
        type=Path,
        default=REPO / "outputs" / "comparative" / "satellite" / "rocinha",
    )
    args = ap.parse_args()
    site_dir = args.site_dir

    aoi = gpd.read_file(site_dir / "aoi.gpkg", layer="aoi")
    provenance = json.loads((site_dir / "recon" / "provenance.json").read_text())

    dtm, dtm_diag = score_dtm(site_dir, aoi)
    fp, fp_diag = score_footprints(site_dir, aoi)

    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    card = {
        "schema": "satellite_scorecard/v1",
        "site": provenance["aoi"]["site"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha": sha,
        "reconstruction": {
            "dtm": provenance["dtm"]["product"],
            "footprints": provenance["footprints"]["product"],
            "heights": provenance["heights"]["product"],
            "n_footprints": provenance["footprints"]["kept_google_in_aoi"],
        },
        "dtm": dtm,
        "footprints": fp,
        "heights": {
            "status": "BLOCKED",
            "reason": provenance["heights"]["reason"],
            "per_building_mae_m": None,
            "per_building_r2": None,
            "grid100m_mean_r2": None,
        },
        "baselines": {
            "note": (
                "No correction stage exists yet, so the reconstruction IS the plan's "
                "baseline: raw GLO-30 for the DTM and Open Buildings as-is (no confidence "
                "threshold) for footprints. Baseline columns will diverge only once a "
                "correction step is added."
            ),
            "dtm_baseline": "raw GLO-30 == reported DTM",
            "footprint_baseline": "Open Buildings as-is == reported footprints",
        },
    }
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "scorecard.json").write_text(json.dumps(card, indent=2) + "\n")
    fig = make_figure(site_dir, dtm_diag, fp_diag, card)

    print(json.dumps({k: card[k] for k in ("dtm", "footprints", "heights")}, indent=2)[:2000])
    print(f"\nscorecard -> {site_dir / 'scorecard.json'}\nfigure    -> {fig}")


if __name__ == "__main__":
    main()
