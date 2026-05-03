# Sky Exposure Plane Analysis: Methodology

## Overview

This document describes the sky exposure plane exceedance analysis, which evaluates
whether buildings comply with sky plane envelopes designed to ensure solar access
and natural ventilation. The analysis operates at street level, implementing
Brazilian building code rulesets (Rio de Janeiro and Sao Paulo).

## Conceptual Framework

The sky exposure plane is an **invisible envelope** that defines the maximum allowed
building height at any point. Buildings within the envelope promote solar access and
ventilation; those exceeding it create environmental deficits.

The envelope works in two zones:

1. **Within setback area** (close to property edges): buildings can rise to a fixed
   base height without restriction.
2. **Beyond setback area**: the allowed height increases linearly with distance from
   the setback boundary, at a ratio defined by the ruleset.

Exceedance at each point is:
```
exceedance = max(0, actual_building_height - envelope_height)
```

---

## Ruleset Specifications

### Rio de Janeiro Ruleset (Default)

| Parameter | Value |
|-----------|-------|
| Sky Plane Ratio | 1/5 (H/5) -- recede 1m for every 5m of height |
| Minimum Setback | 2.50 m |
| Height Measurement | Floor level of first ventilated compartment to floor above last compartment |

```
setback = max(2.50m, building_height / 5)
envelope_height = base_height + (distance_to_setback x 5)
```

### Sao Paulo Ruleset

| Parameter | Value |
|-----------|-------|
| Base Height Threshold | 10.00 m (no restriction below this) |
| Sky Plane Formula | A = (H - 6) / 10 -- 1/10 ratio |
| Minimum Setback | 3.00 m |
| Height Measurement | Lowest point of natural terrain to top (excluding attics/parapets up to 1.20m) |

```
If building_height <= 10m: envelope_height = 10m (no recession)
If building_height >  10m:
  setback = max(3.00m, (building_height - 6) / 10)
  envelope_height = 10 + (distance_to_setback x 10)
```

### Comparison

| Aspect | Rio | Sao Paulo |
|--------|-----|-----------|
| Ratio | 1/5 (more restrictive) | 1/10 (more permissive) |
| Min Setback | 2.50 m | 3.00 m |
| Base Height | Variable (first ventilated floor) | Fixed 10 m |

---

## Street-Based Algorithm

```
FOR each street segment:
  Sample points along centerline (3-5m spacing)

  FOR each street point:
    1. Find nearby buildings (within search radius)

    2. FOR each nearby building:
       a. Calculate building height
       b. Determine base height
       c. Calculate setback per ruleset
       d. Create setback polygon
       e. Calculate envelope height at street point
       f. Extract actual building height (mesh ray casting)
       g. Calculate exceedance

    3. Aggregate exceedances (maximum across buildings)
    4. Store point-level results

  Aggregate to segment-level statistics
```

## Output Structure

### Point-Level (`street_sky_exposure_points.gpkg`)
- `exceedance`: Maximum exceedance at this point (meters)
- `envelope_height`: Allowed height per ruleset
- `actual_height`: Actual building height
- `buildings_affecting`: Number of buildings affecting this point
- `ruleset`: Rio or Sao Paulo

### Segment-Level (`street_sky_exposure_segments.gpkg`)
- `mean_exceedance`, `max_exceedance`: Exceedance statistics
- `exceedance_ratio`: Percentage of segment with violations
- `total_exceedance_volume`: Estimated volume exceeding envelope

## Usage

```bash
python scripts/analyze_sky_exposure_streets.py \
    --stl <stl> --footprints <footprints> \
    [--roads <roads>] --ruleset rio --area <area>
```

## Implementation Status

All phases complete:
1. Core functionality (`analyze_sky_exposure_streets.py`)
2. Building interaction (spatial queries, mesh extraction, height calculation)
3. Envelope and exceedance (Rio and Sao Paulo rulesets)
4. Visualization (maps, sections, statistics)
