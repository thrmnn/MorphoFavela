# Sky Exposure Plane Exceedance: Parameters & Formula

## Current Parameters

The sky exposure plane analysis uses the following default parameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Sky Plane Angle** | `45.0°` | Angle of the inclined plane rising from setback boundaries |
| **Base Height** | `7.5 m` | Maximum height allowed within the setback area (~2-3 floors) |
| **Front Setback** | `5.0 m` | Minimum distance from front edge of building footprint |
| **Side/Rear Setback** | `3.0 m` | Minimum distance from side/rear edges of building footprint |

**Note**: The unified script (`analyze_sky_exposure_streets.py`) now implements Brazilian building code rulesets (Rio and São Paulo) instead of the fixed 45° envelope. See `STREET_SKY_EXPOSURE_METHODOLOGY.md` for details.

The legacy script (`analyze_sky_exposure.py`) with fixed parameters is deprecated but can still be used:
```bash
python scripts/analyze_sky_exposure.py \
    --angle 45.0 \
    --base-height 7.5 \
    --front-setback 5.0 \
    --side-setback 3.0
```

## The Formula Explained in Simple Words

### Conceptual Framework

Think of the sky exposure plane as an **invisible envelope** that defines the maximum allowed building height at any point. This envelope is designed to ensure good environmental performance (solar access and ventilation).

### The Formula (Two Zones)

The envelope height calculation works in two zones:

#### Zone 1: Within Setback Area
**Location**: Close to the building footprint edges (within setback distance)

**Rule**: 
```
Allowed Height = Base Height (7.5m)
```

**In simple words**: 
- Buildings can be up to 7.5 meters tall (about 2-3 floors) right next to their property lines
- This gives a "base allowance" before the sky plane restriction kicks in

#### Zone 2: Beyond Setback Area
**Location**: Further away from the building footprint edges (beyond setback distance)

**Rule**:
```
Allowed Height = Base Height + (Distance from Setback Boundary × tan(45°))
```

**Breaking it down**:
1. **Base Height (7.5m)**: Start with the base allowance
2. **Distance from Setback**: Measure how far you are from the setback line (the line that's 3-5m inside the property edge)
3. **tan(45°)**: At 45 degrees, this equals 1.0, which means:
   - For every 1 meter you move away from the setback line, you're allowed 1 additional meter of height
   - This creates a 1:1 slope (45° angle)

**In simple words**:
- If you're 5 meters away from the setback line, you can build up to 7.5m + (5m × 1.0) = 12.5 meters tall
- If you're 10 meters away, you can build up to 7.5m + (10m × 1.0) = 17.5 meters tall
- The building gets progressively taller as you move away from the edges

### Visual Explanation

```
                    Sky Exposure Plane (45°)
                            ╱
                           ╱
                          ╱
                         ╱
              ┌─────────╱──────────┐  ← Maximum allowed height
              │        ╱            │     (slopes up at 45°)
              │       ╱             │
              │      ╱              │
              │     ╱  ← Base Height │
              │    ╱   (7.5m)      │
    ┌─────────┼────╱───────────────┼─────────┐
    │         │   ╱                │         │ ← Setback line
    │         │  ╱                 │         │   (3-5m from edge)
    └─────────┼──╱─────────────────┼─────────┘
              │ ╱                  │
    ←── 3-5m ──→                    ← Building footprint edge
```

### Example Calculation

For a point that is:
- **8 meters** away from the setback boundary
- **Beyond** the setback area (so Zone 2 applies)

**Calculation**:
```
Envelope Height = 7.5m + (8m × tan(45°))
                = 7.5m + (8m × 1.0)
                = 7.5m + 8m
                = 15.5 meters
```

This point can legally have a building up to 15.5m tall. If the actual building is taller (e.g., 18m), the **exceedance** is 18m - 15.5m = **2.5 meters**.

### Exceedance Calculation

For each building point:
```
Exceedance = Actual Building Height - Envelope Height

If Exceedance > 0: Building exceeds the envelope (bad for environment)
If Exceedance ≤ 0: Building is within envelope (good for environment)
```

The **exceedance ratio** is calculated as:
```
Exceedance Ratio (%) = (Volume Above Envelope / Total Volume) × 100%
```

## Why These Parameters?

### Base Height (7.5m)
- Allows 2-3 story buildings without restriction
- Represents typical informal settlement building heights
- Balances livability with environmental considerations

### Setbacks (5m front, 3m side/rear)
- Creates breathing space between buildings
- Allows light and air to reach ground level
- Different values for front vs side reflect typical street widths

### Sky Plane Angle (45°)
- Creates a gradual height restriction
- Allows taller buildings in the center while protecting edges
- At 45°, height increases 1 meter for every 1 meter away from setback (1:1 ratio)

## Environmental Purpose

This envelope is NOT a legal building code, but rather an **environmental performance standard** that:
- Ensures adequate **solar access** (sunlight reaches ground level)
- Promotes **natural ventilation** (air can flow between buildings)
- Prevents excessive **canyon effects** (buildings too close/tall creating dark, stuffy spaces)

## Parameter Sensitivity

**If you increase the angle** (e.g., 60°):
- Steeper slope → buildings can be taller
- Less restrictive → more built form allowed
- Less environmental protection

**If you decrease the angle** (e.g., 30°):
- Gentler slope → buildings must be shorter
- More restrictive → less built form allowed
- Better environmental protection

**If you increase base height**:
- More building volume allowed near edges
- Less restrictive overall
- May reduce environmental quality

**If you increase setbacks**:
- More space required from edges
- More restrictive on building footprint
- Better environmental performance

## Summary

**Simple Formula**: 
1. Close to edges (< 3-5m): Can build up to 7.5m tall
2. Further from edges (> 3-5m): Can build 7.5m + (distance × 1.0) meters tall
3. Anything above this is "exceedance" - bad for solar access and ventilation

The analysis tells you **how much** and **where** buildings exceed this environmental standard.

