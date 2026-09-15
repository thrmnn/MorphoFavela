# G3 / HD-02 domain sensitivity

_2026-09-15T04:59:29Z — PROVISIONAL — g3_domain_sensitivity, HD-02 not yet PI-signed_

## Grid: frame size, favela share, citywide medians

| threshold | distance_m | frame_cells | n_added_cells | favela_share | citywide SVF median | citywide kWh/m² median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5 | 5556648 | 154059 | 0.1178 | 0.6668 | 1245.1 |
| 0.05 | 10 | 8746557 | 344501 | 0.0966 | 0.7625 | 1414.1 |
| 0.05 | 20 | 12182233 | 3780177 | 0.0762 | 0.8326 | 1525.1 |
| 0.10 | 5 | 5402589 | 0 | 0.1193 | 0.6616 | 1236.4 |
| 0.10 | 10 | 8402056 | 0 | 0.0984 | 0.7548 | 1401.8 |
| 0.10 | 20 | 11317913 | 2915857 | 0.0795 | 0.8185 | 1503.8 |
| 0.20 | 5 | 4899492 | 0 | 0.1215 | 0.6494 | 1215.7 |
| 0.20 | 10 | 7384163 | 0 | 0.1010 | 0.7370 | 1372.6 |
| 0.20 | 20 | 9245387 | 1861224 | 0.0857 | 0.7869 | 1454.4 |
| n/a (WP-04) | n/a (WP-04) | n/a | n/a | n/a | 0.7548 | 1401.8 |

## Per-favela percentile-of-median (SVF)

| variant | Vidigal | Rocinha | Complexo do Alemão | Maré | Rio das Pedras |
|---|---|---|---|---|---|
| 0.05/5m | 16.1 | 18.2 | 39.8 | 9.1 | 14.0 |
| 0.05/10m | 14.5 | 18.4 | 36.6 | 7.6 | 12.6 |
| 0.05/20m | 12.7 | 17.4 | 31.8 | 6.3 | 11.9 |
| 0.10/5m | 16.4 | 18.1 | 40.1 | 9.3 | 14.2 |
| 0.10/10m (base) | 15.0 | 18.3 | 36.8 | 7.9 | 13.0 |
| 0.10/20m | 13.5 | 17.2 | 32.4 | 6.6 | 12.5 |
| 0.20/5m | 17.0 | 18.2 | 39.9 | 9.8 | 15.0 |
| 0.20/10m | 15.6 | 18.2 | 36.5 | 8.4 | 13.9 |
| 0.20/20m | 14.5 | 17.4 | 32.7 | 7.4 | 14.0 |
| wp04_polygon_interior | 16.6 | 25.7 | 43.8 | 8.7 | 17.4 |

## Per-favela percentile-of-median (kWh/m²)

| variant | Vidigal | Rocinha | Complexo do Alemão | Maré | Rio das Pedras |
|---|---|---|---|---|---|
| 0.05/5m | 23.6 | 18.5 | 40.1 | 9.0 | 15.8 |
| 0.05/10m | 21.9 | 18.7 | 36.6 | 7.5 | 14.1 |
| 0.05/20m | 19.7 | 17.5 | 31.9 | 6.3 | 13.4 |
| 0.10/5m | 24.0 | 18.5 | 40.4 | 9.2 | 16.1 |
| 0.10/10m (base) | 22.5 | 18.5 | 36.8 | 7.7 | 14.5 |
| 0.10/20m | 20.7 | 17.3 | 32.5 | 6.6 | 14.1 |
| 0.20/5m | 24.5 | 18.4 | 40.4 | 9.7 | 16.9 |
| 0.20/10m | 23.2 | 18.5 | 36.4 | 8.2 | 15.3 |
| 0.20/20m | 21.9 | 17.5 | 32.7 | 7.3 | 15.6 |
| wp04_polygon_interior | 24.0 | 26.0 | 44.0 | 8.7 | 19.0 |

## Max-min spread of percentile-of-median across all variants (grid + WP-04)

| favela | SVF spread (pts) | kWh/m² spread (pts) |
|---|---:|---:|
| Vidigal | 4.3 | 4.9 |
| Rocinha | 8.5 | 8.7 |
| Complexo do Alemão | 12.1 | 12.1 |
| Maré | 3.6 | 3.4 |
| Rio das Pedras | 5.6 | 5.7 |

## card_draft

```json
{
 "id": "g3_domain",
 "question": "Which analysis-domain definition (config/params.yaml `domain.fabric_coverage_threshold` / `fabric_footprint_distance_m`) should be locked for the headline citywide-percentile claim?",
 "options": [
  {
   "variant": "tightest (0.20 / 5m)",
   "fabric_coverage_threshold": 0.2,
   "fabric_footprint_distance_m": 5.0,
   "detail": {
    "frame_cells": 4899492,
    "favela_share": 0.121519945333108,
    "citywide_svf_median": 0.6493670246712346,
    "citywide_kwh_m2_median": 1215.7256239920214,
    "study_favelas_svf_percentile": {
     "Vidigal": 17.005273199752136,
     "Rocinha": 18.175823126152668,
     "Complexo do Alem\u00e3o": 39.94928453807048,
     "Mar\u00e9": 9.82891695710494,
     "Rio das Pedras": 15.046427262254943
    }
   }
  },
  {
   "variant": "grid centre \u2014 current WP-05 default (0.10 / 10m)",
   "fabric_coverage_threshold": 0.1,
   "fabric_footprint_distance_m": 10.0,
   "detail": {
    "frame_cells": 8402056,
    "favela_share": 0.09843531154755455,
    "citywide_svf_median": 0.7547693369233223,
    "citywide_kwh_m2_median": 1401.8269136093218,
    "study_favelas_svf_percentile": {
     "Vidigal": 14.998269471186576,
     "Rocinha": 18.261006591719934,
     "Complexo do Alem\u00e3o": 36.83535315641791,
     "Mar\u00e9": 7.862432718848815,
     "Rio das Pedras": 12.997015254361552
    }
   }
  },
  {
   "variant": "loosest (0.05 / 20m)",
   "fabric_coverage_threshold": 0.05,
   "fabric_footprint_distance_m": 20.0,
   "detail": {
    "frame_cells": 12182233,
    "favela_share": 0.07624464250519589,
    "citywide_svf_median": 0.8326167372144444,
    "citywide_kwh_m2_median": 1525.0909832369916,
    "study_favelas_svf_percentile": {
     "Vidigal": 12.72486743604395,
     "Rocinha": 17.395012884747814,
     "Complexo do Alem\u00e3o": 31.771716236259806,
     "Mar\u00e9": 6.260272644596438,
     "Rio das Pedras": 11.851841940636007
    }
   }
  }
 ],
 "recommended": "grid centre \u2014 current WP-05 default (0.10 / 10m)",
 "reason": "Closest to the sensitivity grid's centre and identical to the frame every other WP-05/WP-04 deliverable already reports against; the table shows no favela crossing a qualitatively different rank at the grid's edges, so there is no evidence to prefer a tighter or looser domain over the one already in use."
}
```
