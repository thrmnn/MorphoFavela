# G3 / HD-02 domain sensitivity

_2026-10-07T21:00:38Z — PROVISIONAL — g3_domain_sensitivity, HD-02 not yet PI-signed_

## Grid: frame size, favela share, citywide medians

| threshold | distance_m | frame_cells | n_added_cells | favela_share | citywide SVF median | citywide kWh/m² median |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 5 | 5556648 | 154059 | 0.1178 | 0.6815 | 1269.6 |
| 0.05 | 10 | 8746557 | 344501 | 0.0966 | 0.7800 | 1440.6 |
| 0.05 | 20 | 12182233 | 3780177 | 0.0762 | 0.8509 | 1550.3 |
| 0.10 | 5 | 5402589 | 0 | 0.1193 | 0.6763 | 1260.8 |
| 0.10 | 10 | 8402056 | 0 | 0.0984 | 0.7723 | 1428.7 |
| 0.10 | 20 | 11317913 | 2915857 | 0.0795 | 0.8373 | 1531.4 |
| 0.20 | 5 | 4899492 | 0 | 0.1215 | 0.6635 | 1239.1 |
| 0.20 | 10 | 7384163 | 0 | 0.1010 | 0.7542 | 1400.1 |
| 0.20 | 20 | 9245387 | 1861224 | 0.0857 | 0.8062 | 1484.2 |
| n/a (WP-04) | n/a (WP-04) | n/a | n/a | n/a | 0.7723 | 1428.7 |

## Per-favela percentile-of-median (SVF)

| variant | Vidigal | Rocinha | Complexo do Alemão | Maré | Rio das Pedras |
|---|---|---|---|---|---|
| 0.05/5m | 31.1 | 15.8 | 37.1 | 11.8 | 11.8 |
| 0.05/10m | 28.1 | 15.7 | 33.3 | 9.5 | 10.4 |
| 0.05/20m | 23.5 | 14.2 | 27.8 | 7.4 | 9.3 |
| 0.10/5m | 31.8 | 15.7 | 37.3 | 12.1 | 12.1 |
| 0.10/10m (base) | 28.9 | 15.6 | 33.6 | 9.9 | 10.7 |
| 0.10/20m | 25.0 | 14.1 | 28.5 | 7.9 | 10.0 |
| 0.20/5m | 32.8 | 15.9 | 37.3 | 13.0 | 13.0 |
| 0.20/10m | 30.4 | 15.7 | 33.5 | 10.8 | 11.7 |
| 0.20/20m | 27.5 | 14.7 | 29.1 | 9.2 | 11.6 |
| wp04_polygon_interior | 29.7 | 23.0 | 41.0 | 11.0 | 14.6 |

## Per-favela percentile-of-median (kWh/m²)

| variant | Vidigal | Rocinha | Complexo do Alemão | Maré | Rio das Pedras |
|---|---|---|---|---|---|
| 0.05/5m | 29.1 | 16.1 | 37.6 | 14.1 | 13.6 |
| 0.05/10m | 26.1 | 16.0 | 33.7 | 11.3 | 11.7 |
| 0.05/20m | 22.2 | 14.4 | 28.3 | 8.8 | 10.6 |
| 0.10/5m | 29.7 | 16.2 | 37.9 | 14.4 | 13.9 |
| 0.10/10m (base) | 26.9 | 15.8 | 33.9 | 11.7 | 12.0 |
| 0.10/20m | 23.6 | 14.2 | 29.0 | 9.5 | 11.3 |
| 0.20/5m | 30.5 | 16.2 | 38.0 | 15.2 | 14.8 |
| 0.20/10m | 27.9 | 16.0 | 33.7 | 12.6 | 13.0 |
| 0.20/20m | 25.6 | 14.7 | 29.5 | 10.8 | 13.0 |
| wp04_polygon_interior | 28.3 | 23.4 | 41.5 | 13.6 | 16.5 |

## Max-min spread of percentile-of-median across all variants (grid + WP-04)

| favela | SVF spread (pts) | kWh/m² spread (pts) |
|---|---:|---:|
| Vidigal | 9.3 | 8.3 |
| Rocinha | 8.9 | 9.1 |
| Complexo do Alemão | 13.3 | 13.3 |
| Maré | 5.6 | 6.4 |
| Rio das Pedras | 5.3 | 5.9 |

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
    "citywide_svf_median": 0.6634621904998879,
    "citywide_kwh_m2_median": 1239.1248636064756,
    "study_favelas_svf_percentile": {
     "Vidigal": 32.81854526959121,
     "Rocinha": 15.91740531467344,
     "Complexo do Alem\u00e3o": 37.33713617656687,
     "Mar\u00e9": 12.992326551405737,
     "Rio das Pedras": 12.99617388904809
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
    "citywide_svf_median": 0.7722794676209415,
    "citywide_kwh_m2_median": 1428.6951483506584,
    "study_favelas_svf_percentile": {
     "Vidigal": 28.9493845315956,
     "Rocinha": 15.553597833673093,
     "Complexo do Alem\u00e3o": 33.59433690991824,
     "Mar\u00e9": 9.854718892613903,
     "Rio das Pedras": 10.737568280906483
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
    "citywide_svf_median": 0.8508994863142154,
    "citywide_kwh_m2_median": 1550.3151816719444,
    "study_favelas_svf_percentile": {
     "Vidigal": 23.49657078468291,
     "Rocinha": 14.21373240850015,
     "Complexo do Alem\u00e3o": 27.76375234326909,
     "Mar\u00e9": 7.361967218981939,
     "Rio das Pedras": 9.337951424833198
    }
   }
  }
 ],
 "recommended": "grid centre \u2014 current WP-05 default (0.10 / 10m)",
 "reason": "Closest to the sensitivity grid's centre and identical to the frame every other WP-05/WP-04 deliverable already reports against; the table shows no favela crossing a qualitatively different rank at the grid's edges, so there is no evidence to prefer a tighter or looser domain over the one already in use."
}
```
