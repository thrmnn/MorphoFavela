# WP-04F façade comparison — 2026-09-15T06:01:26Z

device=cuda, torch=2.4.1+cu121, default_inset=0.1 m

## vidigal

| variant | n | zero share | <0.01 share | median | p25 | p75 | 0-3m | 3-6m | 6-9m | >9m |
|---|---|---|---|---|---|---|---|---|---|---|
| a_baseline | 864873 | 0.1720 | 0.3838 | 0.0317 | 0.0029 | 0.1564 | 0.0105 | 0.0413 | 0.0937 | 0.1626 |
| b_own_building_exclusion | 864873 | 0.1460 | 0.3597 | 0.0383 | 0.0040 | 0.1651 | 0.0148 | 0.0484 | 0.1000 | 0.1659 |
| c_inset_1_5m | 864873 | 0.2210 | 0.4311 | 0.0242 | 0.0012 | 0.1498 | 0.0059 | 0.0292 | 0.0828 | 0.1562 |
| d_exclusion_and_inset_1_5m | 864873 | 0.2136 | 0.4177 | 0.0272 | 0.0015 | 0.1564 | 0.0060 | 0.0339 | 0.0870 | 0.1581 |

## riodaspedras

| variant | n | zero share | <0.01 share | median | p25 | p75 | 0-3m | 3-6m | 6-9m | >9m |
|---|---|---|---|---|---|---|---|---|---|---|
| a_baseline | 3879596 | 0.2929 | 0.4763 | 0.0175 | 0.0000 | 0.1959 | 0.0001 | 0.0059 | 0.0682 | 0.2969 |
| b_own_building_exclusion | 3879596 | 0.2519 | 0.4497 | 0.0263 | 0.0000 | 0.2153 | 0.0017 | 0.0068 | 0.0741 | 0.3272 |
| c_inset_1_5m | 3879596 | 0.3291 | 0.5105 | 0.0072 | 0.0000 | 0.1768 | 0.0000 | 0.0056 | 0.0460 | 0.2896 |
| d_exclusion_and_inset_1_5m | 3879596 | 0.3151 | 0.4910 | 0.0128 | 0.0000 | 0.1940 | 0.0000 | 0.0058 | 0.0568 | 0.3169 |

## Physics checks

{
 "unobstructed_vertical_facade_svf": {
  "baseline": [
   0.4951980344277826,
   0.4980746435161477,
   0.4951980344277826,
   0.4980746435161477
  ],
  "own_building_exclusion": [
   0.4951980344277826,
   0.4980746435161477,
   0.4951980344277826,
   0.4980746435161477
  ],
  "expected": 0.5,
  "tolerance": 0.04
 },
 "opposite_wall_closed_form": {
  "D_m": 20.0,
  "H_m": 15.0,
  "observer_height_m": 2.0,
  "baseline": {
   "max_abs_deviation_deg": 1.2582425557966488,
   "mean_abs_deviation_deg": 0.8227656360024288,
   "n_patches_compared": 44
  },
  "own_building_exclusion": {
   "max_abs_deviation_deg": 1.2582425557966488,
   "mean_abs_deviation_deg": 0.8227656360024288,
   "n_patches_compared": 44
  }
 }
}