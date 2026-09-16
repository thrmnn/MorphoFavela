# VENTAXIS — definition of record for P1's second axis

Status: definition of record, C′-legal. Owner: WP-06/WP-07. Opened 2026-09-16
(cycle 3, `docs/ventaxis_cprime_spec.md`).

This document is the C′ definition of the per-cell **count (0–3) of
independent geometry constraints** that the WP-07 ledger's 25
`wp06.<site>.{n,share_n0..share_n3}` entries report. It supersedes, for every
P1 purpose, the definition strings in `outputs/paper_figures/ventilation_index.json`
(dated 2026-06-28) — a pre-reframe E2 artifact. That file is not edited or
deleted by this document: `scripts/health/compound_exposure.py` still reads it
as its own input.

The numbers themselves do not change. Producing code:
`src/brisa_solar/wp06_geometry.py`; run of record:
`runs/wp06_geometry_20260915T052604Z/summary.json`; ledgered in
`runs/wp07_ledger_20260915T184619Z/ledger.json`.

## What the count is

For each built 10 m cell, `n_constraints` is a **checklist count** (an
integer in {0, 1, 2, 3}) of how many of three independent geometry
predicates the cell triggers. It is a count of qualitative flags, never a
weighted sum of the underlying continuous magnitudes — the three axes are
incommensurable (a frontal-area density fraction, a distance in metres, a
dimensionless ratio) and are never collapsed into one continuous scale.

## The three constraints

Each predicate below is quoted verbatim from its implementation in
`src/brisa_solar/wp06_geometry.py::compute_site_table`.

### 1. Vertical — `constraint_vertical`

```python
grid["constraint_vertical"] = (np.nan_to_num(grid["lambda_f_mean"].to_numpy(), nan=0.0) >= LAMBDA_F_CONSTRAINT_MIN).astype(int)
```

Predicate: `lambda_f_mean >= 0.65`.

- `LAMBDA_F_CONSTRAINT_MIN` is imported from `src/brisa_solar/constants.py`,
  P1's own constants home. **Vocabulary note:** until 2026-09-16 this module
  imported the same value from the June 2026 E2 script
  `scripts/run_ventilation_index.py`, where its identifier names an air-movement regime — vocabulary P1 does not use, and which the whole-word token lint
  cannot see inside an identifier. The value is unchanged (the predicate and
  therefore every ledgered number are identical); only where P1 reads it from
  changed. `tests/test_ventaxis.py` asserts the two definitions can never
  diverge, so the June index and the P1 ledger cannot come to describe
  different cells.
- Threshold source: Oke (1988) — a geometry threshold on frontal-area
  density (`lambda_f`, the built frontal area presented to the wind divided
  by plan area), never cited by a named regime. Cited here purely on that
  geometric basis.
- `lambda_f_mean` is the cell's mean frontal-area density averaged over the
  8 compass sectors (`lambda_f_N` … `lambda_f_NW`), computed upstream in the
  morphometrics grid.

### 2. Lateral — `constraint_lateral`

```python
grid["constraint_lateral"] = (np.nan_to_num(grid["open_edge_dist_m"].to_numpy(), nan=0.0) >= depth_median).astype(int)
```

Predicate: `open_edge_dist_m >= depth_median`.

- `open_edge_dist_m` is the Euclidean distance (m) from the cell to the
  nearest open cell — an unbuilt interior cell or the settlement perimeter
  (`scripts/run_lateral_connectivity.py::open_edge_distance`).
- `depth_median` is the **pooled** median of `open_edge_dist_m` across all
  five study sites, computed by `src/brisa_solar/wp06_geometry.py::pooled_depth_median`
  and read (never typed) from the run of record:
  `runs/wp06_geometry_20260915T052604Z/summary.json#/depth_median_m` =
  **31.622776601683796 m**.

### 3. Directional — `constraint_directional`

```python
grid["constraint_directional"] = (np.nan_to_num(grid["exposure_ratio"].to_numpy(), nan=0.0) >= 1.0).astype(int)
```

Predicate: `exposure_ratio >= 1.0`.

- `exposure_ratio = wind_exposure / lambda_f_mean`, where `wind_exposure`
  is the cell's frontal-area density weighted by the measured wind rose
  (`scripts/run_wind_exposure.py::wind_exposure`, Σ over the 8 sectors of
  `freq(sector) · lambda_f(sector)`).
- Threshold `1.0` is the isotropic baseline: a ratio above 1 means the
  prevailing wind meets an above-average frontal axis of the cell's own
  fabric; a ratio below 1 means the prevailing wind hits the cell's more
  open axis. The threshold is literal in the predicate — no external
  citation.

## What the count is NOT

`n_constraints` is a geometry checklist, not a measurement of air exchange,
ventilation adequacy, or any simulated quantity. P1 makes no claim about
exchange itself. The honest phrasing: these three predicates each describe a
geometric condition that the literature (Oke 1988, and its extensions) has
associated with reduced air exchange — but P1 stops at the geometry. The
quantitative version of the exchange claim — if and when it is made — is
P3's, not P1's, and is referenced here only forward, not asserted.

## Provenance

This document supersedes the definition strings carried by
`outputs/paper_figures/ventilation_index.json` (dated 2026-06-28) for every
P1 purpose. That file predates the C′ reframe (2026-09-08), is a June
artifact of E2, and is not edited or deleted by this task — it remains
another output's input (`scripts/health/compound_exposure.py` reads it
directly). Where the two documents' vocabulary differs, this document
governs P1's WP-06/WP-07 numbers; the June file's own numbers are unchanged
and out of scope here.

The WP-07 ledger's 25 `wp06.*` entries (`runs/wp07_ledger_<UTC>/ledger.json`)
carry a `_meta.definition_of_record` pointer to this file.
