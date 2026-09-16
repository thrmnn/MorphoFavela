# CITYHOURS — a citywide direct-sun-hour layer so the 2 h floor has a citywide denominator

Opened 2026-09-16 (cycle 3). Owner: agent. Repo: MorphoFavela.
Verify gate: `TMPDIR=/tmp python -m pytest tests/test_cityhours.py -q` then `.claude/verify-cmd`.

## The gap, as measured

P1's C′ spine is "the citywide ground-level solar-access distribution, with the
five sites positioned within it", and P1's habitability threshold is the **2 h
direct-sun floor — Athens Charter (1943) Point 26, never WHO**
(`config/params.yaml#/reference_days/floor_provenance` says so in the file).

Measured 2026-09-16 against `runs/wp07_ledger_20260915T184619Z/ledger.json`:

  - the **per-site** layer expresses the floor — `site.<site>.{ground,street}.
    share_ge_{1,2,3,4}h_{winter_solstice,equinox}` and the `sun_h_*` percentiles;
  - the **citywide** layer does not. Its 18 entries are `citywide.svf.p*` and
    `citywide.kwh_m2.p*` only, and the five `favela.<name>.*.percentile`
    positions likewise exist for SVF and annual irradiation alone.

So the paper can today say where a favela sits in the citywide distribution of
*sky view* and of *annual irradiation*, but cannot say what share of the city
meets the floor, nor where a favela sits in the distribution of *hours*. The
headline is expressible only in the geometry currency, not in the habitability
currency the framing rests on.

This is **closing a declared scope, not widening it.** C′ declares the citywide
solar-access distribution as the spine; a spine that cannot express the floor
citywide is an incomplete instantiation of it. Do not read this task as licence
to add anything else to P1.

## Method — and the fidelity fork you must not fudge

The citywide run of record (`runs/wp05_full_20260914T215419Z/wp05_full.parquet`,
8,402,056 rows) stores `visibility_packed` — the binary per-patch sky visibility
(`pack_visibility` / `unpack_visibility` in `src/brisa_solar/wp05_pilot.py`).
The site pipeline's `direct_sun_hours` (`src/brisa_solar/wp04_sites.py`) does
**not** consume that mask: it consumes continuous per-azimuth `horizon_deg` and
tests `sun_altitude > horizon_angle at the nearest sampled patch azimuth`, at
1 h and 10 min resolution.

Two routes, and they are not equivalent:

  - **(a) from the stored mask.** Cheap (minutes). But a horizon recovered from
    a binary patch mask is quantised to the sky scheme's altitude rings, far
    coarser than the continuous horizon the site numbers use. Mixing an (a)
    citywide number with a site number in one sentence is exactly the
    one-resolution violation §1.2 of the C′ plan exists to prevent.
  - **(b) re-run the horizon engine emitting continuous horizon angles, then
    the same `direct_sun_hours`.** The full run of record took **wall_s 1948.7
    (32.5 min), peak 0.13 GB, 325 tiles, device cuda** — read those from
    `runs/wp05_full_20260914T215419Z/manifest.json`, do not trust this line.
    Affordable.

**Take route (b).** Compute (a) as well, but *only* as a cross-check: report the
correlation and the median absolute difference between (a) and (b), so the
project learns how lossy the stored mask is. (a) is never a published number.

**Follow the project's own pilot rule** (`config/params.yaml#/sampling/
pilot_rule`): a 1–2 % stratified pilot runs FIRST and its wall-time/memory
extrapolation is the go/no-go for the full run. If the pilot extrapolates past
~3 hours, say so and STOP for a decision — **ORCD/HPC is not authorised for P1
and a pilot overrun is answered by cutting the sample, never by reaching for the
cluster.** Laptop GPU only.

**Reproduction check (free, and the point of doing it this way).** The re-run
recomputes `svf` and `kwh_m2` for the same cells. Assert they reproduce the run
of record. Bitwise identity is NOT a property this kernel has (chunked parallel
reductions — C′ plan §1.3); use the project's established golden-tile tolerance
instead (corr ≥ 0.9999, max-abs-diff ≤ 0.01) and report the achieved numbers. A
miss is a STOP-AND-REPORT finding about the run of record, not something to fix
by loosening the tolerance.

## Outputs, and the release boundary

**Withheld (red line L1).** The per-cell citywide layer — hours per cell — is a
per-cell citywide layer and is **withheld**. It stays in `runs/cityhours_<UTC>/`
and is never copied into `shared/`, `papers/`, `outputs/_distribution/`, or
anything the public hub serves. Compute-but-withhold is the standing policy;
staging is the agent's job, promotion is the PI's tap.

**Publishable-candidate summaries** (the same shapes that already cleared for
SVF and kWh/m²), written to `runs/cityhours_<UTC>/summary.json`:
  - citywide `share_ge_{1,2,3,4}h` for each reference day;
  - citywide sun-hour percentiles (the same percentile set the existing
    `citywide.*` entries use — read that set from the ledger, do not retype it);
  - each favela's **percentile position within the citywide sun-hour
    distribution**, matching `favela.<name>.svf.percentile` in shape.

**Not computed, at all:** any favela-versus-formal contrast, difference, deficit
or ratio. The parquet carries `favela_id`; a non-favela aggregate is one line of
code away and it is a red line. If you find yourself writing it, stop.

**Ledger.** Extend `src/brisa_solar/wp07_ledger.py` so the new summaries enter
the ledger under `citywide.sun_h.*`, `citywide.share_ge_*` and
`favela.<name>.sun_h.percentile`, each with its source run and pointer, and
regenerate into a new `runs/wp07_ledger_<UTC>/`. Every pre-existing entry must
keep its value unchanged — assert that in code and paste the output.

## Explicitly OUT

- Any HPC/ORCD submit. Laptop GPU only.
- Promoting anything out of `runs/`.
- The favela-vs-formal comparison in any form.
- Manuscript prose (paper voice is the PI's, always-ask).
- Re-opening the façade surface, G2, or any /ops card.
- Changing the sky scheme, the domain thresholds, the cell size or the seed —
  every one of those is a decided parameter and changing it silently invalidates
  the reproduction check.

## Never

- Never type a number that exists in a file — read it by code, including the
  sky-patch count (import it from `src/brisa_solar/constants.py`).
- Never say "WHO" for the 2 h floor — Athens Charter (1943), Point 26.
- Never publish an (a)-route number.
- Never loosen an acceptance tolerance to make a check pass.
- Background jobs never notify you: run the pilot in the foreground; for the
  full run, background it and wait on its PID with
  `timeout 540 tail --pid=<PID> -f /dev/null`, reading the log between waits.
