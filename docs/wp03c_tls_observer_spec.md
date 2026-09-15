# WP-03C — TLS vs ALS at ground with a shared observer elevation — spec (2026-09-15)

Status: third and LAST bounded round for task WP03 (refinement, not critical
path). Phases 1–2 are merged (`runs/wp03_tls_20260915T205422Z`,
`runs/wp03_tls_20260915T213422Z`). Phase 2 concluded "genuine mismatch" but
left the one confound the orchestrator can point to untested:

## The untested confound

`wp03_tls.py` (G2 ground comparison, `patch_visibility(...)` calls near
line 487–493) passes NO `obs_z`, so on each surface the observer stands at
`surface[cell] + 1.5 m`. On the TLS `max`-DSM a ground cell that received any
façade, overhang, wire or vegetation point is lifted by metres, so the
observer on the "TLS" surface is standing on a wall while the ALS observer
stands on the ground. That alone produces "TLS sees more sky" (Δ < 0) and
destroys the per-point correlation — regardless of how holes are filled.
Phase 2's variant (c) was also void by construction: the coverage disc used
`max_dist_m` = 500 m, larger than the TLS raster, so no observer passed.

## Deliverables (extend `wp03_tls.py`; new run folder)

1. **Shared observer elevation**: for every G2 observer, `obs_z = ALS DTM at
   the observer cell + 1.5 m` on BOTH surfaces (the DTM is the C′ ground of
   record); report variant (e) = ALS-fill surface + shared obs_z. Add variant
   (f) = the same with `obs_z = SMRF TLS ground + 1.5` (phase 2 wrote
   `tls_ground_smrf_1m.tif`) to show sensitivity to the ground source.
2. **Observer-cell sanity**: share of G2 observer cells whose TLS DSM is
   > 2 m above the DTM (the "standing on a wall" share) per alley class —
   this number tells the reader whether the confound was real.
3. **Variant (c) repaired**: coverage share on a 50 m disc (state the radius
   in the manifest), threshold ≥ 80 %, then (e) restricted to those observers;
   report n per class.
4. **Street-point view**: rerun (e) on the Vidigal `svf_v2` street sample
   points (`svf_v2.paths.resolve_paths("vidigal")`, as WP-04 does) inside the
   scanned extent — these are the points the paper's ground-level claim is
   about; report the same table.
5. `g2_result_v3.json` + `report_v3.md` with ALL variants a–f side by side
   (copy a–d from phase 2's json by code) and the G2 floor label per variant;
   `tests/test_wp03_tls.py` extended: (a) shared obs_z gives Δ = 0 exactly on
   two identical surfaces even when one is lifted at the observer cell only;
   (b) the wall-share diagnostic on a synthetic pair.

## Gate (unpiped)

```
TMPDIR=/tmp python -m pytest tests/test_wp03_tls.py tests/test_wp02_horizon.py tests/test_p1_sky_resolution_consistency.py -q
python3 scripts/lint_p1_columns.py && python3 scripts/lint_p1_tokens.py
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

Commit last; paste hashes. Never the literal 145; no typed numbers; no edits
to params.yaml, solar_gates, or earlier run folders; poll background jobs
with `timeout 540 tail --pid=<PID> -f /dev/null`, never "wait for the
notification". If (e) and (f) still read r < 0.5 in every class, say so plainly
— the fallback in the plan of record (DTM + footprint decomposition with a
stated accuracy note) then applies and WP-03 closes as "TLS not usable as a
validity reference at this site"; do not iterate further.
