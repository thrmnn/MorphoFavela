# MorphoFavela handoff — canonical figure + analysis tasks (2026-06-24)

For the agent that owns the **MorphoFavela** repo. brisaverse is read-only on MorphoFavela, so these must run in the canonical `outputs/paper_figures/` pipeline (single source of truth). brisaverse has STAGED local previews/specs where noted — port them into the canonical scripts, then export the PNGs so brisaverse can promote them into `shared/figures/`. Full per-figure rationale: brisaverse `papers/p1-nature-cities/communication/figure_audit_2026-06-24.md`.

**Data-provenance rule (must hold in every figure/caption/data-availability):** inputs are NOT open — **ALS/LiDAR heights = MIT** (TLS of a Vidigal subset via **SondoTecnica**); **cadaster footprints = IPP** (municipal, not redistributable). The **pipeline is open**, the data is not.

## ⭐ TOP PRIORITY — figure-rework feedback (PI, 2026-06-24)
These supersede ordering below; they are the figures the dashboard "key results" must carry well.
- **fig04 — four-state diagnostic map: REWORK + ADD COLOURS.** The current map reads flat. Adopt a **severity-ordered 4-colour palette** (adequate → sunlight-deprived → ventilation-deprived → compound) with the four states unambiguously distinct and a clear legend; move per-panel inline % off the map into a side table. This is the dashboard HEADLINE — it must be legible at a glance.
- **fig03 — ventilation × solar field: NOT READABLE, re-render.** The two-axis field render is unreadable as-is. Re-plot for legibility: per-site/per-panel colour with explicit, separate colour scales for each axis; shade the sub-threshold mass; ensure the structure (bimodality / mass-at-zero) the caption claims is actually visible.
- **fig05 — cell-level predictor: AUDIT the parameters shown.** PI flags that *the parameters shown don't really make sense*. Before recoloring (task 1 below), **audit which predictors are displayed and how**: confirm the feature set, signs, and importances against `rf_predictor_stats.json`; drop/rename any that are mislabelled or not interpretable; only then apply the sign/family recolor.
- **solar-access cross-site + parameter-impact: KEEP ITERATING.** Continue refining `fig08_solar_cross_site` (clarity of the aspect dissociation) and the parameter-sensitivity figures (how the shortfall responds to each geometric driver) — these are an active iteration target, not frozen.

Export reworked PNGs so brisaverse can promote them into `shared/figures/`; the dashboard `KEY_FIGS` already flags fig04/fig03/fig05 as REWORKING and will pick up the new renders.

## Prioritized tasks
1. **fig05 recolor (canonical sync).** Port the brisaverse-staged recolor into `fig05_predictors.py`: Panel C coefficient markers **sign-colored** (diverging — negative β one hue, positive β another, saturation ∝ |β|, with a sign legend); **one shared feature-family palette** across A/B/C (SVF/openness, terrain-aspect, density each one hue everywhere); Panel A bars **grouped by family**; Panel D kept as the honest pending-CFD placeholder. Target look + reference implementation: brisaverse `shared/figures/fig05_predictors_v2.png` + `shared/figures/scripts/fig05_recolor.py`. No new analysis — sign/group/importance are in `rf_predictor_stats.json`/`rf_pd_curves.json`.

2. **fig01 composite (highest-impact gap).** Panel A (the geometry→adequacy pipeline schematic) is built in brisaverse (`shared/figures/fig01_panelA.png`, header "INPUTS / not openly redistributable"). Composite it as **Panel A** with **Panel B** (regional site map) + **Panel C** (3-D model excerpts) into the canonical `fig01_study_sites.png`. Reconcile the "C. do Alemão (mixed vs hillside)" label across figures.

3. **fig03 C/D re-render.** Re-plot the ridge panels as **filled, per-site colored densities** with the sub-threshold mass shaded, so the bimodality (Maré) / mass-at-zero (Rocinha, Vidigal) the caption claims is visible (currently flat grey ribbons). Headline figure in the PH framing.

4. **fig04 cleanup + spatial-clustering statistic (highest-value analytic add).** Move the per-panel inline % off the maps into a clean side table; adopt a **severity-ordered 4-colour** palette (adequate→compound). Compute **Moran's I / join-count + the contiguous-patch-size distribution** for the compound state and add as an fig04 inset or ED panel — converts "tend to occur in contiguous groups" from assertion (currently `\TODO` in both manuscripts) to evidence.

5. **Full-sample imputed LOSO.** Re-run the LOSO RF on the **full 56,631-cell sample** (imputed aspect) to bound the 0.90-AUC complete-case caveat (subset 56% vs full 46% prevalence) so the headline AUC stands less hedged. Export the full-sample AUC + range.

6. **λf canonical 0–1.5 recompute.** Recompute frontal-area density on the **Grimmond–Oke 0–1.5 scale** (vs the current 0–10 field) for the calibration sites; report the skimming onset relative to ~0.35. Lifts E2 from PARTIAL toward SOLID and removes the λf-threshold inconsistency (P1 0.4–0.5 vs E2 ~0.35).

7. **Ray-caster cross-validation.** Validate the in-house ray-caster against **Radiance/SOLWEIG** on a cell subset (kills P2's single-point-of-failure attack; hardens the delivered solar axis of P1) + a **vegetation-sensitivity bound** on south-slope sun.

8. **Morphotope/signature figures for P4 + E2** (see the coherent-unification note). Provide print-quality `signature/figures_v2/{fingerprint_heatmap, dendrogram, composition_by_site}.png` and a clean **morphotype × site fingerprint heatmap** showing the five favelas share a recurrent, classifiable fabric — destined for P4 (tool capability) + E2 (informal-typology evidence), NOT the flagship main set.

## Handoff
Export produced PNGs to a path brisaverse can read (or push to a shared branch); brisaverse promotes them into `shared/figures/` and updates captions. The brisaverse-staged `fig01_panelA.png` + `fig05_predictors_v2.png` are previews of intent, not the canonical outputs.
