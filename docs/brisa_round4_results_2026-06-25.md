# MorphoFavela → brisaverse — ROUND 4/5 results (2026-06-25)

Deliverables for the round-4 handoff (`morphofavela_handoff_2026-06-25_round4.md`)
and the round-5 extension embedded in `morphofavela_sync_2026-06-25.md`
("Commissioned from MF — round-5"). All produced this session on the
dissolved-λf canonical baseline. Source of every number below:
`scripts/brisa_ventilation/09_regime_taxonomy.py` →
`outputs/brisa_ventilation_fix/taxonomy_regime.json` +
`methods_morphometric_row.csv` (both regenerable; outputs/ is gitignored).

Commits: `1e5b67c` (keystone), `f999b0b` (fig04), `4ec9898` (lambda_f_regime),
`504c062` (morphotope composite), `1cdb3ac` (TR figure + PDF).

---

## R5 · Regime-stratified four-state taxonomy — CANONICAL, supersedes λf>2.75

Ventilation axis ≡ skimming flow regime, **dissolved λf > 0.65** (Grimmond &
Oke 1999). Sunlight axis unchanged (winter direct sun < 2 h). **Denominator =
n_classified = built ∩ sun-known ∩ λf-known = 56,631** (the canonical
street-solar denominator via nearest-3-within-25 m fallback — NOT the 64,389
raw built mask, NOT the 24,862 direct-support subset).

### Pooled + terrain (primary, λf > 0.65)

| group | n | adequate | sunlight | ventilation | compound |
|---|---|---|---|---|---|
| **Pooled** | 56,631 | 26.9 | 6.5 | 27.0 | **39.6** |
| Hillside | 26,573 | 26.8 | 10.3 | 20.8 | **42.2** |
| Flatland | 30,058 | 27.0 | 3.2 | 32.5 | **37.3** |

### Per-site (primary, λf > 0.65)

| site | n | adeq | sun | vent | comp |
|---|---|---|---|---|---|
| Vidigal | 2,548 | 20.7 | 13.0 | 22.8 | 43.4 |
| Rocinha | 7,637 | 11.9 | 7.6 | 13.7 | 66.7 |
| Complexo do Alemão | 16,388 | 34.6 | 11.1 | 23.7 | 30.6 |
| Rio das Pedras | 5,710 | 9.9 | 3.6 | 24.0 | 62.4 |
| Maré | 24,348 | 31.0 | 3.1 | 34.5 | 31.4 |

### ⚠ The "typology inversion" did NOT survive — it REVERSED

The deprecated λf>2.75 pre-screen gave **flatland 21.6 % > hillside 14.8 %**
compound. Regime-stratified gives the opposite: **hillside 42.2 % > flatland
37.3 %** (gap +4.9 pp). The λf-independent **H/W > 0.65 (Oke 1988) cross-check
agrees**: hillside 48.1 % > flatland 39.5 %. The old inversion finding must be
retired from decks + manuscripts; the corrected ordering is hillside-heavier
compound, robust across both the λf and H/W criteria.

### H/W cross-check (own sparser denominator n_classified_hw ≈ 55,768)

| group | n | adeq | sun | vent | comp |
|---|---|---|---|---|---|
| Pooled | 55,768 | 27.8 | 3.0 | 25.6 | 43.5 |
| Hillside | 26,438 | 21.1 | 4.4 | 26.5 | 48.1 |
| Flatland | 29,330 | 34.0 | 1.8 | 24.8 | 39.5 |

Compound clustering re-run on the regime mask (`compound_spatial_clustering.json`):
Moran's I **0.47–0.70**, all p=0.001; the contiguity finding holds *stronger*
than under the old mask (larger compound fraction → larger contiguous patches).

## R5 · Dissolved per-site Methods-table morphometric row

Medians + means over the classified denominator (`methods_morphometric_row.csv`):

| site | SVF med | λp med | **dissolved λf med** | H_mean med | σH med | slope med |
|---|---|---|---|---|---|---|
| Vidigal | 0.44 | 0.68 | 0.84 | — | — | — |
| Rocinha | 0.21 | 0.82 | 1.14 | — | — | — |
| Complexo do Alemão | 0.35 | 0.70 | 0.69 | — | — | — |
| Rio das Pedras | 0.15 | 0.93 | 1.17 | — | — | — |
| Maré | 0.35 | 0.82 | 0.83 | — | — | — |

(Full medians + means for SVF, λp, λf, H_mean, σH, slope are all in the CSV;
λf medians here are over the classified subset, ~equal to the full-mask values
in `lambda_f_regime.json`: Vid 0.83 · Roc 1.12 · Ale 0.68 · RdP 1.15 · Maré 0.81.)

## Figures re-exported (brisaverse deck assets)

- `lambda_f_regime.png` — R4-1 two-panel: Panel A per-site regime maps
  (isolated/wake/skimming, dissolved λf, CVD-safe luminance-ordered),
  Panel B regime-share strip (per-site + pooled, canonical shares). `4ec9898`.
- `morphotope_maps_repartition.png` — R4-2 combined k=3 tissue maps + per-site
  repartition bars (Compact Hillside / Mixed Dense / Saturated Flatland). `504c062`.
- `fig04_diagnostic_taxonomy.png` — R4-3 re-rendered on the regime axis (λf>0.65),
  nominal four-state, CFD-pending caption, H/W cross-check note. `f999b0b`.

All honour the round-3 §0 text-fit hard gate.

## SVF "6×" claim — VERIFIED, qualified (sync §4 action)

On the **pooled RF** (`rf_predictor_stats.json`, `pooled_rf.permutation_importance`):
SVF = 0.283, southness = 0.059, slope = 0.042. So SVF carries **~4.8× the next
feature (southness)** and ~6.7× slope. The "~6×" headline is defensible **only**
if it referenced SVF-vs-slope (or a mean-of-rest baseline), not the literal next
feature. The sync's "≈1.8×" used the *per-fold LOSO* permutation importances
(SVF ≈0.137–0.16), a different quantity from the pooled aggregate. Recommended
deck/manuscript wording: **"SVF's permutation importance is ~5× the next-ranked
geometric feature in the pooled model."** Qualitative "SVF dominates" holds.

## Not done (out of scope this round, per handoff)

Solar figs (fig06–08), predictor fig (fig05) beyond the SVF text, roughness
envelope — untouched. fig04/lambda_f_regime are manuscript/deck assets, not TR
figures; the TR's own taxonomy is the CFD `U_mean<1.0 m/s` proposition
(`fig_0_4_diagnostic.png`) and its §4.5 regime paragraph is already on dissolved λf.
