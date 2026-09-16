# VENTAXIS — a C′ definition of record for P1's second axis

Opened 2026-09-16 (cycle 3). Owner: agent. Repo: MorphoFavela.
Verify gate: `.claude/verify-cmd` (pytest + `emit_cockpit.py --check` + both P1 lints).

## The problem, as measured

P1 v2's second axis is the per-cell **count (0–3) of independent geometry
constraints**. Its numbers are final and ledgered:

    runs/wp07_ledger_20260915T184619Z/ledger.json
      -> 25 entries `wp06.<site>.{n,share_n0..share_n3}`
      -> status "final", release_class "publishable-candidate"
      -> source: runs/wp06_geometry_20260915T052604Z/summary.json

Its **definition** is not. The definition string those numbers inherit lives in

    outputs/paper_figures/ventilation_index.json        (dated 2026-06-28)

which predates the C′ reframe (2026-09-08). That file names the vertical
constraint with the flow-regime word that is the **ninth entry of
`scripts/lint_p1_tokens.py::BANNED_TOKENS`**, and describes itself as
"pre-CFD" / "CFD-gated". Do not type either word into anything you write;
refer to the threshold by its geometry, exactly as the lint's own docstring
does: `lambda_f_mean >= 0.65`, Oke (1988).

`python3 scripts/lint_p1_tokens.py` returns 0 today. That is not a pass on
this file — `outputs/paper_figures/**` is outside all five
`P1_SOURCE_GLOBS`/`P1_ARTIFACT_GLOBS`. Verified this pass:
`grep -c` on that file returns 3 hits.

`src/brisa_solar/wp06_geometry.py` IS in scope and IS clean: the producing
code already speaks geometry. This is a definition-and-rail gap, not a
numbers gap. **No recompute of the shares is authorised or wanted** — they
are final and any change to them is a ledger event, not a spec fix.

## Goal

By the end of this task the PI can write WP-07 prose about the second axis
from a document that (a) defines each of the three constraints in geometry
terms with its literature citation, (b) is inside the lint's scope so the
vocabulary rail actually guards it, and (c) is pointed at by the ledger.

## Scope

**1. `docs/ventaxis_canonical.md` — the definition of record.** New file.
Contents, all read from code/data, none typed from memory:

  - The three constraints, each with: the exact predicate as implemented in
    `src/brisa_solar/wp06_geometry.py` (quote the line), the numeric
    threshold read from the code or from `config/params.yaml`, and its
    source. The vertical threshold's source is Oke (1988) — cite it as a
    geometry threshold on frontal-area density, never by regime name.
  - The pooled depth median: read it, do not type it. It is in
    `runs/wp06_geometry_20260915T052604Z/summary.json#/depth_median_m`.
  - What the count IS: a checklist count of independent geometry
    constraints. What it IS NOT: air exchange, adequacy, or any simulated
    quantity — say this in C′ vocabulary (the honest phrasing is that these
    are geometric conditions associated with reduced exchange in the
    literature, and that P1 makes no claim about exchange itself; the
    quantitative version of that claim is P3's, referenced forward).
  - A provenance note: this document supersedes the definition strings in
    `outputs/paper_figures/ventilation_index.json` (2026-06-28) for every
    P1 purpose. That file is a June artifact of E2 and is NOT to be edited
    or deleted — it is another output's input (`scripts/health/compound_exposure.py`
    reads it; check before you touch anything).

**2. Widen the lint rail.** Add `docs/ventaxis_canonical.md` — and, more
generally, a glob covering P1 definition docs — to `P1_SOURCE_GLOBS` in
`scripts/lint_p1_tokens.py`. Pick the narrowest glob that covers the new
file and any sibling P1 canonical doc; do NOT widen to all of `docs/**`
(it would catch the C′ plan's own deliberate P3 references and the WP specs,
which legitimately name the deferred work). Prove the widened rail can go
red: plant the token in the new doc, confirm rc 1, remove it, confirm rc 0.
Paste both exit codes.

**3. Re-ledger the definition pointer.** The ledger's 25 `wp06.*` entries
should carry a pointer to the definition of record. Extend
`src/brisa_solar/wp07_ledger.py` so `_meta` gains a
`definition_of_record` map naming `docs/ventaxis_canonical.md` for the
`wp06.*` family, and regenerate the ledger into a NEW run directory
(`runs/wp07_ledger_<UTC>/`). The 371 values must be **bit-identical** to
`runs/wp07_ledger_20260915T184619Z/ledger.json` — write a check that
compares the two entry dicts on `value` and fails loudly on any difference,
and paste its output. A changed value is a stop-and-report, not a fix.

**4. Test.** One test asserting the new doc exists, is inside the lint's
globs (assert by calling the lint's own glob expansion, not by re-typing the
glob), and that the ledger `_meta.definition_of_record` resolves to a file
that exists.

## Explicitly OUT

- Recomputing any share, count or threshold.
- Editing `outputs/paper_figures/ventilation_index.json` or anything under
  `scripts/health/`.
- Any prose destined for the manuscript (paper voice is the PI's, always-ask).
- Widening the lint to `docs/**`.
- Touching the façade surface, G2, or any /ops card.

## Never

- Never type a number that exists in a file — read it by code.
- Never write the banned regime word, "CFD", or the Greek exchange-time
  symbol into any new file, including this task's commit messages.
- Never say "WHO" for the 2 h floor — it is the Athens Charter (1943),
  Point 26.
