# Maré morphology brief v2 — spec (2026-09-16, night cycle)

Status: implementer brief for task MAREBRIEF2. Branch
`night-20260916/mare-brief-v2`. Laptop only; the build takes seconds.

The PI reviews the PDF tomorrow morning; their words: "concise, well
presented", audience = researchers writing research proposals on Maré. v1
(2026-09-15, `docs/briefs/mare/`, 8 pages) is a correct but plain render with
half-empty pages. The round-1 critic report `docs/critic/mare_brief_round1.md`
is the WORK LIST — mandatory first read; every rule in `docs/mare_brief_spec.md`
("Never" section included) still applies. Everything the critic listed under
"What already works" must survive.

## Deliverables

1. **Stylesheet** (in `build_brief.py`): choose one font pair from what
   weasyprint sees on this laptop — body Lato or Instrument Sans, display
   Fraunces or Bitstream Charter (`fc-list : family`); A4; ≤ 6 pages; no page
   less than about 60 % filled except the last; the key-numbers block as a
   two-column card; the data-inventory table styled to the text width; every
   figure sized to the text width; captions numbered by CSS counters on
   `figure`/`figcaption` — the spelled-out "Figure One/Two…" words go (pandoc
   renders an image paragraph with a caption as `<figure>`; the caption text
   stays authored in the template, WITHOUT a number); footer unchanged
   ("draft for PI review"); one accent colour.
2. **Figures** (`render_figures.py`): distributions as a 2×2 panel with tick
   labels legible at print size; a colour-blind-safe sequential ramp; legends
   that state the class edges; scale bar + north arrow; still NO coordinate
   axes/ticks and NO basemap. `figure_manifest.json` classes unchanged.
3. **Disclosure provisional defaults** (brisaverse tasks.json
   `_meta.provisional_default_policy`: reversible, no external dependency,
   PI tap overrides): (a) the T0–T5 morphotype labels and names do not appear
   in the default render — the composition passage describes "six recurring
   fabric clusters" by share and a plain description, no T-labels; a build
   flag `--named-morphotypes` restores the named variant so the PI's tap can
   flip it; record the default and the flag in `disclosure_sweep.md`.
   (b) "MorphoFavela" stays, still flagged for the PI. (c) `${pi_contact}`
   stays unfilled. (d) Footer stays "draft for PI review".
4. **Content**: edit prose only where a critic finding asks; every sentence
   must remain defensible from `mare_numbers.json`. NO new typed numbers — a
   number the brief needs and the JSON lacks is added to `collect_numbers.py`
   (read by code from the outputs of record), never typed.
5. **Tests** (`tests/test_mare_brief.py`): page bound ≤ 6; rendered markdown
   contains no spelled-out `Figure (One|Two|Three|Four|Five)`; the default
   build's rendered markdown contains no `\bT[0-5]\b`; tests a–e kept.

## Preview path (the worktree has no outputs/)

```
python3 docs/briefs/mare/build_brief.py --outputs-root /home/theo/SCL/SCR/MorphoFavela/outputs
python3 scripts/critic_sheet.py pages docs/briefs/mare/mare_morphology_brief.pdf /tmp/brief_pages --dpi 80
python3 scripts/critic_sheet.py sheet /tmp/brief_pages/sheet.png /tmp/brief_pages/page-*.png --cols 3
```

Read the sheet and every page PNG yourself and iterate at least twice before
you finish. Never ship on assumption.

## Gate (unpiped, one per line)

```
TMPDIR=/tmp python -m pytest tests/test_mare_brief.py -q
python3 docs/briefs/mare/build_brief.py --outputs-root /home/theo/SCL/SCR/MorphoFavela/outputs
TMPDIR=/tmp python -m pytest tests/ -q --ignore=tests/test_roughness.py
```

## Never

Everything in `docs/mare_brief_spec.md` §Never. Plus: no writes outside
`docs/briefs/mare/` and `tests/test_mare_brief.py`; `scripts/critic_sheet.py`
and `docs/critic/*` are read-only for you; no email, no contact details, no
health/TB content, no CFD results, no coordinates, no ranking of favelas, no
"deficit"/"formal"/"WHO"/"flow".

Final message: each round-1 finding → fixed / deferred + one line why; file
list; commit hashes; gate output tails; page count.
