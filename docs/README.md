# `docs/`

Project documentation. The technical report is the **canonical**
deliverable; everything else is supporting material.

## Layout

```
docs/
├── README.md                            # This file
├── technical_report/                    # Canonical deliverable
│   ├── technical_report.md              # Source (markdown)
│   ├── technical_report.pdf             # Built artefact (rebuild with build_pdf.py)
│   ├── build_pdf.py                     # Pandoc + WeasyPrint pipeline
│   └── figures/                         # Figures referenced from the report (PNG)
│
├── methodology/                         # Per-feature methodology deep dives
│   ├── sky_exposure.md                  # Sky-exposure plane envelope (Rio + São Paulo rulesets)
│   ├── street_svf.md                    # Street-level SVF along centrelines
│   └── morphometric_indicators.md       # Formal definitions of the 25 indicators
│
├── onboarding/                          # Reader paths for first-time visitors
│   └── local_setup.md                   # Concentrated troubleshooting (GDAL, conda, GPU)
│
├── FAVELA_EXTRACTION_WORKFLOW.md        # GIS workflow for new-site building extraction
├── GPU_SVF_EXACT_VALIDATION.md          # GPU-vs-CPU SVF parity report (Phase 3.5)
└── cfd_sampling_overrides.yaml          # Documented coverage-gap downgrades for sampling-auditor
```

## Where to start

| If you want to … | Read |
|---|---|
| Get a fresh clone running locally | [`onboarding/local_setup.md`](onboarding/local_setup.md) — concentrated troubleshooting for GDAL, conda, GPU, common errors |
| Understand the methodology | [`technical_report/technical_report.md`](technical_report/technical_report.md) — full project description (start with the metadata block + §0 Glossary) |
| Reproduce a figure or number | TR §12 (Reproducibility) — every figure and table mapped to a producer command |
| Onboard a new site | Project root [`data/README.md`](../data/README.md) + [`FAVELA_EXTRACTION_WORKFLOW.md`](FAVELA_EXTRACTION_WORKFLOW.md) |
| Compute a specific indicator | [`methodology/morphometric_indicators.md`](methodology/morphometric_indicators.md) |
| Validate SVF against a benchmark | [`GPU_SVF_EXACT_VALIDATION.md`](GPU_SVF_EXACT_VALIDATION.md) (CPU↔GPU parity) and TR §10.3 (UMEP cross-val) |

## Conventions

- **The PDF must be rebuilt in the same commit as a `technical_report.md`
  edit**, so the rendered artefact never drifts from its source.
- **Methodology docs** are reference, not narrative. They define one
  thing each, with formulas and units.
