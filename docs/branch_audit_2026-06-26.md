# Branch audit — 2026-06-26

Report only. No branch was deleted, merged, or force-pushed. This is a
snapshot to inform manual cleanup decisions.

Reference point: `main` (= `origin/main`), tip `bf44adb`
(`feat(cfd-ingest): synthetic/provenance fields + wind_direction enforcement (S4)`).

## Method

- `git branch --merged main` / `git branch -r --merged origin/main` for
  merge status.
- `git rev-list --left-right --count main...<branch>` for ahead/behind
  (read as `main-ahead  branch-ahead`).
- `git log -1` for each branch's last-commit date.

A branch is "fully merged" when it has **0 commits ahead of main** — its tip
is already reachable from `main`, so deleting it loses nothing.

## Local branches

| Branch | Tip | Last commit | Ahead of main | Merged? | Recommendation |
|---|---|---|---|---|---|
| `main` | `bf44adb` | 2026-06-?? | — | — | keep (current) |
| `track/morpho-signature` | `5645e16` | 2026-06-19 | 0 | yes | safe to delete — work is on main (see memory `project_morpho_signature_track.md`, merged 2026-06-19) |
| `track/roughness` | `1558ac2` | 2026-06-23 | 0 | yes | delete once the active roughness work has landed on main; tip is already reachable from main, so the local ref is redundant |

## Remote branches (`origin/*`)

| Branch | Tip | Last commit | Ahead of main | Merged? | Recommendation |
|---|---|---|---|---|---|
| `origin/main` | `bf44adb` | — | — | — | keep |
| `origin/track/morpho-signature` | `5645e16` | 2026-06-19 | 0 | yes | safe to delete on remote — fully merged |
| `origin/track/roughness` | `1558ac2` | 2026-06-23 | 0 | yes | keep until roughness track is declared complete; fully reachable from main, delete after |
| `origin/track/morphotope` | `7a7d57f` | 2026-06-24 | 0 | yes | fully merged; delete once morphotope stream confirms no further pushes are planned |
| `origin/track/viz` | `f918a58` | 2026-06-23 | 0 | yes | fully merged; delete after viz stream confirms completion |
| `origin/feature/publication-report` | `4e5f1d4` | 2026-03-27 | **5** (also 283 behind) | **no** | **stale (3 months).** Has 5 unique commits not on main. Before deleting, confirm the report content improvements were superseded by `docs/technical_report/`; if so, delete. Otherwise cherry-pick the 5 commits first. |
| `origin/feature/pyviewfactor` | `805cacf` | 2026-03-12 | **1** (also 325 behind) | **no** | **stale (3 months).** 1 unique commit (experimental PyViewFactor SVF backend). PyViewFactor is now a declared runtime dep and used in `src/svf_v2/compute.py`, so the experiment likely landed in a different form. Confirm the experimental commit is obsolete, then delete. |

## Summary

- **Fully merged, safe to prune** (0 ahead of main): `track/morpho-signature`,
  `track/roughness`, `origin/track/morpho-signature`,
  `origin/track/roughness`, `origin/track/morphotope`, `origin/track/viz`.
  The `track/*` branches are active work-streams; prune each only after its
  stream is declared done.
- **Stale with unique commits, review before pruning**:
  `origin/feature/publication-report` (5 ahead),
  `origin/feature/pyviewfactor` (1 ahead). Neither is merged; both predate the
  current report/SVF architecture and are candidates for deletion *after* a
  human confirms the unique commits are obsolete.

No action taken — this document is advisory.
