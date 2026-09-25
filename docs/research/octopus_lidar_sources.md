# LiDAR / point-cloud sources for the Octopus–Maré paper

Researched 2026-09-24 via the PI's Google Drive (read-only search, `search_files` /
`get_file_metadata`). No files downloaded — sizes and paths only. Scope per the
OCTOPUS INPUTS task: (1) 2024 airborne LiDAR/DSM of Maré, (2) 2026 terrestrial OM2
scan (TLS/point cloud).

## 0. Headline finding

**Neither dataset is actually sitting in Drive as a usable point cloud or raster
right now.** The 2024 airborne LiDAR has a fully scaffolded folder tree
(`LiDAR_DSM_DTM/2024/{Maré,Rio das Pedras,Rocinha_Vidigal}/{DSM,DTM}`, built
2026-01-13 → 01-23 by `cmoroz@mit.edu`) but every leaf folder is **empty** — no
`.las`/`.laz`/`.tif` ever landed inside it, for any of the three sites. The 2026
terrestrial OM2 scan has **no candidate at all**: no `.e57`/`.ply`/`.pcd`/`.xyz`,
no folder titled "terrestrial"/"TLS"/"scan" that isn't unrelated JS library code.
The airborne data most likely lives on a local disk or external drive belonging to
Carlo Moroz's team, not yet pushed to the shared Drive; the terrestrial scan may
not have been processed into a deliverable file yet, or lives outside Drive
entirely (device SD card, a different cloud, etc.).

This means: **the rclone pull below will currently return nothing for the point
clouds themselves.** What it's for is (a) confirming that the moment someone
uploads into the scaffolded folders, we can pull without re-searching, and (b)
grabbing the metadata/log files that already do exist (§2) so tree-shade / SVF
work has *something* to start from while the real rasters are chased down.

## 1. Top 5 candidates (ranked)

| # | Title | id | Size | Owner | Modified | What it would unblock |
|---|---|---|---|---|---|---|
| 1 | `LiDAR_DSM_DTM/2024/Maré/DSM/tiles/` (folder, **empty**) | `1i7dtLBl-4RzKmdOnvTnD1oxAMYtH4kjv` | 0 (no children) | cmoroz@mit.edu | scaffold created 2026-01-19 | Would be canopy-height / building-height DSM for Maré if populated — tree shade + height-change vs. 2019 IPP cadaster |
| 2 | `LiDAR_DSM_DTM/2024/Maré/DTM/` (folder, **empty**) | `1ZbRSwJ-BuRLGA_Z4dqRMattZo6QoQr4_` | 0 (no children) | cmoroz@mit.edu | scaffold created 2026-01-13 | Bare-earth terrain model — needed to derive DSM-DTM = height above ground for canopy/building separation |
| 3 | `LAS/` tile grid, e.g. `311C43A.las`, `310B14D.las`, … (~30+ tiles seen, more paginated) | parent `1r0XINv550GCI8SUOjlDBh0JIaH5Gxg3g` | 36 KB–7.6 MB per tile | unlisted (shared, not `cmoroz@`/`thermann.ai@`) | 2025-05-19/20 | **Unconfirmed relevance** — tile-coded LAS grid nested under an `MDT` (Modelo Digital do Terreno) folder several levels below an unidentified root; could be a citywide Rio basemap LiDAR product that happens to cover Maré, but nothing in the path names Maré or Octopus explicitly. Needs a direct ask to whoever owns the parent chain before treating as a Maré source. |
| 4 | `ORTOFOTO` folders + `Ortofoto 2011.jpg` (28.8 MB) | e.g. `15uuJtXYi9bBRQk98oIObk1eXKkMifC1u` | folder / 28.8 MB | `lab.fotogrametria@eng.uerj.br` | 2023-04 (folder), imagery dated 2011 | Orthophoto only, not LiDAR, and 2011 not 2024 — low priority, but the UERJ photogrammetry lab (`lab.fotogrametria@eng.uerj.br`) is a plausible institutional source to ask directly for a 2024 airborne product if MIT's own copy can't be found |
| 5 | `octopus_IDs_locations` (spreadsheet, device ID ↔ install location) | `1w8VU5_xFeQC9hjOTZWEqFW6FHvwBSO5xgTtFfDNb9Jg` | 5.3 KB | `moras@mit.edu` (shared) | 2026-09-17 | Not LiDAR, but gives device geolocations — needed to georeference OM2 sensor readings against whatever DSM/point-cloud eventually arrives |

Folders checked and confirmed **empty** (not worth re-polling until someone
uploads): `LiDAR_DSM_DTM/2024/Maré/DSM/`, `…/DSM/tiles/`, `…/DTM/`,
`…/Rio das Pedras/DSM/`, `…/Rio das Pedras/LAS_DSM/`, `…/Rocinha_Vidigal/DSM/`,
`…/Rocinha_Vidigal/LAS_DSM/`.

## 2. What *is* in Drive under Octopus/OM2 (not LiDAR, but adjacent)

- `Octopus/OM_{1,2,3,4}_inferred_route.json` + `om_routes.gpkg` (folder
  `1oEzF6aCrsCIfUfu0Esv31Vya0QtWsSFQ`, owner `thermann.ai@gmail.com`, 2026-09-16) —
  inferred device routes, already geospatial (gpkg). Useful once a DSM exists to
  clip along the route.
- `OM2/*.CSV` (parent `10cP66MfsrlMGYbYieazCHuZcATkgqIeb`, owner `cmoroz@mit.edu`,
  uploaded 2026-07-06) — dozens of per-session device CSVs named by timestamp
  (e.g. `26062008.CSV`). **All have `modifiedTime: 2000-01-01T04:00:00Z`** — a
  device-clock artifact, not a real modification date. This is direct physical
  evidence for the clock-offset question in the team message (§3 below): the
  device's own internal clock was reset to the epoch default at some point, which
  is consistent with a no-GPS-fix row falling back to an unset/UTC-epoch clock
  rather than a live Rio-time clock.
- `04_Octopus_Maré/_data collection/Zenodo_release/fixed_data/*.csv` (parent
  `1O6ILE6cBiXyq8_zMvXWRc6IgCSX9czoH`, owner `cmoroz@mit.edu`, **created
  2026-09-23**, i.e. yesterday) — device-and-date-labelled CSVs (`I_1_20260324_…`,
  `O_3_20260308_…`, `O_4_…`) for devices I1/I3/I4/O3/O4, spanning
  2025-12 → 2026-04. This looks like exactly the kind of "raw OM2 CSVs, all
  devices and dates" the team message is asking Vincent/Jingxue for — but it's
  already on Carlo's Drive and unclear whether it's the full/raw set or a
  filtered "fixed_data" cut. Worth checking with Carlo before assuming Vincent
  and Jingxue's answer duplicates this.
- `calibration_OM2.png` (`10n0pjnwnihTYUmsabNl8opXkDtzn13k2`, owner
  `cmoroz@mit.edu`, 2026-09-23) — same-day calibration figure, adjacent to the
  fixed_data drop above.

## 3. rclone command for the PI (before any large download)

Read-only scope, so this can't accidentally modify or delete anything in Drive:

```bash
rclone config create gdrive drive scope=drive.readonly
```

Then, once someone populates the scaffolded folders (§0), pull just the Maré
airborne LiDAR tree:

```bash
rclone copy "gdrive:LiDAR_DSM_DTM/2024/Maré" ./data/lidar_2024_mare --progress
```

(`gdrive:` here assumes the remote's root is the PI's My Drive; adjust the path
if the remote is scoped to a shared drive instead — `rclone lsd gdrive:` first to
confirm.) Nothing terrestrial (2026 OM2/TLS) to pull yet — no candidate file or
folder found under that name anywhere in the account.

## 4b. The tile-coded LAS grid (2026-09-25 update)

Walked the parent chain of `1r0XINv550GCI8SUOjlDBh0JIaH5Gxg3g` (metadata
only — no file content downloaded, per instruction; every `.las`/`.tif`
here is 36 KB–56 MB, well above "small"):

```
LIDAR 2019/                              id 1cn_1jPhsGIwumhJtd81u4lhE4U4wvz8y
  created 2024-11-01, sharedWithMeTime 2025-12-02 (shared TO the PI;
  owner not surfaced by the connector for this folder — likely a
  shared-drive item, not one of cmoroz@/thermann.ai@'s own uploads)
  └── MDT/                               id 1ulCdBKt8qdVW4__g9cjzACtnvVvXNd--
        ("Modelo Digital do Terreno" — Portuguese for Digital Terrain Model)
        ├── LAS/                         id 1r0XINv550GCI8SUOjlDBh0JIaH5Gxg3g
        │     311C43A.las (455 KB, created 2025-05-19), 310B14D.las, … —
        │     ~30+ tiles seen, more paginated; code format `<3-digit><letter><2-digit><letter>`
        └── Geotiff/                     id 1CtfFTLKaqe_krgA0VdBFZo0QitAYo62z
              311C43A.tif (56 MB, modifiedTime 2020-02-19) + .tfw
              (world file) — SAME tile code as the LAS, different
              product/date; a sibling folder (id 1uef1hZ8PCoOFKpq1UN5pGVqoxxoy_rY_)
              holds 311C43A_3.tif / 311C43A_4.tif (modifiedTime 2020-07-17)
```

**What this answers, and what stays UNVERIFIED:**

- **Survey/year**: the top folder is literally named `LIDAR 2019/`. That is
  real evidence the *survey* is 2019-vintage — but the files inside it
  have THREE different `modifiedTime`s (`.las` 2025-05-19/20, `.tif`
  2020-02-19, `_3/_4.tif` 2020-07-17), which are re-upload/re-export
  dates, not acquisition dates. **UNVERIFIED**: whether this is the SAME
  2019 survey the package's own `buildings_mare.shp`/DTM already cite
  (package README's Sources table: "2019 airborne survey") is plausible
  given the matching year and Rio/IPP-style tile coding, but nothing in
  the folder names or metadata says so explicitly — needs a direct
  confirmation, not an inference from a shared folder name alone.
- **CRS**: **UNVERIFIED**. A `.las` file's CRS lives in its own binary
  header (and the `.tfw` gives the `.tif`'s affine transform, not its
  CRS) — reading either requires downloading the file, which the task
  scope excludes ("metadata only, never download large files"). The
  `.tif`'s CRS is very likely SIRGAS 2000/UTM 23S (EPSG:31983, matching
  every other Maré layer in this repo), but that is an inference from
  local convention, not a read header.
- **Which tiles cover OM2**: **UNVERIFIED**. The tile code
  `311C43A`-style naming (`<3 digits><letter><2 digits><letter>`) matches
  the general shape of Rio de Janeiro's municipal cartographic base-map
  sheet index (IPP's Base Cartográfica Digital uses a similar
  hierarchical zone/quadrant scheme at 1:2000/1:10000), which is
  consistent with — but does not confirm — this being an IPP-sourced
  citywide product that happens to cover Maré among many other tiles. No
  tile-index/grid-reference document was found alongside the LAS/Geotiff
  folders (a `fullText` search for "articulação"/"malha"/"index" inside
  this Drive account returned nothing under this folder tree). Confirming
  coverage needs either (a) the PI/Carlo naming the source survey and its
  published tile index, or (b) downloading one tile's header (a small
  read, not the whole point cloud) to check its bounding box against
  Maré's — neither was done here, per the "metadata only" scope.

**Recommendation**: ask directly (folded into the same message that asks
about the 2024 airborne LiDAR, §4 below) whether `LIDAR 2019/MDT/` is a
citywide IPP product, and if so, whether its tile index or a coverage
shapefile exists so tile-to-Maré coverage can be confirmed without
downloading the ~30+ candidate tiles one at a time.

## 5. Zenodo_release/fixed_data pull (2026-09-25) — a related, separate finding

Downloaded (not just searched) a 5-file, one-per-device pilot from
`04_Octopus_Maré/_data collection/Zenodo_release/fixed_data/` (§0/§2
above; full manifest: `data/maré/octopus/csv/manifest.json`). All 5 share
`Timestamp,Temperature,Humidity,PM1.0,PM2.5,PM2.5_cal,PM4.0,PM10.0` — **no
Latitude/Longitude column at all**. This is a fixed-site indoor/outdoor
logger schema (device codes I_1/I_3/I_4/O_3/O_4), not the OM2 GPS-track
schema `src/om_package/shade.py`'s `OCTOPUS_JOIN_EXAMPLE` documents (which
needs Latitude/Longitude to pick the nearest OM2 point and to drop the
0/0 no-fix sentinel). **UNVERIFIED**: whether I_1/I_3/I_4/O_3/O_4 ARE the
OM2 walking-route device under a different naming convention (i.e. this
folder's "fixed_data" means "the corrected/fixed-up data", and the device
was carried, not stationary), or a genuinely separate fixed-site
deployment. This is exactly the ambiguity §2 already flagged before any
file was opened; opening 5 files did not resolve it, only sharpened the
question — asked of Carlo in the team message (communication/
team_message_draft.md).

## 6. Recommended next action

Before spending PI time on rclone: ask Carlo Moroz directly whether the 2024
airborne LiDAR for Maré exists anywhere reachable (the empty scaffold strongly
suggests it's still local to his machine or an external drive), and whether a
2026 terrestrial OM2/TLS scan has been captured and processed at all yet, or is
still raw device output. That one question resolves the ambiguity faster than
continuing to poll Drive.
