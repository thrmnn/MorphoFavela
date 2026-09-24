"""SITETERR (2026-09-24): config/sites.yaml + src/sites/territory.py is now
the one declared, reviewable territory per campaign site — replacing six
places a site's boundary used to be typed (see config/sites.yaml's header).

Needs the main checkout's data/ (Favelas_Limit_2019.shp, each site's raw
boundary/buildings, data/maré/neighbourhoods.gpkg): a plain worktree
checkout has none of these, so the whole module skips cleanly there, same
pattern as tests/test_mare_study_area.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.affinity import rotate as _shapely_rotate
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sites.territory import (  # noqa: E402
    BETWEEN_SUBUNITS_LABEL,
    label_subunits,
    load_sites_config,
    load_territory,
    rotation_deg,
    within_mask,
)

MAIN_ROOT = Path("/home/theo/SCL/SCR/MorphoFavela")
FAVELAS_LIMIT = MAIN_ROOT / "data" / "RJ" / "Favelas_Limit_2019.shp"
DATA_PRESENT = FAVELAS_LIMIT.exists()
pytestmark = pytest.mark.skipif(
    not DATA_PRESENT, reason="main checkout data/ (Favelas_Limit_2019.shp etc.) absent on this checkout")

SITES = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]


@pytest.fixture(scope="module")
def registry():
    return load_sites_config()


@pytest.fixture(scope="module", params=SITES)
def territory(request):
    return load_territory(request.param, root=MAIN_ROOT)


def test_registry_has_exactly_the_five_campaign_sites(registry):
    assert set(registry) == set(SITES)


def test_registry_loads_for_every_site(territory):
    assert territory.data_extent.area > 0
    assert territory.study_area.area > 0


@pytest.mark.parametrize("site", ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras"])
def test_citywide_kind_sites_study_area_is_a_subset_of_data_extent(site):
    """The four non-Maré sites' study_area.kind is citywide_polygons, built
    directly from the same declared boundary shapefile the data extent
    comes from — the two must (near-)coincide. Kept separate from Maré's
    own subset check below: Maré's promoted study area (2026-09-24, PI
    ruling) is an independently digitized IPP source and is NOT a subset of
    the data extent any more (see test_mare_study_area_extends_slightly_
    beyond_the_data_extent)."""
    t = load_territory(site, root=MAIN_ROOT)
    assert t.study_area.difference(t.data_extent).area < 1.0


def test_mare_study_area_extends_slightly_beyond_the_data_extent():
    """Maré's promoted study area (the IPP Territórios Sociais outline) is
    an independently digitized source from the bairro polygon used as the
    data extent (data/maré/raw/mare_boundary.shp) — the two boundaries were
    never meant to be pixel-identical, so a small sliver of the outline
    falls outside the data extent. Documented here with a real number (not
    "basically zero"): must stay a small fraction of the outline's own
    area, never balloon into a real definitional drift."""
    t = load_territory("maré", root=MAIN_ROOT)
    outside = t.study_area.difference(t.data_extent).area
    assert 0 < outside < 20_000  # m² — observed ~13,400 m² at promotion time
    assert outside / t.study_area.area < 0.01


def test_mare_alias_resolves_to_same_registry_entry():
    t_accented = load_territory("maré", root=MAIN_ROOT)
    t_ascii = load_territory("mare", root=MAIN_ROOT)
    assert t_accented.study_area.equals_exact(t_ascii.study_area, 1e-6)
    assert t_accented.data_extent.equals_exact(t_ascii.data_extent, 1e-6)


# --- citywide match must be identical to match_favela_group's own output ---

@pytest.fixture(scope="module")
def favelas():
    return gpd.read_file(FAVELAS_LIMIT)


@pytest.mark.parametrize("site,target", [
    ("vidigal", "Vidigal"), ("rocinha", "Rocinha"),
    ("complexo_do_alemao", "Complexo do Alemão"),
    ("riodaspedras", "Rio das Pedras"), ("maré", "Maré"),
])
def test_citywide_polygons_identical_to_match_favela_group(favelas, site, target):
    from src.brisa_solar.wp05_full import match_favela_group
    expected, expected_method = match_favela_group(favelas, target)
    t = load_territory(site, root=MAIN_ROOT)
    assert sorted(t.citywide["cod_favela"].tolist()) == sorted(expected["cod_favela"].tolist())
    assert t.citywide_method == expected_method


def test_wp07_ledger_favela_names_all_resolve_through_the_registry():
    """The WP-07 ledger's FAVELAS mapping (slug -> IPP display name) must
    match a citywide_rule the registry actually carries, for every site —
    a stale registry (renamed favela, changed slug) would fail loudly here
    instead of silently diverging from the ledger."""
    from src.brisa_solar.wp07_ledger import FAVELAS
    cfg = load_sites_config()
    for slug, display in FAVELAS.items():
        site_key = "maré" if slug == "mare" else slug
        assert site_key in cfg, f"WP-07 ledger slug {slug!r} has no config/sites.yaml entry"
        assert cfg[site_key]["citywide_rule"]["target"] == display


# --- Maré: active study area is the IPP Territórios Sociais outline (PI
# ruling 2026-09-24, resolved_decisions id mare_site_study_area) ---

def test_mare_active_study_area_matches_the_ts03_outline():
    """Re-derive Maré's ACTIVE study area directly from the raw IPP outline
    file (config/sites.yaml maré.study_area.path), independent of
    src.sites.territory's own polygon_file branch, and check
    load_territory("maré") returns the exact same geometry — not merely
    "close", byte-identical (area diff < 1 m², symmetric difference < 1
    m²)."""
    cfg = load_sites_config()["maré"]
    assert cfg["study_area"]["kind"] == "polygon_file"
    outline = gpd.read_file(MAIN_ROOT / "data" / cfg["study_area"]["path"])
    outline = outline.set_crs(31983) if outline.crs is None else outline.to_crs(31983)
    expected = outline.geometry.union_all()

    t = load_territory("maré", root=MAIN_ROOT)
    assert abs(t.study_area.area - expected.area) < 1.0
    assert t.study_area.symmetric_difference(expected).area < 1.0


def test_mare_communities_union_kept_as_reproducible_historical_candidate():
    """The pre-2026-09-24 study area (union of the 16 communities ∩ data
    extent) is retired from `study_area` but must stay reproducible as a
    declared candidate/history entry (PI ruling task: 'keep the
    community-union definition as a declared candidate/history entry, so
    the old definition stays reproducible'). Re-derive it independently
    with the original formula and check the registry's candidate geometry
    matches byte-for-byte, and that it is smaller than the promoted outline
    (the outline covers the gaps between communities the union does not)."""
    cfg = load_sites_config()["maré"]
    cand_cfg = next(c for c in cfg["study_area_candidates"] if c["id"] == "communities_union_in_extent")
    assert cand_cfg["kind"] == "subunits_union_in_extent"

    bairro = gpd.read_file(MAIN_ROOT / "data" / cfg["data_extent"])
    bairro = bairro.set_crs(31983) if bairro.crs is None else bairro.to_crs(31983)
    extent = bairro.geometry.union_all()
    comm = gpd.read_file(MAIN_ROOT / "data" / cfg["subunits"]["file"], layer=cfg["subunits"]["layer"])
    comm = comm.set_crs(31983) if comm.crs is None else comm.to_crs(31983)
    share = comm.geometry.intersection(extent).area / comm.geometry.area
    included = comm[share.to_numpy() >= cand_cfg["share_threshold"]]
    expected_union = included.geometry.union_all().intersection(extent)

    t = load_territory("maré", root=MAIN_ROOT)
    cand = t.study_area_candidates["communities_union_in_extent"]
    assert abs(cand["geometry"].area - expected_union.area) < 1.0
    assert cand["geometry"].symmetric_difference(expected_union).area < 1.0
    assert cand["geometry"].area < t.study_area.area
    assert len(t.subunits_excluded) == len(comm) - len(included)


def test_mare_label_subunits_assigns_exactly_one_label_per_study_area_point():
    """Every study-area point/cell gets exactly one subunit label: a
    community name, or the literal "between communities" for study-area
    ground inside none of the 16 — and "between communities" must be
    non-empty (the whole reason the outline differs from the old
    community-union definition)."""
    t = load_territory("maré", root=MAIN_ROOT)
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = t.study_area.bounds
    xs = rng.uniform(x0, x1, 6000)
    ys = rng.uniform(y0, y1, 6000)
    in_area = within_mask(xs, ys, t.study_area)
    assert in_area.sum() > 0

    labels = label_subunits(xs, ys, t)
    assert (labels[in_area] != None).all()  # noqa: E711 — every study-area point labelled
    assert (labels[~in_area] == None).all()  # noqa: E711 — nothing outside the study area labelled

    valid_names = set(t.subunits["name"]) | {BETWEEN_SUBUNITS_LABEL}
    assert set(labels[in_area].tolist()) <= valid_names
    assert (labels[in_area] == BETWEEN_SUBUNITS_LABEL).any()  # non-empty, per the PI ruling

    # "exactly one": no study-area point may fall inside two community
    # polygons at once (a geometry sanity check, not just a labeller check).
    hits = np.zeros(len(xs), dtype=int)
    for _, row in t.subunits.iterrows():
        hits += within_mask(xs, ys, row.geometry).astype(int)
    assert (hits[in_area] <= 1).all()


def test_mare_marcilio_dias_excluded_from_active_study_area_by_geometry():
    """Marcílio Dias — one of the 16 communities — lies outside the
    promoted outline (and outside the data extent). The exclusion must be
    computed from geometry against the ACTIVE study area, not merely
    inherited from the old data-extent-based split (the two happen to
    agree here, but for the right reason: verified independently)."""
    t = load_territory("maré", root=MAIN_ROOT)
    excluded = t.subunits_excluded
    assert list(excluded["community"]) == ["Marcílio Dias"]
    marcilio = t.subunits.loc[t.subunits["community"] == "Marcílio Dias"].iloc[0]
    assert marcilio.geometry.intersection(t.study_area).area < 1.0
    assert float(marcilio["share_in_study_area"]) < 0.01
    # a clean separation, not a threshold that happens to bite once
    included_shares = t.subunits_included["share_in_study_area"].to_numpy()
    assert (included_shares >= 0.85).all()


def test_mare_rotation_recomputed_for_the_promoted_study_area():
    """display_rotation_deg is always `auto` (derived, never typed) — the
    promotion must have actually re-derived it from the new geometry, not
    carried over the old union's angle."""
    t = load_territory("maré", root=MAIN_ROOT)
    assert t.display_rotation_deg == rotation_deg(t.study_area)
    old_rotation = rotation_deg(t.study_area_candidates["communities_union_in_extent"]["geometry"])
    assert old_rotation != t.display_rotation_deg  # promotion changed the driving geometry


def test_mare_study_area_from_registry_matches_mare_study_area_module():
    """src.brisa_solar.mare_study_area is now a thin wrapper — its output
    for the same root must be pixel-for-pixel what load_territory returns."""
    from src.brisa_solar import mare_study_area as msa
    sa = msa.load_study_area(MAIN_ROOT)
    t = load_territory("maré", root=MAIN_ROOT)
    assert sa["study_area"].symmetric_difference(t.study_area).area < 1.0
    assert sorted(sa["excluded"]["community"]) == sorted(t.subunits_excluded["community"])


# --- consistency: study_area vs citywide must carry a definition_note when they differ ---

@pytest.mark.parametrize("site", SITES)
def test_study_area_vs_citywide_consistency_requires_definition_note(site, registry):
    t = load_territory(site, root=MAIN_ROOT)
    citywide_union = t.citywide.geometry.union_all() if len(t.citywide) else None
    if citywide_union is None or t.study_area.is_empty:
        differ = True
    else:
        symdiff = t.study_area.symmetric_difference(citywide_union).area
        larger = max(t.study_area.area, citywide_union.area, 1.0)
        differ = (symdiff / larger) > 1e-6
    note = registry[site].get("definition_note")
    if differ:
        assert note, (f"{site}: study_area and citywide_rule describe different "
                      f"boundaries but config/sites.yaml has no definition_note")
    # (a site whose two definitions agree is not required to carry a note,
    # but nothing here forbids one either)


def test_mare_definition_note_names_the_open_ops_card(registry):
    note = registry["maré"]["definition_note"]
    assert "mare_citywide_definition" in note


# --- rotation is derived from geometry, never typed ---

def test_display_rotation_deg_is_always_auto_in_the_registry(registry):
    for site, cfg in registry.items():
        assert cfg["display_rotation_deg"] == "auto", (
            f"{site}: display_rotation_deg must be 'auto' (derived), not a typed number")


def test_rotation_deg_changes_when_geometry_is_perturbed():
    base = box(0, 0, 100, 20)  # long axis already horizontal -> ~0 deg
    rotated = _shapely_rotate(base, 30, origin="centroid")
    a0 = rotation_deg(base)
    a1 = rotation_deg(rotated)
    assert a0 != a1
    assert abs((a1 - a0 - 30 + 90) % 180 - 90) < 1  # rotating the input by 30 deg rotates the answer by ~30 deg


def test_rotation_deg_is_a_whole_degree_in_minus90_90(territory):
    assert isinstance(territory.display_rotation_deg, int)
    assert -90 <= territory.display_rotation_deg <= 90


# --- subunits: Alemão/Rio das Pedras from Favelas_Limit_2019 grouped by complexo ---

def test_alemao_and_riodaspedras_subunits_are_favelas_limit_named_by_complexo():
    t_alemao = load_territory("complexo_do_alemao", root=MAIN_ROOT)
    t_rdp = load_territory("riodaspedras", root=MAIN_ROOT)
    assert len(t_alemao.subunits) == len(t_alemao.citywide)
    assert len(t_rdp.subunits) == len(t_rdp.citywide)
    assert set(t_alemao.subunits["nome"]) == set(t_alemao.citywide["nome"])


def test_vidigal_and_rocinha_have_no_subunits():
    assert load_territory("vidigal", root=MAIN_ROOT).subunits is None
    assert load_territory("rocinha", root=MAIN_ROOT).subunits is None


# --- Maré's IPP Territórios Sociais outline is ACTIVE (promoted 2026-09-24) ---

def test_mare_ipp_territorios_sociais_is_the_active_study_area_not_a_candidate():
    """Promotion (PI ruling 2026-09-24): the outline is now `study_area`
    itself, not a `study_area_candidates` entry — flip side of
    test_mare_communities_union_kept_as_reproducible_historical_candidate
    above, which checks the union went the other way."""
    reg = load_sites_config()["maré"]
    assert reg["study_area"]["kind"] == "polygon_file"
    assert "ipp_territorios_sociais.gpkg" not in str(reg["study_area"]["path"])  # sanity: not a stray old id
    assert reg["study_area"]["path"].endswith("ipp_territorios_sociais_territorio03.gpkg")
    assert all(c["id"] != "ipp_territorios_sociais" for c in reg["study_area_candidates"])

    t = load_territory("maré", root=MAIN_ROOT)
    cand = t.study_area_candidates["communities_union_in_extent"]
    assert cand["area_m2"] < t.study_area.area  # the contiguous outline is bigger than the community union


def test_build_study_area_raises_on_unknown_kind():
    from src.sites.territory import build_study_area
    with pytest.raises(ValueError):
        build_study_area({"kind": "not_a_real_kind"}, box(0, 0, 1, 1),
                         gpd.GeoDataFrame(geometry=[]), None, MAIN_ROOT)
