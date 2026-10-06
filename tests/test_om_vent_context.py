import re

import pytest

from src.om_package import vent_context as vc

FORBIDDEN = ["—", "–", "λ", "H/W", "z0", "SBGL", "METAR"]


@pytest.fixture
def facts():
    return {
        "regimes": {"campaign": {
            "reg1": {"name": "east-southeast", "slug": "east_southeast", "dir": 117.4},
            "reg2": {"name": "north-northwest", "slug": "north_northwest", "dir": 333.6},
        }},
        "vent": {
            "reg1": {"frontal_median": 1.234, "shelter": [21.2, 44.4, 63.6], "align_along": 0.413,
                     "align_across": 0.527},
            "reg2": {"frontal_median": 0.876, "shelter": [11.1, 37.7, 71.3], "align_along": 0.061,
                     "align_across": 0.019},
        },
        "align_band_deg": 30.0,
        "hw_median_of_ratios": 1.87,
        "height_m": 1.5,
        "z0_median": {"z0_macdonald_m_east_southeast": 0.071, "z0_macdonald_m_north_northwest": 0.046},
        "zd_median": 8.64,
    }


def _texts(f):
    return vc.report_paragraphs(f) + [vc.readme_subsection(f)]


def test_no_forbidden_strings(facts):
    for text in _texts(facts):
        for s in FORBIDDEN:
            assert s not in text, s


def test_numbers_come_from_facts(facts):
    report = "\n".join(vc.report_paragraphs(facts))
    for s in ["1.9"]:
        assert s in report
    readme = vc.readme_subsection(facts)
    for s in ["1.23", "0.88", "21°", "64°", "11°", "71°", "8.6 m", "0.07 m", "0.05 m", "117°", "334°", "41%", "53%"]:
        assert s in readme, s
    facts["vent"]["reg1"]["frontal_median"] = 2.468
    assert "2.47" in vc.readme_subsection(facts)


def test_skimming_reading_follows_ratio(facts):
    assert "skimming range" in vc.alignment_paragraph(facts)
    facts["hw_median_of_ratios"] = 0.2
    assert "isolated roughness range" in vc.alignment_paragraph(facts)


def test_every_cited_source_has_a_reference(facts):
    text = "\n".join(_texts(facts))
    cited = {"oke_1988": "Oke, 1988", "voordeckers_2021": "Voordeckers et al.", "ng_2011": "Ng et al.",
             "nunez_oke_1977": "Nunez and Oke", "macdonald_1998": "Macdonald et al.",
             "grimmond_oke_1999": "Grimmond and Oke"}
    for key, needle in cited.items():
        assert needle in text
        assert re.search(r"https://doi\.org/10\.", vc.REFERENCES[key])
    for ref in vc.references_used().values():
        assert "—" not in ref and "–" not in ref


def test_schematic_at_text_width(tmp_path):
    from PIL import Image

    from src.om_package.vent_schematic import build_fig_vent_schematic

    p = build_fig_vent_schematic(tmp_path / "fig_vent_schematic.png")
    assert Image.open(p).size[0] == 1260
