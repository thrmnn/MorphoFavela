"""Plain-language context for the ventilation measures: what each one
measures, how it is computed here, how to read its values, why it matters for
air temperature and what it cannot capture.

Two entry points return markdown: ``report_paragraphs`` for report section 8
and ``readme_subsection`` for the README methods. Every data value comes from
the facts dict of ``report.compute_facts``; the flow regime thresholds are
literature constants, kept here with their source.
"""
from __future__ import annotations

from src.om_package.shade import OM2_SHADE_MAX_DIST_M
from src.om_package.vent_indices import (
    DEFAULT_BUFFER_M,
    MACDONALD_A,
    MACDONALD_BETA,
    MACDONALD_CD,
    VON_KARMAN,
)
from src.om_package.ventilation import MAX_JOIN_DIST_GRID_M

#: Height-to-width ratios bounding isolated roughness, wake interference and
#: skimming flow for wind across a long street (Oke 1988, as summarised with
#: these values by Voordeckers et al. 2021, section 4.1.2).
WAKE_FROM_RATIO = 0.35
SKIM_FROM_RATIO = 0.65
#: Voordeckers et al. 2021, section 4.1.3: a street counts as deep above a ratio of 1.5 to 2.0.
DEEP_FROM_RATIO = (1.5, 2.0)
GRID_CELL_M = 10
N_COMPASS = 8

REFERENCES = {
    "grimmond_oke_1999": (
        "Grimmond, C. S. B., Oke, T. R. (1999). Aerodynamic properties of urban areas derived from analysis "
        "of surface form. Journal of Applied Meteorology 38(9), 1262 to 1292. "
        "https://doi.org/10.1175/1520-0450(1999)038<1262:APOUAD>2.0.CO;2"
    ),
    "macdonald_1998": (
        "Macdonald, R. W., Griffiths, R. F., Hall, D. J. (1998). An improved method for the estimation of "
        "surface roughness of obstacle arrays. Atmospheric Environment 32(11), 1857 to 1864. "
        "https://doi.org/10.1016/S1352-2310(97)00403-2"
    ),
    "ng_2011": (
        "Ng, E., Yuan, C., Chen, L., Ren, C., Fung, J. C. H. (2011). Improving the wind environment in "
        "high-density cities by understanding urban morphology and surface roughness: a study in Hong Kong. "
        "Landscape and Urban Planning 101(1), 59 to 74. https://doi.org/10.1016/j.landurbplan.2011.01.004"
    ),
    "nunez_oke_1977": (
        "Nunez, M., Oke, T. R. (1977). The energy balance of an urban canyon. Journal of Applied Meteorology "
        "16(1), 11 to 19. https://doi.org/10.1175/1520-0450(1977)016<0011:TEBOAU>2.0.CO;2"
    ),
    "oke_1988": (
        "Oke, T. R. (1988). Street design and urban canopy layer climate. Energy and Buildings 11(1 to 3), "
        "103 to 113. https://doi.org/10.1016/0378-7788(88)90026-6"
    ),
    "voordeckers_2021": (
        "Voordeckers, D., Lauriks, T., Denys, S., Billen, P., Tytgat, T., Van Acker, M. (2021). Guidelines "
        "for passive control of traffic-related air pollution in street canyons: an overview for urban "
        "planning. Landscape and Urban Planning 207, 103980. https://doi.org/10.1016/j.landurbplan.2020.103980"
    ),
}


def _regimes(f: dict):
    camp = f["regimes"]["campaign"]
    k1, k2 = sorted(camp)
    return (camp[k1], f["vent"][k1]), (camp[k2], f["vent"][k2])


def _regime_value(f: dict, key: str, fmt: str) -> str:
    (r1, v1), (r2, v2) = _regimes(f)
    return (f"{format(v1[key], fmt)} for the {r1['name']} wind and {format(v2[key], fmt)} for the "
            f"{r2['name']} wind")


def _skim_reading(hw: float) -> str:
    if hw >= SKIM_FROM_RATIO:
        return ("in the skimming range: where the wind blows across a street, it mostly passes over the "
                "roofs while the street air turns in a vortex below")
    if hw >= WAKE_FROM_RATIO:
        return "in the wake interference range"
    return "in the isolated roughness range"


def frontal_paragraph(f: dict) -> str:
    return (
        "**How to read frontal area density.** It measures how much building wall the wind meets. For one "
        f"wind direction, each building in the {GRID_CELL_M} m grid cell is seen from the wind: its width "
        "across the wind times its height gives the wall area facing the wind, and the sum is divided by the "
        "ground area of the cell. A value of 0 means no building; a value of 1 means as much wall faces the "
        "wind as there is ground. Low values mean the buildings stand apart and the wind reaches the street; "
        "high values mean they stand close, and less wind reaches the street (Ng et al., 2011). Along this "
        f"route the median is {_regime_value(f, 'frontal_median', '.2f')}.\n"
    )


def alignment_paragraph(f: dict) -> str:
    hw = f["hw_median_of_ratios"]
    return (
        "**How to read canyon alignment.** It is the angle between the street and the wind, from 0° to 90°. "
        "A street has no front or back, so wind along it in either sense counts the same. Near 0°, the wind "
        "blows along the street and is channelled down it. Near 90°, it blows across, and the flow depends on "
        f"the height-to-width ratio. Below about {WAKE_FROM_RATIO:g}, each building acts as a separate "
        f"obstacle (isolated roughness flow). Between about {WAKE_FROM_RATIO:g} and {SKIM_FROM_RATIO:g}, the "
        "wake behind one building reaches the next (wake interference flow). Above about "
        f"{SKIM_FROM_RATIO:g}, the wind skims over the roofs and drives a turning vortex inside the street "
        "(skimming flow) (Oke, 1988; Voordeckers et al., 2021). At angles in between, the air moves along the "
        "street in a corkscrew (Voordeckers et al., 2021). The median of the point height-to-width ratios on "
        f"this route is {hw:.1f}, {_skim_reading(hw)}.\n"
    )


def shelter_paragraph(f: dict) -> str:
    return (
        "**How to read the upwind shelter angle.** Stand in the street, look into the wind and raise your "
        "eyes until no building or ground blocks the view: that angle above the horizontal is the upwind "
        f"shelter angle, seen from {f['height_m']:g} m above the street and searched up to "
        f"{OM2_SHADE_MAX_DIST_M:g} m away. At 0°, nothing rises above eye level towards the wind. At 45°, the "
        "top of the obstruction is as high above eye level as it is far away. The higher the angle, the "
        "closer or taller the obstruction the wind meets before it reaches the point.\n"
    )


def why_paragraph() -> str:
    return (
        "**Why it matters for air temperature.** By day, a street absorbs more radiation than it emits. Most "
        "of that surplus leaves through turbulent mixing with the air above, and air carried in by the wind "
        "adds or removes heat depending on wind direction and speed (Nunez and Oke, 1977). The three "
        "measures say how open each point is to that exchange.\n"
    )


def limits_paragraph() -> str:
    return (
        "**What they cannot capture.** They describe geometry, not wind: no wind was measured in the streets "
        "or simulated. The airport wind is a regional reference, and each measure uses the mean direction of "
        "its regime, not the wind of the hour. They leave out air set in motion by heating, trees, and "
        "openings between and through buildings. Real street flow mixes channelling, vortices and corkscrews "
        "in three dimensions (Voordeckers et al., 2021).\n"
    )


def report_paragraphs(f: dict) -> list[str]:
    """Markdown paragraphs for report section 8, after the definitions."""
    return [frontal_paragraph(f), alignment_paragraph(f), shelter_paragraph(f), why_paragraph(),
            limits_paragraph()]


def readme_subsection(f: dict, heading: str = "###") -> str:
    """Fuller methods text for the README: each measure with its computation."""
    (r1, v1), (r2, v2) = _regimes(f)
    deep_lo, deep_hi = DEEP_FROM_RATIO
    z0 = f["z0_median"]
    z0_txt = " and ".join(f"{v:.2f} m for the {r['name']} wind" for r, v in
                          ((r1, z0[f"z0_macdonald_m_{r1['slug']}"]), (r2, z0[f"z0_macdonald_m_{r2['slug']}"])))
    sh1, sh2 = v1["shelter"], v2["shelter"]
    band = f["align_band_deg"]
    parts = [
        f"{heading} Ventilation measures\n",
        "Each ventilation measure is computed from 2019 building and terrain geometry for the mean direction of "
        f"each wind regime ({r1['name']}, {r1['dir']:.0f}°; {r2['name']}, {r2['dir']:.0f}°). None is a "
        "measured or simulated wind. They say how open a point is to a wind from that direction.\n",
        why_paragraph(),
        f"{heading}# Frontal area density facing the wind\n",
        "It measures how much building wall the wind meets per unit of ground. For each of the "
        f"{N_COMPASS} compass directions, every building footprint is clipped to the {GRID_CELL_M} m grid "
        "cell; its width across the wind (the width of its smallest enclosing rectangle seen from that "
        "direction) times its height gives its wall area facing the wind, and the sum is divided by the cell "
        "area. The value for a regime is interpolated between the two compass directions on either side of "
        "its mean direction. Each point takes the value of the grid cell whose centre is nearest, within "
        f"{MAX_JOIN_DIST_GRID_M:g} m.\n",
        "A value of 0 means no building in the cell; 1 means as much wall faces the wind as there is ground. "
        "Low values mean the buildings stand apart and the wind reaches the street; high values mean they "
        "stand close, so less wind reaches the street. Ng et al. (2011) used frontal area density, with the "
        "share of ground covered by buildings, to map how well wind reaches pedestrians in Hong Kong, checked "
        f"against wind tunnel tests. Along this route the median is {v1['frontal_median']:.2f} for the "
        f"{r1['name']} wind and {v2['frontal_median']:.2f} for the {r2['name']} wind: in most cells, more wall "
        "faces the wind than there is ground.\n",
        f"{heading}# Canyon alignment\n",
        "It is the angle between the street axis and the wind direction. The street axis at each point is "
        "the bearing between its two neighbouring route points. Both the street and the wind are folded so "
        "that the angle runs from 0° (wind along the street) to 90° (wind across it).\n",
        "Near 0°, the wind is channelled along the street. Near 90°, the flow depends on the height-to-width "
        f"ratio. Below about {WAKE_FROM_RATIO:g}, each building acts as a separate obstacle and the wind "
        f"recovers before the next (isolated roughness flow). Between about {WAKE_FROM_RATIO:g} and "
        f"{SKIM_FROM_RATIO:g}, the wake behind one building reaches the next (wake interference flow). Above "
        f"about {SKIM_FROM_RATIO:g}, the wind skims over the roofs and drives a turning vortex inside the "
        "street, which exchanges less air with the flow above (skimming flow) (Oke, 1988; Voordeckers et "
        f"al., 2021). In deep streets, with a ratio above {deep_lo:g} to {deep_hi:g}, two vortices turning in "
        "opposite senses can form, the lower one weaker. At angles in between, the air moves along the street "
        "in a corkscrew (Voordeckers et al., 2021). The median of the point height-to-width ratios on this "
        f"route is {f['hw_median_of_ratios']:.1f}, {_skim_reading(f['hw_median_of_ratios'])}.\n",
        f"For the {r1['name']} wind, {v1['align_along']:.0%} of points lie within {band:.0f}° of along the "
        f"street and {v1['align_across']:.0%} within {band:.0f}° of across it; for the {r2['name']} wind, "
        f"{v2['align_along']:.0%} and {v2['align_across']:.0%}.\n",
        f"{heading}# Upwind shelter angle\n",
        "It is the angle above the horizontal at which buildings and terrain stop blocking the view when you "
        f"look into the wind from {f['height_m']:g} m above the street. It comes from the same horizon "
        "profiles as the sun and shade measures: a 1 m surface of terrain plus buildings, searched up to "
        f"{OM2_SHADE_MAX_DIST_M:g} m from the point, read at the profile direction nearest the wind "
        "direction.\n",
        "At 0°, nothing rises above eye level towards the wind. At 45°, the top of the obstruction is as high "
        "above eye level as it is far away. The higher the angle, the closer or taller the obstruction the "
        f"wind meets before it reaches the point. Half of the points lie between {sh1[0]:.0f}° and "
        f"{sh1[2]:.0f}° for the {r1['name']} wind and between {sh2[0]:.0f}° and {sh2[2]:.0f}° for the "
        f"{r2['name']} wind.\n",
        f"{heading}# Roughness length and displacement height\n",
        "These two lengths describe how the buildings slow the wind above them. The displacement height is "
        "the height at which the wind above behaves as if the ground were raised; the roughness length says "
        "how much the surface above that height brakes the wind. Both follow Macdonald et al. (1998), "
        f"from the mean building height and plan area density in a {DEFAULT_BUFFER_M} m circle around the "
        "point and the frontal area density facing the wind, with the constants the authors give for "
        f"staggered arrays (A = {MACDONALD_A:g}, β = {MACDONALD_BETA:g}, drag coefficient {MACDONALD_CD:g}, "
        f"von Kármán constant {VON_KARMAN:g}). In this method the displacement height approaches the "
        "building height as buildings cover more of the ground, and the roughness length then falls: the "
        "wind skims over a closed roof surface. Along this route the median displacement height is "
        f"{f['zd_median']:.1f} m and the median roughness length is {z0_txt}. These estimates from building "
        "form are uncertain: Grimmond and Oke (1999) found few wind observations good enough to test them, "
        "and only weak relations between measured values and building density.\n",
        f"{heading}# Open space fraction\n",
        f"It is the share of the {DEFAULT_BUFFER_M} m circle around the point not covered by building "
        "footprints. Streets, courtyards and empty plots all count as open.\n",
        f"{heading}# What the measures cannot capture\n",
        limits_paragraph().replace("**What they cannot capture.** ", ""),
    ]
    return "\n".join(parts)


def references_used() -> dict[str, str]:
    """Reference strings for the README references list."""
    return dict(REFERENCES)
