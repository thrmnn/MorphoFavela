"""Per-site display metadata for the Folha de Rua dashboard."""

from __future__ import annotations

SITE_ORDER = ["vidigal", "rocinha", "complexo_do_alemao", "riodaspedras", "maré"]

DISPLAY_NAME = {
    "vidigal": "Vidigal",
    "rocinha": "Rocinha",
    "complexo_do_alemao": "Complexo do Alemão",
    "riodaspedras": "Rio das Pedras",
    "maré": "Maré",
}

TYPOLOGY = {
    "vidigal": ("HILLSIDE CANYON", "#2C5F8D"),
    "rocinha": ("HILLSIDE CANYON", "#2C5F8D"),
    "complexo_do_alemao": ("HILLSIDE CANYON", "#2C5F8D"),
    "maré": ("DENSE LOW-RISE GRID", "#6B7280"),
    "mare": ("DENSE LOW-RISE GRID", "#6B7280"),
    "riodaspedras": ("FLAT-DENSE ALLEYS", "#5B7C5B"),
}

SHEET_NUMBER = {
    "vidigal": "01",
    "rocinha": "02",
    "complexo_do_alemao": "03",
    "riodaspedras": "04",
    "maré": "05",
}
