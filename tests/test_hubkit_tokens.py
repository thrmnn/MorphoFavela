"""Token-completeness invariants for scripts/hubkit.py's CSS.

hubkit.CSS is served inside the brisaverse hub, which forces a dark body when
the viewer prefers dark mode (see hub/server.py's MIRROR_BRIDGE_CSS). Any
component rule that hardcodes a light-mode colour instead of a --token can
silently win the cascade over the dark body and paint text dark-on-dark. These
tests enforce the fix mechanically: every colour lives in one of the three
token blocks (light :root, the dark @media block, the dark [data-theme]
block) except an explicit, justified WHITELIST, and every token the light
block defines also has a value in both dark blocks.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hubkit  # noqa: E402

_COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)")
_TOKEN_RE = re.compile(r"--([a-z0-9-]+)\s*:\s*([^;}]+)")


def _token_blocks(css: str):
    """Return (light_root_body, dark_media_body, dark_attr_body) — the raw
    text inside each of the three token-defining braces, in source order."""
    root = re.search(r":root\{(.*?)\}\n/\* Dark palette", css, re.DOTALL)
    media = re.search(
        r'@media \(prefers-color-scheme: dark\)\{\n:root:not\(\[data-theme="light"\]\)\{(.*?)\}\n\}',
        css, re.DOTALL)
    attr = re.search(r':root\[data-theme="dark"\]\{(.*?)\}\n\*\{', css, re.DOTALL)
    assert root and media and attr, "could not locate the three hubkit token blocks"
    return root.group(1), media.group(1), attr.group(1)


def _spans(css: str):
    """Byte spans of the three token blocks (including their braces), so
    literals found inside them are excluded from the outside-token scan."""
    spans = []
    for pat in (
        r":root\{.*?\}\n/\* Dark palette",
        r'@media \(prefers-color-scheme: dark\)\{\n:root:not\(\[data-theme="light"\]\)\{.*?\}\n\}',
        r':root\[data-theme="dark"\]\{.*?\}\n\*\{',
    ):
        m = re.search(pat, css, re.DOTALL)
        assert m, f"pattern not found: {pat}"
        spans.append(m.span())
    return spans


def _literals_outside_token_blocks(css: str) -> list[str]:
    spans = _spans(css)

    def inside_any(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    out = []
    for m in _COLOR_RE.finditer(css):
        if inside_any(m.start()):
            continue
        out.append(m.group(0))
    return out


def test_no_color_literal_outside_token_blocks_except_whitelist():
    literals = _literals_outside_token_blocks(hubkit.CSS)
    offenders = [lit for lit in literals if lit not in hubkit.CSS_LITERAL_WHITELIST]
    assert not offenders, (
        f"hardcoded colour(s) outside the token blocks and not in "
        f"CSS_LITERAL_WHITELIST: {offenders}"
    )


def test_every_light_token_has_a_dark_value_in_both_dark_blocks():
    light, media, attr = _token_blocks(hubkit.CSS)
    light_tokens = dict(_TOKEN_RE.findall(light))
    media_tokens = dict(_TOKEN_RE.findall(media))
    attr_tokens = dict(_TOKEN_RE.findall(attr))
    # --lh and --serif are non-colour tokens with no theme variance; every
    # colour token must reappear in both dark blocks.
    color_tokens = {
        name for name, val in light_tokens.items() if _COLOR_RE.fullmatch(val.strip())
    }
    missing_media = color_tokens - media_tokens.keys()
    missing_attr = color_tokens - attr_tokens.keys()
    assert not missing_media, f"tokens missing from the @media dark block: {missing_media}"
    assert not missing_attr, f'tokens missing from the :root[data-theme="dark"] block: {missing_attr}'


def test_sabotaged_css_is_caught_by_the_literal_scan():
    """Prove the literal-scan assertion can actually fire: inject a raw hex
    colour into a copy of the CSS string, outside all three token blocks, and
    confirm the same check that passes above now finds it."""
    sabotaged = hubkit.CSS.replace(
        ".doc strong{color:var(--ink)}",
        ".doc strong{color:var(--ink);background:#ff00ff}",
        1,
    )
    assert ".doc strong{color:var(--ink);background:#ff00ff}" in sabotaged
    literals = _literals_outside_token_blocks(sabotaged)
    offenders = [lit for lit in literals if lit not in hubkit.CSS_LITERAL_WHITELIST]
    assert "#ff00ff" in offenders, "sabotage did not register as an offending literal"


def test_sabotaged_css_missing_dark_token_is_caught():
    """Prove the completeness assertion can fire: delete one token's dark
    value from the @media block only, and confirm the check catches it."""
    light, media, attr = _token_blocks(hubkit.CSS)
    sabotaged_media = media.replace("--accent:#3ecab2;", "")
    sabotaged_css = hubkit.CSS.replace(media, sabotaged_media, 1)
    _, media2, attr2 = _token_blocks(sabotaged_css)
    light_tokens = dict(_TOKEN_RE.findall(light))
    media_tokens = dict(_TOKEN_RE.findall(media2))
    color_tokens = {
        name for name, val in light_tokens.items() if _COLOR_RE.fullmatch(val.strip())
    }
    missing_media = color_tokens - media_tokens.keys()
    assert "accent" in missing_media, "sabotage did not register as a missing dark token"
