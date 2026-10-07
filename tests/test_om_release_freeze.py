"""A version sent to the team is frozen: the build refuses to write it, the
package page keeps its links on it, and its files are checked against the
fingerprints in package_docs.RELEASED_VERSIONS."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest

from src.om_package import package_docs
from src.om_package.package_docs import RELEASED_VERSIONS, frozen_release_error, verify_released

REPO = Path(__file__).resolve().parents[1]
REAL_PACKAGE_ROOT = REPO / "outputs" / "_packages" / "mare_om2"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _snapshot(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): (p.stat().st_mtime_ns, _sha(p)) for p in d.rglob("*") if p.is_file()}


def _fake_release(package_root: Path, version: str, monkeypatch) -> Path:
    vdir = package_root / version
    (vdir / "data").mkdir(parents=True)
    (vdir / "data" / "t.csv").write_text("a\n1\n")
    (vdir / "README.md").write_text("# README\n")
    files = {rel: _sha(vdir / rel) for rel in ("README.md", "data/t.csv")}
    (vdir / "manifest.json").write_text(json.dumps({"files": files}))
    zip_path = package_root / f"octopus_om2_{version}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(vdir / "README.md", "README.md")
    monkeypatch.setitem(RELEASED_VERSIONS, version, {
        "manifest_sha256": _sha(vdir / "manifest.json"), "zip_sha256": _sha(zip_path), "sent": "2026-10-06"})
    return vdir


def test_default_version_is_not_a_released_one():
    assert package_docs.VERSION not in RELEASED_VERSIONS


def test_red_build_targeting_a_released_version_refuses_and_touches_nothing(tmp_path, monkeypatch, capsys):
    package_root = tmp_path / "outputs" / "_packages" / "mare_om2"
    vdir = _fake_release(package_root, "v1.0.0", monkeypatch)
    before = _snapshot(package_root)
    bom = _load("build_om_package")
    for stage in ("all", "compute", "package"):
        monkeypatch.setattr(sys, "argv", ["build_om_package.py", "--root", str(tmp_path), "--version", "v1.0.0",
                                          "--stage", stage, "--skip-page"])
        assert bom.main() != 0
        assert "v1.0.0 was sent to the team on 2026-10-06 and is frozen" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["build_om_package.py", "--root", str(tmp_path), "--out", str(vdir)])
    assert bom.main() != 0
    assert _snapshot(package_root) == before
    assert not (tmp_path / "outputs" / "_packages" / "_cache").exists()


def test_scratch_out_with_a_released_version_name_is_allowed(tmp_path):
    assert frozen_release_error(tmp_path / "scratch" / "v1.0.0", tmp_path / "outputs" / "_packages" / "mare_om2") is None
    assert frozen_release_error(REAL_PACKAGE_ROOT / "v1.0.0", REAL_PACKAGE_ROOT) is not None


def test_red_verify_released_names_every_change(tmp_path, monkeypatch):
    package_root = tmp_path / "mare_om2"
    vdir = _fake_release(package_root, "v9.9.9", monkeypatch)
    assert verify_released(package_root, "v9.9.9") == []
    (vdir / "data" / "t.csv").write_text("a\n2\n")
    (vdir / "figures").mkdir()
    (vdir / "figures" / "new.png").write_bytes(b"x")
    (package_root / "octopus_om2_v9.9.9.zip").write_bytes(b"not the sent zip")
    fails = "\n".join(verify_released(package_root, "v9.9.9"))
    assert "t.csv: sha256 differs from the manifest" in fails
    assert "new.png: not listed in the released manifest" in fails
    assert "octopus_om2_v9.9.9.zip: sha256 differs" in fails


@pytest.mark.skipif(not (REAL_PACKAGE_ROOT / "v1.0.0").is_dir(), reason="released v1.0.0 not on disk")
def test_released_v1_0_0_matches_its_fingerprints():
    assert verify_released(REAL_PACKAGE_ROOT, "v1.0.0") == []


@pytest.fixture
def page_root(tmp_path):
    """A root whose package dir holds the real released v1.0.0 (symlinked,
    never written) and an unreleased v1.0.1 draft beside it."""
    if not (REAL_PACKAGE_ROOT / "v1.0.0").is_dir():
        pytest.skip("released v1.0.0 not on disk")
    root = tmp_path / "root"
    pkg = root / "outputs" / "_packages" / "mare_om2"
    internal = root / "outputs" / "_packages" / "_internal" / "mare_om2"
    pkg.mkdir(parents=True)
    internal.mkdir(parents=True)
    (pkg / "v1.0.0").symlink_to(REAL_PACKAGE_ROOT / "v1.0.0")
    (pkg / "octopus_om2_v1.0.0.zip").symlink_to(REAL_PACKAGE_ROOT / "octopus_om2_v1.0.0.zip")
    (pkg / "v1.0.1").symlink_to(REAL_PACKAGE_ROOT / "v1.0.0")
    (pkg / "octopus_om2_v1.0.1.zip").write_bytes(b"draft")
    real_internal = REPO / "outputs" / "_packages" / "_internal" / "mare_om2"
    for v in ("v1.0.0", "v1.0.1"):
        (internal / v).symlink_to(real_internal / "v1.0.0")
    for p in (REPO / "docs").glob("*"):
        (root / "docs").mkdir(exist_ok=True)
        (root / "docs" / p.name).symlink_to(p)
    return root


def test_page_links_stay_on_the_newest_released_version(page_root):
    page = _load("build_om_package_page")
    html = page.render_page(page_root)
    hrefs = re.findall(r'href="([^"]+)"', html)
    assert any(h.endswith("octopus_om2_v1.0.0.zip") for h in hrefs)
    assert any(h.endswith("v1.0.0/report.pdf") for h in hrefs)
    assert not [h for h in hrefs if "v1.0.1" in h]
    assert "DRAFT, not sent to the team: v1.0.1" in html


def test_red_page_refuses_a_released_version_that_changed(page_root, monkeypatch):
    page = _load("build_om_package_page")
    monkeypatch.setitem(RELEASED_VERSIONS, "v1.0.0", {**RELEASED_VERSIONS["v1.0.0"], "zip_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="released v1.0.0 no longer matches its fingerprints"):
        page.render_page(page_root)
