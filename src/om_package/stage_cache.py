"""Content-addressed stage cache for the OM2 build.

A stage's key is the sha256 of a canonical JSON over four components: the
sha256 of every input file, the stage's parameters, the sha256 of the
stage's own source files (file content, so an uncommitted edit counts) and
the keys of its upstream stages. Artefacts live under
``<cache_root>/<stage>/<key>/`` with a ``meta.json`` that records the four
components, git HEAD and dirty flag, run time and the sha256 of every
artefact file. Same idea as a DVC stage, in plain Python.

Artefacts are stored in layout-independent logical names: a stage function
writes files into its work directory and/or returns Python objects, stored
as parquet (DataFrames, when the round trip is exact), .npy (arrays), JSON
(when the round trip is exact) or pickle (anything else).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

META = "meta.json"
OBJECTS = "_objects"
HASH_MEMO = "_input_hash_memo.json"
#: libraries whose version can change the bytes a stage writes; part of the key.
KEY_LIBRARIES = ("numpy", "pandas", "pyarrow", "geopandas", "shapely", "pyproj", "rasterio", "scipy", "torch")


class StaleCacheError(RuntimeError):
    """A cache entry is missing, or does not match its inputs, code, params or artefacts."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def expand_inputs(inputs: dict[str, Path]) -> dict[str, Path]:
    """A directory input becomes one entry per file under it (label/relpath)."""
    out: dict[str, Path] = {}
    for label, p in inputs.items():
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    out[f"{label}/{f.relative_to(p).as_posix()}"] = f
        else:
            out[label] = p
    return out


class HashMemo:
    """sha256 per file, reused while (path, size, mtime_ns) is unchanged."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.data: dict = {}
        if self.path and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        self.dirty = False

    def sha256(self, p: Path) -> str:
        p = Path(p)
        if not p.is_file():
            raise StaleCacheError(f"input file missing: {p}")
        st = p.stat()
        k = str(p.resolve())
        hit = self.data.get(k)
        if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
            return hit["sha256"]
        digest = sha256_file(p)
        self.data[k] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        self.dirty = True
        return digest

    def save(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)
            self.dirty = False


def _resolve_import(mod: str, repo_root: Path) -> list[Path]:
    base = repo_root.joinpath(*mod.split("."))
    return [c for c in (base.with_suffix(".py"), base / "__init__.py") if c.is_file()]


def module_closure(entry_files: list[Path], repo_root: Path, exclude: set[Path] = frozenset()) -> list[Path]:
    """Repo source files reachable from ``entry_files`` through import
    statements (anywhere in the file, function-local ones included).
    Third-party imports are ignored; ``exclude`` files are neither hashed
    nor followed."""
    repo_root = Path(repo_root).resolve()
    exclude = {Path(p).resolve() for p in exclude}
    seen: set[Path] = set()
    todo = [Path(p).resolve() for p in entry_files]
    while todo:
        f = todo.pop()
        if f in seen or f in exclude:
            continue
        seen.add(f)
        pkg = f.parent.relative_to(repo_root).parts if f.is_relative_to(repo_root) else ()
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    anchor = list(pkg[: len(pkg) - (node.level - 1)])
                    base = ".".join(anchor + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                mods = [base] + [f"{base}.{a.name}" for a in node.names]
            for m in mods:
                todo.extend(_resolve_import(m, repo_root))
    return sorted(seen)


def library_versions() -> dict[str, str | None]:
    out = {}
    for lib in KEY_LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


@dataclass
class StageSpec:
    name: str
    inputs: dict[str, Path]
    params: dict
    code_modules: list[Path]
    upstream: dict[str, str] = field(default_factory=dict)
    repo_root: Path | None = None


def stage_components(spec: StageSpec, memo: HashMemo) -> dict:
    root = Path(spec.repo_root).resolve() if spec.repo_root else None

    def rel(p: Path) -> str:
        p = Path(p).resolve()
        return p.relative_to(root).as_posix() if root and p.is_relative_to(root) else p.as_posix()

    return {
        "stage": spec.name,
        "inputs": {label: memo.sha256(p) for label, p in sorted(expand_inputs(spec.inputs).items())},
        "params": json.loads(_canonical(spec.params)),
        "code": {rel(p): sha256_file(p) for p in sorted(spec.code_modules)},
        "upstream": dict(sorted(spec.upstream.items())),
        "libraries": library_versions(),
    }


def key_of(components: dict) -> str:
    return hashlib.sha256(_canonical(components).encode()).hexdigest()


def stage_key(name: str, inputs: dict[str, Path], params: dict, code_modules: list[Path],
              upstream: dict[str, str], memo: HashMemo | None = None, repo_root: Path | None = None) -> str:
    spec = StageSpec(name, inputs, params, code_modules, upstream, repo_root)
    return key_of(stage_components(spec, memo or HashMemo(None)))


def _diff_components(old: dict, new: dict) -> list[str]:
    out = []
    for comp in ("inputs", "params", "code", "upstream", "libraries"):
        a, b = old.get(comp, {}), new.get(comp, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                what = "added" if k not in a else "removed" if k not in b else "changed"
                out.append(f"{comp} {k} {what}")
    return out


def git_state(repo_root: Path | None) -> tuple[str | None, bool | None]:
    if repo_root is None:
        return None, None
    try:
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _store_object(d: Path, name: str, obj) -> str:
    if isinstance(obj, pd.DataFrame) and not hasattr(obj, "geometry"):
        p = d / f"{name}.parquet"
        obj.to_parquet(p, index=False, compression="snappy")
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(p), obj, check_exact=True)
            return p.name
        except (AssertionError, TypeError, ValueError):
            p.unlink()
    elif isinstance(obj, np.ndarray) and obj.dtype != object:
        p = d / f"{name}.npy"
        np.save(p, obj, allow_pickle=False)
        return p.name
    else:
        try:
            text = json.dumps(obj, indent=2)
            if json.loads(text) == obj:
                p = d / f"{name}.json"
                p.write_text(text)
                return p.name
        except (TypeError, ValueError):
            pass
    p = d / f"{name}.pkl"
    p.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    return p.name


def _load_object(p: Path):
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pickle.loads(p.read_bytes())


@dataclass
class Entry:
    """A verified cache entry: ``files`` maps every artefact file name to its
    path, ``objects`` the returned Python objects (loaded lazily on access)."""

    name: str
    key: str
    dir: Path
    meta: dict
    ran: bool

    def file(self, name: str) -> Path:
        if name not in self.meta["artefacts"]:
            raise StaleCacheError(f"stage {self.name}: no artefact {name!r} in {self.dir}")
        return self.dir / name

    def obj(self, name: str):
        rel = self.meta["objects"].get(name)
        if rel is None:
            raise StaleCacheError(f"stage {self.name}: no object {name!r} in {self.dir}")
        return _load_object(self.dir / rel)


def verify(meta: dict, entry_dir: Path, components: dict | None = None) -> None:
    """Raise StaleCacheError if the entry's recorded components differ from
    ``components`` (the current ones) or any artefact's bytes differ from
    its recorded sha256. Names the differing component or file."""
    if components is not None:
        key = key_of(components)
        if meta.get("key") != key:
            diff = _diff_components(meta.get("components", {}), components)
            raise StaleCacheError(f"stage {meta.get('stage')}: cached key {meta.get('key', '?')[:12]} != current "
                                  f"{key[:12]}; differs in: {', '.join(diff) or 'unknown component'}")
    for rel, digest in meta["artefacts"].items():
        p = Path(entry_dir) / rel
        if not p.is_file():
            raise StaleCacheError(f"stage {meta.get('stage')}: artefact {rel} missing from {entry_dir}")
        if sha256_file(p) != digest:
            raise StaleCacheError(f"stage {meta.get('stage')}: artefact {rel} was modified after it was cached "
                                  f"(sha256 differs from meta.json in {entry_dir})")


def _latest_other(stage_dir: Path, key: str) -> dict | None:
    metas = [p for p in stage_dir.glob(f"*/{META}") if p.parent.name != key]
    if not metas:
        return None
    return json.loads(max(metas, key=lambda p: p.stat().st_mtime).read_text())


def load_or_run(spec: StageSpec, cache_root: Path, fn: Callable[[Path], dict] | None, *,
                use_cache: bool = True, memo: HashMemo | None = None, log=print) -> Entry:
    """Return the verified entry for ``spec``. Runs ``fn(work_dir)`` when the
    entry is missing (or ``use_cache`` is False); ``fn=None`` means the
    caller forbids running and a missing entry raises StaleCacheError,
    naming what changed against the latest cached entry of the stage."""
    memo = memo or HashMemo(Path(cache_root) / HASH_MEMO)
    comps = stage_components(spec, memo)
    memo.save()
    key = key_of(comps)
    stage_dir = Path(cache_root) / spec.name
    entry_dir = stage_dir / key
    meta_p = entry_dir / META

    if use_cache and meta_p.is_file():
        meta = json.loads(meta_p.read_text())
        verify(meta, entry_dir, comps)
        log(f"[stage_cache] {spec.name}: cache hit {key[:12]}")
        return Entry(spec.name, key, entry_dir, meta, ran=False)
    if fn is None:
        other = _latest_other(stage_dir, key)
        why = (f"; latest cached entry ({other['key'][:12]}) differs in: "
               + ", ".join(_diff_components(other.get("components", {}), comps))) if other else "; the stage was never cached"
        raise StaleCacheError(f"stage {spec.name}: no cache entry for key {key[:12]} under {stage_dir}{why}. "
                              "Run the compute stage first (--stage compute or all).")

    stage_dir.mkdir(parents=True, exist_ok=True)
    work = stage_dir / f".tmp-{key[:12]}-{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    t0 = time.time()
    returned = fn(work) or {}
    seconds = time.time() - t0
    objects: dict[str, str] = {}
    if returned:
        (work / OBJECTS).mkdir()
        for name, obj in returned.items():
            objects[name] = f"{OBJECTS}/{_store_object(work / OBJECTS, name, obj)}"
    artefacts = {p.relative_to(work).as_posix(): sha256_file(p) for p in sorted(work.rglob("*")) if p.is_file()}
    head, dirty = git_state(spec.repo_root)
    meta = {
        "stage": spec.name, "key": key, "components": comps, "git_head": head, "dirty": dirty,
        "seconds": round(seconds, 2), "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artefacts": artefacts, "objects": objects,
    }
    (work / META).write_text(json.dumps(meta, indent=2))
    if entry_dir.exists():
        shutil.rmtree(entry_dir)
    work.rename(entry_dir)
    log(f"[stage_cache] {spec.name}: computed {key[:12]} in {seconds:.1f} s")
    return Entry(spec.name, key, entry_dir, meta, ran=True)


def place(src: Path, dest: Path) -> None:
    """Hard-link src to dest (same filesystem), else copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)
