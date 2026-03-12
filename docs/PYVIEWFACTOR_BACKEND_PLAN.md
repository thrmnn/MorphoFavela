# PyViewFactor SVF Backend – Problem Statement & Proposed Solution

This document describes the current issue with the experimental PyViewFactor (pvf) SVF backend, and a proposed strategy to make it robust and independent of the original CPU implementation.

The goal is **not implemented fully yet** – this is a design plan to guide the next iteration.

---

## 1. Context

We currently support three SVF backends via `src.svf_compute.compute_svf`:

- **Original CPU** – PyVista ray-casting against the STL mesh
- **GPU-accelerated** – PyTorch3D-based ray tracing (`svf_gpu_compute`)
- **PyViewFactor** – experimental backend using the `pyviewfactor` library

The PyViewFactor backend is implemented in `compute_svf_pyvf(...)` and is reachable via:

```python
compute_svf(..., backend="pyviewfactor")
```

We have synthetic tests that exercise this backend (empty scene, single building, two buildings) and they pass. However, when we run the Vidigal SVF analysis, we see repeated warnings:

> `PyViewFactor computation failed at point N, falling back to CPU SVF. Error: division by zero`

These messages are emitted from our `compute_svf_pyvf` wrapper when `pyviewfactor.batch_compute_viewfactors(...)` raises a `ZeroDivisionError`.

Currently, the backend catches this error **per point** and falls back to the original CPU method for that observer, which hides the underlying geometry/configuration problem.

---

## 2. Problem

### 2.1 Symptom

For many observer points in a real scene (Vidigal STL + footprints), the PyViewFactor call fails:

- `batch_compute_viewfactors(pts1_arr, nverts1, pts2_arr, nverts2, const_arr)` raises `ZeroDivisionError`.
- Our code logs the message and uses the CPU SVF for that point.

### 2.2 Likely cause

PyViewFactor assumes reasonably well-behaved **planar polygon pairs**:

- Non-degenerate polygons (non-zero area)
- No overlapping or coincident polygons
- Reasonable separation between facets for the analytical integrals

Our current integration does **not** enforce these invariants:

- We take all above-ground STL faces as “obstruction facets” without filtering:
  - Some faces may be degenerate or extremely small (zero/near-zero area).
  - Faces may be nearly coplanar with the ground or with each other.
- For each observer:
  - We build a small horizontal patch centered at the observer.
  - This patch can **overlap or coincide** with building faces when the observer is very close to or numerically on a façade or roof.

Such configurations commonly cause:

- Division by zero in view factor formulas (e.g., distance → 0).
- Numerical instabilities in the quadrature/integration routines inside PyViewFactor.

### 2.3 Why the current fallback is not ideal

The per-point CPU fallback means:

- The pvf backend is **not truly independent** – results can be a mix of pvf and CPU ray-tracing behaviour.
- Geometry / configuration problems are **hidden** instead of being surfaced early.
- It’s harder to debug or compare pvf vs CPU/GPU on real scenes.

We want pvf to be:

- A **strict backend**: either the geometry/observers satisfy pvf assumptions and the run succeeds, or we fail fast with a clear error.
- Not silently “fixed” by another backend.

---

## 3. Proposed Solution (High-Level)

The fix has two main components:

1. **Mesh preprocessing** – ensure the 3D scene is “pvf-safe” *before* any computation.
2. **Observer preprocessing** – ensure observer positions and receiver patches don’t overlap or coincide with obstructions.

When both checks pass, `compute_svf_pyvf` runs **only PyViewFactor**; if anything fails during computation, we treat it as a hard error, not as a reason to fall back to CPU.

If either check detects invalid conditions, the pvf backend should **fail fast** with a descriptive error message.

---

## 4. Mesh Preprocessing for PyViewFactor

Introduce a dedicated function, e.g.:

```python
def preprocess_mesh_for_pyvf(full_mesh: pv.PolyData) -> list[np.ndarray]:
    """
    Extract above-ground obstruction polygons from the STL mesh, suitable for
    use with PyViewFactor. Raise if the mesh is too degenerate or otherwise
    unsafe for pvf-based integration.
    """
```

Responsibilities:

### 4.1 Face extraction

- Parse `mesh.faces` into individual polygons:
  - For each face: `[n_verts, i0, i1, ..., i_(n-1)]`
  - Convert to `poly = points[vertex_ids]` of shape `(n, 3)`.

### 4.2 Ground vs obstructions

- Estimate ground level:

  ```python
  ground_z = np.min(points[:, 2])
  z_tol = 1e-3
  ```

- Classify a polygon as **ground** if `max(poly[:, 2]) <= ground_z + z_tol`.
- Everything else is an obstruction candidate.

### 4.3 Drop degenerate facets

- For each obstruction polygon, compute its area, e.g.:

  ```python
  from pyviewfactor import polygon_area
  area = polygon_area(poly.astype(np.float64))
  ```

- Drop polygons with `area < AREA_EPS` (e.g. `1e-4 m²`).
- If:
  - No obstructions remain, we can trivially return SVF = 1 everywhere.
  - A large fraction of faces are degenerate (e.g. >50%) → raise:

    > `RuntimeError("PyViewFactor: too many degenerate faces; cannot safely run pvf")`

### 4.4 Optional cleanup

- Remove exact duplicates (same vertices within a small tolerance).
- Potentially enforce consistent winding/orientation (useful but not strictly required).

**Outcome**: A clean list of above-ground obstruction polygons for pvf, or an **early error** if the mesh is unsuitable.

---

## 5. Observer Preprocessing

Before calling pvf, we must guarantee that, for every observer point:

> The receiver patch around the observer is strictly separated from all obstruction facets by at least a small distance ε.

If an observer violates this invariant (e.g. on a façade or inside a building), we should **reject the configuration** before starting pvf.

### 5.1 Distance and inside-building checks

For each ground point (or in batches):

1. **Inside-building check** (2D footprint / 3D polygon test):
   - Using building footprints or projection of obstruction polygons onto XY:
     - If observer XY lies inside a building footprint → observer is effectively inside a building volume.
     - Strict behaviour: raise an error like:

       > `"PyViewFactor: some observer points are inside buildings; adjust ground mask or use another backend."`

2. **Proximity to obstruction facets**:
   - Precompute Axis-Aligned Bounding Boxes (AABBs) for obstruction polygons.
   - For each observer:
     - Find nearby facets via AABB overlap + simple distance estimate to polygon / plane.
     - If distance < ε (e.g. `0.2 m`):
       - The receiver patch of size `patch_size` will nearly intersect or coincide with that facet.
       - Again, strict behaviour: either
         - abort pvf for the entire run, or
         - mark such observers as unsupported for this backend (e.g. NaN), but **do not** silently use CPU.

### 5.2 Patch size and vertical offset

To further reduce overlap risk:

- Use a **small patch size**, e.g. `0.5 × 0.5 m` instead of `1.0 × 1.0 m`.
- Slightly raise the patch above the observer height:

  ```python
  patch_z = ground_z + evaluation_height + 0.1  # small vertical offset
  ```

This makes it less likely that the receiver patch lies exactly on the same plane as a roof or terrace.

---

## 6. Backend Behaviour without Fallback

After both preprocessing steps pass:

- `compute_svf_pyvf` should:
  - Use only PyViewFactor for view factor integrations.
  - No per-point try/except that falls back to CPU.

Pseudocode sketch:

```python
obstruction_polys = preprocess_mesh_for_pyvf(full_mesh)
check_observers_for_pyvf_safety(ground_points, obstruction_polys, evaluation_height)

for each observer in ground_points:
    build receiver patch
    build pts1_arr, nverts1 (receiver)
    use shared pts2_arr, nverts2 (obstructions)

    vf_vals = pvf.batch_compute_viewfactors(
        pts1_arr, nverts1, pts2_arr, nverts2, const_arr, verbose=False
    )

    F_tot = sum(vf_vals)
    svf[i] = clip(1 - F_tot, 0, 1)
```

If pvf raises an exception **after** these prechecks, that’s a real backend error and should be surfaced, not patched over.

---

## 7. Testing Strategy

To validate this solution we should extend the test suite as follows:

1. **Valid pvf runs**:
   - Existing synthetic tests (empty scene, single building, two buildings):
     - Ensure they pass with `backend="pyviewfactor"` after preprocessing.
     - Compare qualitative behaviour to CPU backend (high SVF in open sky, low under buildings, partial at edges).

2. **Expected failures** (geometry / observer violations):
   - New tests where inputs purposely violate pvf assumptions:
     - Observers inside buildings or directly on walls/roofs.
     - Mesh with intentionally degenerate faces (zero area, repeated vertices).
   - For `backend="pyviewfactor"`, assert that `compute_svf` raises a **clear RuntimeError** indicating:

     - Invalid mesh for pvf (too many degenerate facets), or
     - Invalid observer positions (inside/too close to obstructions for pvf backend).

This will ensure:

- The pvf backend is **strict and independent** (no hidden fallbacks).
- Geometry and configuration issues are caught early, with actionable error messages.

---

## 8. Summary

- The current `ZeroDivisionError` issues arise when PyViewFactor is given:
  - Degenerate or overlapping polygon pairs.
  - Receiver patches overlapping or nearly coincident with obstruction facets.
- The planned fix is to:
  - Add robust **mesh preprocessing** for pvf (extract, filter, validate obstruction facets).
  - Add **observer preprocessing** to ensure receiver patches are well-separated from obstructions.
  - Remove the per-point CPU fallback and treat pvf errors as proper failures for this backend.
- With these steps, the PyViewFactor backend becomes a well-defined, standalone SVF backend with clear preconditions and explicit failure modes, suitable for scientific comparison with the existing CPU/GPU implementations.

