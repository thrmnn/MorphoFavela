"""3D-print model generation from CFD analysis patches.

Turns a 100 m analysis patch (buildings.gpkg + terrain.tif) into a single
watertight, scaled STL ready to slice for resin printing. Distinct from
``svf_v2.scene`` (visualization surface meshes at 1:1 world coordinates):
here the terrain is a closed solid plinth, buildings are embedded into it so
nothing floats, and the result is translated to a local origin and scaled to
model millimetres.
"""

from .model import build_print_model, heightmap_to_solid

__all__ = ["build_print_model", "heightmap_to_solid"]
