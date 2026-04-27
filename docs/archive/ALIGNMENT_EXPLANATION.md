# Current Alignment Process Explanation

## Current Approach (INCORRECT)

### What I'm Currently Doing:
1. **STL**: Loaded in local coordinates (e.g., bounds: -597 to 597, centered around origin)
2. **DTM/Buildings/Roads**: Loaded in georeferenced coordinates (e.g., EPSG:31983, bounds: 679703 to 680878)
3. **Transformation**: I transform roads/buildings FROM world coordinates TO STL-local coordinates
   - Calculate: `dx = stl_center - world_center` (e.g., `dx = 0 - 680291 = -680291`)
   - Apply: `roads.translate(xoff=dx, yoff=dy)` to move roads into STL space
4. **Result**: Everything is in STL-local coordinates for computation
5. **Output**: Transform results back to world coordinates

### Problems with This Approach:
- ❌ Georeferenced data loses its coordinate system context
- ❌ DTM can't be used properly (it's georeferenced, we should use it as-is)
- ❌ Harder to validate alignment (comparing local vs world coordinates)
- ❌ More complex transformation chain (world → local → world)

## Correct Approach (What You're Suggesting)

### What Should Happen:
1. **DTM/Buildings/Roads**: Keep in georeferenced coordinates (EPSG:31983)
2. **STL**: Transform FROM local coordinates TO georeferenced coordinates
   - Calculate: `dx = world_center - stl_center` (e.g., `dx = 680291 - 0 = 680291`)
   - Apply transformation to STL mesh vertices
3. **Result**: Everything is in the same georeferenced coordinate system
4. **Computation**: All in world coordinates
5. **Output**: Already in world coordinates, no back-transformation needed

### Benefits:
- ✅ All data stays in georeferenced coordinate system
- ✅ DTM can be used directly for elevation extraction
- ✅ Easier to validate alignment (all in same CRS)
- ✅ Simpler transformation (only STL needs transformation)
- ✅ More intuitive and correct geospatial workflow

## Questions to Clarify:

1. **STL Mesh Transformation**: Should I transform the PyVista mesh vertices directly?
   - Transform all mesh points: `mesh.points[:, 0] += dx`, `mesh.points[:, 1] += dy`?
   - Or create a new mesh with transformed coordinates?

2. **DTM Usage**: When DTM is provided, should it be the PRIMARY reference for alignment?
   - Use DTM bounds to determine where STL should be positioned?
   - Or use DTM + Buildings + Roads together to determine alignment?

3. **Alignment Method**: How should we match STL to georeferenced data?
   - **Option A**: Match centers (current approach, but in reverse direction)
   - **Option B**: Match bounds (align min/max corners)
   - **Option C**: Match using building footprints (since STL was created from footprints)

4. **Validation**: How should we validate alignment?
   - Check that STL bounds overlap with DTM/building/road bounds?
   - Check that building footprints align with STL building geometry?
   - Visual inspection in debug plot?

5. **Terrain Extraction**: After transforming STL, should terrain extraction still work the same way?
   - Or does terrain extraction need to account for the transformation?

Please clarify these points so I can implement the correct alignment methodology.
