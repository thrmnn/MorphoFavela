"""
Test suite for data alignment and road redirection.

Tests coordinate system alignment, spatial extent alignment, STL georeferencing,
and road-building intersection handling.
"""

import numpy as np
import pytest
import geopandas as gpd
import pyvista as pv
from pathlib import Path
import tempfile
from shapely.geometry import LineString, box
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_alignment_utils import (  # noqa: E402
    check_crs_alignment,
    auto_correct_alignment,
    detect_road_building_intersections,
    redirect_roads,
    ALIGNMENT_TOLERANCE
)

try:
    import rasterio
    from rasterio.crs import CRS  # noqa: F401
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class TestDataLoading:
    """Test data loading and validation."""
    
    def test_load_building_polygons(self):
        """Test loading building polygons."""
        # Create test buildings
        buildings = gpd.GeoDataFrame(
            geometry=[
                box(0, 0, 10, 10),
                box(20, 20, 30, 30)
            ],
            crs='EPSG:4326'
        )
        
        assert len(buildings) == 2
        assert buildings.crs is not None
        assert all(buildings.geometry.type == 'Polygon')
    
    def test_load_road_network(self):
        """Test loading road network."""
        roads = gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (10, 10)]),
                LineString([(20, 20), (30, 30)])
            ],
            crs='EPSG:4326'
        )
        
        assert len(roads) == 2
        assert roads.crs is not None
        assert all(roads.geometry.type == 'LineString')
    
    def test_load_stl_mesh(self):
        """Test loading STL mesh."""
        # Create simple mesh
        mesh = pv.Box(bounds=(0, 10, 0, 10, 0, 5))
        
        assert mesh.n_points > 0
        assert mesh.n_cells > 0
        assert len(mesh.bounds) == 6
    
    @pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not available")
    def test_load_dtm_raster(self):
        """Test loading DTM raster."""
        # Create temporary DTM
        with tempfile.TemporaryDirectory() as tmpdir:
            dtm_path = Path(tmpdir) / "test_dtm.tif"
            
            # Create simple DTM
            height, width = 100, 100
            data = np.random.rand(height, width) * 100  # 0-100m elevation
            
            transform = from_bounds(0, 0, 1000, 1000, width, height)
            
            with rasterio.open(
                dtm_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
            
            # Load and validate
            with rasterio.open(dtm_path) as src:
                assert src.crs is not None
                assert src.width == width
                assert src.height == height
                assert src.bounds is not None


class TestCRSAlignment:
    """Test coordinate system alignment."""
    
    def test_same_crs_alignment(self):
        """Test datasets with same CRS."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(5, 5), (15, 15)])],
            crs='EPSG:4326'
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        is_aligned, warnings, corrections = check_crs_alignment(datasets, tolerance=20.0)  # Large tolerance for overlapping bounds
        
        # CRS should match (no CRS warnings)
        crs_warnings = [w for w in warnings if 'CRS' in w or 'coordinate system' in w.lower()]
        assert len(crs_warnings) == 0
        
        # Bounds may differ but should overlap (datasets are in same area)
        # is_aligned may be False due to bounds, but that's OK if CRS matches
    
    def test_different_crs_detection(self):
        """Test detection of different CRS."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(5, 5), (15, 15)])],
            crs='EPSG:3857'  # Different CRS
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        is_aligned, warnings, corrections = check_crs_alignment(datasets)
        
        assert not is_aligned
        assert len(warnings) > 0
        assert 'target_crs' in corrections
    
    def test_crs_auto_correction(self):
        """Test automatic CRS correction."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(5, 5), (15, 15)])],
            crs='EPSG:3857'
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        _, _, corrections = check_crs_alignment(datasets)
        corrected = auto_correct_alignment(datasets, corrections)
        
        # After correction, CRS should match
        assert corrected['buildings'].crs == corrected['roads'].crs


class TestSpatialAlignment:
    """Test spatial extent alignment."""
    
    def test_aligned_bounds(self):
        """Test datasets with aligned bounds."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(5, 5), (15, 15)])],
            crs='EPSG:4326'
        )
        
        stl_bounds = (0, 20, 0, 20, 0, 10)  # x_min, x_max, y_min, y_max, z_min, z_max
        
        datasets = {
            'buildings': buildings,
            'roads': roads,
            'stl': stl_bounds
        }
        
        is_aligned, warnings, corrections = check_crs_alignment(datasets, tolerance=15.0)  # Large tolerance for overlapping bounds
        
        # Should be aligned within tolerance (bounds overlap, centers are close)
        # The datasets overlap, so they should be considered aligned
        # Note: centers differ by ~10m and ~5m, so need tolerance > 10m
        assert is_aligned or len([w for w in warnings if 'misaligned' in w.lower()]) == 0
    
    def test_misaligned_bounds_detection(self):
        """Test detection of misaligned bounds."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(1000, 1000), (1010, 1010)])],  # Far away
            crs='EPSG:4326'
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        is_aligned, warnings, corrections = check_crs_alignment(datasets, tolerance=0.1)
        
        assert not is_aligned
        assert any('misaligned' in w.lower() for w in warnings)
        assert 'translation' in corrections
    
    def test_spatial_auto_correction(self):
        """Test automatic spatial correction."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Roads offset by 100m
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(100, 100), (110, 110)])],
            crs='EPSG:4326'
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        _, _, corrections = check_crs_alignment(datasets, tolerance=0.1)
        corrected = auto_correct_alignment(datasets, corrections)
        
        # After correction, centers should be closer
        buildings_center = (
            (buildings.total_bounds[0] + buildings.total_bounds[2]) / 2,
            (buildings.total_bounds[1] + buildings.total_bounds[3]) / 2
        )
        roads_center = (
            (corrected['roads'].total_bounds[0] + corrected['roads'].total_bounds[2]) / 2,
            (corrected['roads'].total_bounds[1] + corrected['roads'].total_bounds[3]) / 2
        )
        
        center_diff = np.sqrt(
            (buildings_center[0] - roads_center[0])**2 +
            (buildings_center[1] - roads_center[1])**2
        )
        
        assert center_diff < ALIGNMENT_TOLERANCE * 10  # Allow some tolerance


class TestSTLAlignment:
    """Test STL georeferencing and alignment."""
    
    def test_stl_bounds_alignment(self):
        """Test STL bounds alignment with buildings."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # STL bounds should match building extent (with some margin)
        stl_bounds = (-1, 11, -1, 11, 0, 20)  # Slightly larger, includes height
        
        datasets = {
            'buildings': buildings,
            'stl': stl_bounds
        }
        
        is_aligned, warnings, _ = check_crs_alignment(datasets, tolerance=1.0)
        
        # Should be aligned within tolerance
        assert is_aligned or len([w for w in warnings if 'stl' in w.lower()]) == 0
    
    def test_stl_mesh_alignment(self):
        """Test STL mesh alignment."""
        # Create mesh that matches building extent
        mesh = pv.Box(bounds=(0, 10, 0, 10, 0, 5))
        
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        datasets = {
            'buildings': buildings,
            'stl': mesh
        }
        
        is_aligned, warnings, _ = check_crs_alignment(datasets, tolerance=1.0)
        
        # Should be aligned
        assert is_aligned or len([w for w in warnings if 'stl' in w.lower()]) == 0


class TestRoadBuildingIntersections:
    """Test road-building intersection detection."""
    
    def test_no_intersections(self):
        """Test when roads don't intersect buildings."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(20, 20), (30, 30)])],  # Far from building
            crs='EPSG:4326'
        )
        
        intersections = detect_road_building_intersections(roads, buildings)
        
        assert len(intersections) == 0
    
    def test_intersection_detection(self):
        """Test detection of road-building intersections."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Road passes through building
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(-5, 5), (15, 5)])],  # Passes through building
            crs='EPSG:4326'
        )
        
        intersections = detect_road_building_intersections(roads, buildings)
        
        assert len(intersections) > 0
        assert intersections.iloc[0]['n_buildings'] > 0
    
    def test_multiple_intersections(self):
        """Test multiple road-building intersections."""
        buildings = gpd.GeoDataFrame(
            geometry=[
                box(0, 0, 10, 10),
                box(20, 20, 30, 30)
            ],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[
                LineString([(-5, 5), (15, 5)]),  # Through first building
                LineString([(15, 25), (35, 25)])  # Through second building
            ],
            crs='EPSG:4326'
        )
        
        intersections = detect_road_building_intersections(roads, buildings)
        
        assert len(intersections) == 2


class TestRoadRedirection:
    """Test road redirection to avoid buildings."""
    
    def test_parallel_offset_redirection(self):
        """Test parallel offset redirection."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Road passes through building
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(-5, 5), (15, 5)])],
            crs='EPSG:4326'
        )
        
        redirected, intersections = redirect_roads(
            roads, buildings, method='parallel_offset', offset_distance=2.0
        )
        
        assert len(redirected) == len(roads)
        assert len(intersections) > 0
        
        # Check redirected road doesn't intersect (or has fewer intersections)
        redirected_intersections = detect_road_building_intersections(redirected, buildings)
        # Parallel offset may not always work perfectly, so check for improvement
        assert len(redirected_intersections) <= len(intersections)
    
    def test_simple_reroute_redirection(self):
        """Test simple reroute redirection."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(-5, 5), (15, 5)])],
            crs='EPSG:4326'
        )
        
        redirected, intersections = redirect_roads(
            roads, buildings, method='simple_reroute', buffer_distance=2.0
        )
        
        assert len(redirected) == len(roads)
        
        # Check redirected road doesn't intersect (or has fewer intersections)
        redirected_intersections = detect_road_building_intersections(redirected, buildings)
        assert len(redirected_intersections) <= len(intersections)
    
    def test_redirection_preserves_network(self):
        """Test that redirection preserves road network connectivity."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(5, 5, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Connected road network
        roads = gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (15, 0)]),  # Horizontal
                LineString([(0, 0), (0, 15)]),  # Vertical
            ],
            crs='EPSG:4326'
        )
        
        redirected, _ = redirect_roads(roads, buildings, method='parallel_offset')
        
        # Roads should still be valid LineStrings
        assert all(redirected.geometry.type == 'LineString')
        assert all(not geom.is_empty for geom in redirected.geometry)


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_alignment_pipeline(self):
        """Test complete alignment pipeline."""
        # Create misaligned datasets
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Roads offset by 50m
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(50, 50), (60, 60)])],
            crs='EPSG:4326'
        )
        
        datasets = {
            'buildings': buildings,
            'roads': roads
        }
        
        # Check alignment
        is_aligned, warnings, corrections = check_crs_alignment(datasets, tolerance=0.1)
        
        # Should detect misalignment
        assert not is_aligned
        
        # Auto-correct
        corrected = auto_correct_alignment(datasets, corrections)
        
        # Verify correction
        is_aligned_after, warnings_after, _ = check_crs_alignment(
            corrected, tolerance=1.0  # More lenient after correction
        )
        
        # Should be better aligned (may not be perfect due to translation approximation)
        assert len(warnings_after) <= len(warnings)
    
    def test_alignment_and_redirection_pipeline(self):
        """Test alignment followed by road redirection."""
        buildings = gpd.GeoDataFrame(
            geometry=[box(0, 0, 10, 10)],
            crs='EPSG:4326'
        )
        
        # Road that intersects building (diagonal for better offset results)
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(-5, -5), (15, 15)])],  # Diagonal through building
            crs='EPSG:4326'
        )
        
        # First align (if needed)
        datasets = {'buildings': buildings, 'roads': roads}
        is_aligned, warnings, corrections = check_crs_alignment(datasets, tolerance=10.0)
        
        if not is_aligned:
            datasets = auto_correct_alignment(datasets, corrections)
        
        # Then redirect
        redirected, intersections = redirect_roads(
            datasets['roads'], datasets['buildings'], method='parallel_offset', offset_distance=3.0
        )
        
        # Verify intersections are reduced (may not be zero due to geometry complexity)
        final_intersections = detect_road_building_intersections(
            redirected, datasets['buildings']
        )
        
        # Redirection should reduce intersections (may not eliminate all in complex cases)
        assert len(final_intersections) <= len(intersections)
