# Favela Data Extraction Workflow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    City-Wide Dataset                        │
│                  data/RJ/                                   │
├─────────────────────────────────────────────────────────────┤
│  • buildings_RJ_2019.shp (2.36M buildings)                 │
│  • Favelas_Limit_2019.shp (1,074 favelas)                   │
│  • Logradouros.shp (134K streets)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: Interactive Exploration               │
│              notebooks/explore_favelas.ipynb                 │
├─────────────────────────────────────────────────────────────┤
│  1. Load favela boundaries                                  │
│  2. Build spatial index (cache)                             │
│  3. Pre-compute statistics (buildings, streets per favela)   │
│  4. Interactive map (folium/ipyleaflet)                     │
│  5. Searchable table with metadata                           │
│  6. Selection interface                                      │
│  7. Export selected list (JSON/CSV)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    [User Selection]
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: Data Extraction                       │
│              scripts/extract_favela_data.py                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Spatial Index Builder                              │    │
│  │  • Build R-tree index for buildings                │    │
│  │  • Build R-tree index for streets                   │    │
│  │  • Cache to disk (reuse across favelas)             │    │
│  └────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │  For Each Favela:                                    │    │
│  │                                                      │    │
│  │  1. Extract Buildings                                │    │
│  │     • Query spatial index                            │    │
│  │     • Filter with within/intersects                  │    │
│  │     • Validate geometry                              │    │
│  │                                                      │    │
│  │  2. Extract Streets                                  │    │
│  │     • Query spatial index                            │    │
│  │     • Apply buffer (50m)                              │    │
│  │     • Filter by intersection %                     │    │
│  │                                                      │    │
│  │  3. Save Results                                     │    │
│  │     • Write to data/{favela}/raw/                    │    │
│  │     • Generate metadata                              │    │
│  │     • Create README                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Batch Processing (Optional)                         │    │
│  │  • Parallel extraction (multiprocessing)             │    │
│  │  • Progress tracking (tqdm)                         │    │
│  │  • Error handling per favela                         │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Structure                          │
│                                                              │
│  data/{favela_name}/                                        │
│  ├── raw/                                                   │
│  │   ├── {favela_name}_buildings.shp                       │
│  │   ├── roads_{favela_name}.shp                          │
│  │   └── {favela_name}_boundary.shp (optional)            │
│  └── README.md (metadata)                                   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
City Buildings (2.36M)
         │
         │ Spatial Index (R-tree)
         │
         ▼
    ┌─────────┐
    │ Query   │ ← Favela Boundary
    │ Index   │
    └─────────┘
         │
         ▼
    Filtered Buildings
    (within boundary)
         │
         ▼
    Extract & Save
    → {favela}_buildings.shp

─────────────────────────────────────

City Streets (134K)
         │
         │ Spatial Index (R-tree)
         │
         ▼
    ┌─────────┐
    │ Query   │ ← Favela Boundary + Buffer (50m)
    │ Index   │
    └─────────┘
         │
         ▼
    Filtered Streets
    (intersect buffered boundary)
         │
         ▼
    Extract & Save
    → roads_{favela}.shp
```

## Performance Optimization Flow

```
Without Optimization:
┌─────────────────┐
│ 2.36M buildings  │
│      │          │
│      ▼          │
│ Check each one  │ → O(n) = 2.36M operations
│ against boundary│
└─────────────────┘
Time: 5-30 minutes per favela

With Spatial Index:
┌─────────────────┐
│ 2.36M buildings  │
│      │          │
│      ▼          │
│ Build R-tree    │ → One-time cost: ~30 seconds
│      │          │
│      ▼          │
│ Query index     │ → O(log n) = ~22 operations
│      │          │
│      ▼          │
│ Get candidates  │ → ~100-10K buildings
│      │          │
│      ▼          │
│ Filter exact    │ → O(m) where m << n
└─────────────────┘
Time: 5-30 seconds per favela
```

## Implementation Checklist

### Notebook (explore_favelas.ipynb)
- [ ] Load favela boundaries
- [ ] Build/cache spatial index
- [ ] Pre-compute statistics
- [ ] Create interactive map
- [ ] Add searchable table
- [ ] Selection interface
- [ ] Export functionality

### Extraction Script (extract_favela_data.py)
- [ ] Spatial index builder
- [ ] Building extraction function
- [ ] Street extraction function
- [ ] Single favela extraction
- [ ] Batch extraction
- [ ] Error handling
- [ ] Progress tracking
- [ ] Metadata generation

### Configuration
- [ ] Config file (src/favela_config.py)
- [ ] Parameter documentation
- [ ] Default values

### Testing
- [ ] Test on known favela (Vidigal)
- [ ] Validate output structure
- [ ] Check data quality
- [ ] Performance benchmark
- [ ] Error scenarios

## Usage Scenarios

### Scenario 1: Manual Selection (Notebook)
```
1. Open notebook
2. Explore favelas on map
3. Search/filter by criteria
4. Select favelas interactively
5. Export selection list
6. Run extraction script with list
```

### Scenario 2: Single Favela (CLI)
```bash
python scripts/extract_favela_data.py \
    --favela "Vidigal" \
    --output-dir data/vidigal_tls/raw
```

### Scenario 3: Batch Extraction (CLI)
```bash
python scripts/extract_favela_data.py \
    --favela-list selected_favelas.json \
    --parallel \
    --workers 4
```

### Scenario 4: All Favelas (CLI)
```bash
python scripts/extract_favela_data.py \
    --all \
    --output-base data \
    --parallel \
    --workers 8
```
