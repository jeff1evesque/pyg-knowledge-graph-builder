# Memory and Scaling

What the driver holds while a job runs, and how that grows with the graph.

## Build / Train / Inference Lifecycle

```
BUILD TIME (this codebase — pyg_builder)
│
├── graph_schema.json     ← constructor.py after HeteroData assembly
├── feature_spec.json     ← VectorLayout.to_dict() + EdgeVectorLayout.to_dict()
├── normalization.json    ← feature_extractor normalization stats pass
├── encoding_config.json  ← feature_extractor + edge_feature_extractor configs
├── ontology_schema.json  ← feature_extractor ontology structure collection
└── slot_mapping.json     ← feature_extractor hash slot computation
│
▼
TRAIN TIME (downstream GNN training code)
│
├── Reads graph_schema.json  → decides which node/edge types to include,
│                              sets up data splits, validates .pt after loading
├── Reads feature_spec.json  → builds model architecture (layer dims,
│                              conv routing, segment projections)
├── Loads hetero_data.pt     → validates against graph_schema.json
└── Trains model             → saves checkpoint + references metadata path
│
▼
INFERENCE TIME (deployed model serving)
│
├── Reads feature_spec.json    → reconstructs identical model architecture
├── Reads encoding_config.json → configures feature encoder with same hash params
├── Reads normalization.json   → applies same z-score stats to new data
├── Reads ontology_schema.json → encodes new nodes with correct class hierarchy
├── Reads graph_schema.json    → validates that new data produces compatible types
├── Loads model checkpoint     → loads trained weights into reconstructed architecture
└── Encodes new data → runs inference
```

## Driver Memory Impact

Metadata collection adds negligible driver memory overhead. All `collect()` calls during metadata collection target small aggregated DataFrames:

| Metadata collect | Rows collected | Timing |
|-----------------|---------------|--------|
| Node type URI mapping | <500 (one per type URI) | Step 1, after node_id_df is cached |
| Edge predicate URI mapping | <100 (one per relation) | Step 2, from cached edges_final_df |
| Normalization stats | <200 (one per predicate) | Step 3, single-pass agg |
| Type URI → PyG name mapping | <500 (distinct type URIs) | Step 3, from triples_df |
| Class hierarchy | <5000 (transitive closure) | Step 3, from class_hierarchy_df |
| Property schema | <500 (properties with domain/range) | Step 3, from property_schema_df |
| Numeric predicate list | <100 (distinct predicates) | Step 3, for slot mapping |
| Categorical predicate list | <100 (distinct predicates) | Step 3, for slot mapping |
| Class URI list | <500 (distinct type URIs) | Step 3, for slot mapping |
| Superclass URI list | <200 (distinct superclasses) | Step 3, for slot mapping |

No per-node or per-edge data is ever collected for metadata. The `MetadataCollector` object holds only small Python dicts — no tensors, no DataFrames, no Spark references. Total metadata memory is well under 1 MB.

## Driver Memory Safety

The pipeline is designed to prevent driver OOM even with millions of nodes per type:

```
Driver memory lifecycle during PyG construction:
═══════════════════════════════════════════════════

Step 1: Node ID table
  Driver holds: node_counts dict (~1 KB)
  Executors hold: node_id_df (cached)

Step 2: Edge indices (collected one type at a time)
  Driver holds: edge_indices dict (accumulating)
    Per type: [2, N] int64 → ~16 bytes/edge
    Total: ~200-500 MB for 15-30M edges
  Peak per-type: Pandas DataFrame + tensor → Pandas freed immediately
  Executors hold: edges_final_df (cached for Step 4)

Step 3: Feature tensors (collected one type at a time, largest first)
  Driver holds: feature_tensors dict (accumulating) + edge_indices
  Per type:
    a) Pre-allocate dense numpy: num_nodes × vector_dim × 4 bytes
    b) Collect sparse entries via toPandas():
       - Small types (<500K nodes): single collect, ~120 MB peak
       - Large types (>500K nodes): chunked by node_id range,
         ~120 MB per chunk
    c) Scatter into dense array (in-place, no copy)
    d) Delete Pandas DataFrame, gc.collect()
    e) Convert numpy → torch (zero-copy via from_numpy)

Step 4: Edge feature tensors (collected one type at a time)
  Driver holds: edge_feature_tensors dict (accumulating)
    + feature_tensors + edge_indices
  Per type:
    a) Pre-allocate dense numpy: num_edges × edge_vector_dim × 4 bytes
       (much smaller than node features: 32-d vs 1024-d)
    b) Collect sparse entries via toPandas():
       - Small types (<1M edges): single collect
       - Large types (>1M edges): chunked by edge_idx range
    c) Scatter into dense array, delete Pandas, gc.collect()
    d) Convert numpy → torch (zero-copy via from_numpy)
  Typical total: 32 dims × 4 bytes × 2M edges = ~256 MB per type
  Reuses cached edges_final_df — no additional executor memory

  gc.collect() is safe here because it runs on the driver process
  only — all Spark executor work is complete before collection.
  It reclaims Pandas/numpy circular references that CPython's
  reference counting alone may not free.

Step 5: Assemble HeteroData
  HeteroData stores references to existing tensors (no copy)
  Attach edge_attr for featurized edge types (reference only)
  Delete intermediate dicts → only HeteroData holds references
  Unpersist edges_final_df (executor cache freed)
  Unpersist node_id_df (executor cache freed)
  gc.collect() to reclaim dict overhead

Post-construction: Save outputs
  torch.save() → local .pt (and, when archiving, a BytesIO buffer
    streamed to S3 via upload_fileobj)
  Peak: HeteroData + serialized buffer (same size)
  Buffer freed after upload

  MetadataCollector.to_metadata_files() → six JSON dicts (<1 MB total)
  write_metadata_to_local() → six files locally
    (and write_metadata_to_s3() → six put_object calls when archiving)
  MetadataCollector holds only small Python dicts throughout
```

**Chunked collection for large node types**: When a node type has more than 500K nodes (configurable via `chunk_node_threshold`), the sparse `(node_id, dim, value)` entries are collected in chunks by node_id range. Each chunk's Pandas DataFrame is scattered into the pre-allocated dense array and immediately freed. This bounds peak Pandas memory to ~120 MB per chunk regardless of total type size.

**Chunked collection for large edge types**: When an edge type has more than 1M edges (configurable via `chunk_edge_threshold`), the sparse `(edge_idx, dim, value)` entries are collected in chunks by edge_idx range, following the same pattern as node features.

**Batched collection for small edge types**: Edge types at or below the threshold are unioned in bounded batches and collected with one grouped `toPandas()` per batch, then scattered into per-type tensors on the driver. Batches are capped on two independent axes — total edges (bounding driver memory) and type count via `max_edge_types_per_batch` (bounding Catalyst plan width).

**Shared frames are checkpointed, not cached**: `edges_with_idx`, the endpoint numeric properties, and the endpoint labels are materialized with `localCheckpoint(eager=True)` rather than `.cache()`. Caching materializes rows but leaves the logical plan intact, so every per-edge-type query re-analyzes the full enrichment lineage on the driver — driver CPU that is flat per edge type and independent of edge count. On the e2e fixtures this dominated the phase (139s → 14s once truncated). Same reasoning as `EnrichmentPipeline._settle`.

**Largest types processed first**: Node types are sorted by node count (descending) so that if a type is too large for available driver memory, the job fails fast rather than after processing all smaller types.

**Edge feature tensors are much smaller than node features**: At 32 dims × 4 bytes per edge vs. 1024 dims × 4 bytes per node, edge feature tensors are ~32× smaller per element. A 2M-edge type's tensor is ~256 MB at 32-d, compared to ~8 GB for a 2M-node type at 1024-d.

## Scaling Characteristics

| Component | Where it runs | Memory model |
|-----------|--------------|--------------|
| N-Triples parsing | Spark executors | `spark.read.text` + regex, no driver I/O |
| Node ID assignment | Spark executors | URI → int mapping via Window functions, cached on executors |
| URI-to-name conversion | Spark executors | Pure Spark `WHEN` expressions (JVM-native, no Python UDF) |
| Edge resolution | Spark executors | Double-join resolves URIs to ints on executors |
| Literal isolation | Spark executors | Anti-join against node_id_df filters out edge triples |
| Ontology structure extraction | Spark executors | `rdfs:subClassOf` transitive closure via iterative joins |
| Property schema extraction | Spark executors | `rdfs:domain`/`rdfs:range` join, cached on executors |
| VectorLayout computation | Driver (init) | Pure Python arithmetic from `vector_dim`, ~1 μs |
| EdgeVectorLayout computation | Driver (init) | Pure Python arithmetic from `edge_vector_dim`, ~1 μs |
| Hash-based node encoding | Spark executors | `F.hash()`, `F.abs()`, `F.lit()` — JVM-native, no Python UDF |
| Node feature normalization | Spark executors | Single-pass `agg()` for per-predicate stats, broadcast joined |
| Sparse entry aggregation (nodes) | Spark executors | `groupBy(node_id, dim).agg(sum)` — handles hash collisions |
| Edge type classification | Driver | Substring matching on relation names, ~1 μs per type |
| Edge idx assignment | Spark executors | `Window.partitionBy(...).orderBy("src_id", "dst_id")` — deterministic |
| Endpoint property extraction | Spark executors | Anti-join + cast + groupBy — reuses literal isolation pattern |
| Temporal signal encoding | Spark executors | Regex-matched month/year properties, arithmetic on executors |
| Numeric contrast encoding | Spark executors | Inner join on shared predicates, difference/ratio/magnitude |
| Cross-property derivation | Spark executors | Regex-matched property names (strikePrice, observedPrice, etc.) |
| Label similarity (Jaccard) | Spark executors | `F.array_intersect` / `F.array_union` — JVM-native array ops |
| Sparse entry aggregation (edges) | Spark executors | `groupBy(edge_idx, dim).agg(sum)` — handles hash collisions |
| Edge index collection | Driver | Per-edge-type [2, N] int64 — ~16 bytes/edge |
| Node feature collection | Driver | Per-type sparse entries → dense [N, vector_dim] float32, chunked for large types |
| Edge feature collection | Driver | Per-type sparse entries → dense [N, edge_vector_dim] float32, chunked for large types |
| HeteroData assembly | Driver | Only compact tensors, no strings |
| Metadata collection | Driver | Small aggregated DataFrames only (<5000 rows per collect), <1 MB total |
| Metadata serialization | Driver | `json.dumps()` on small Python dicts; six local writes (+ `put_object` calls when archiving) |
| Enriched Parquet write | Spark executors | `repartition` + `write.parquet` — executors write directly to the shared local dir |
| PyG .pt write | Driver | `torch.save` to a `BytesIO` buffer, then written by scheme — direct local I/O for a POSIX `--local_work_dir`, Hadoop FileSystem for an `s3a://` one (+ `upload_fileobj` streaming to S3 when archiving) |

**Universal node feature width**: HeteroData stores node feature tensors of the same `vector_dim` for every node type. The ontology-aware encoding keeps vectors informative even for types with few literal properties — the ontology structure and property schema segments still carry meaningful signal.

**Selective edge feature width**: HeteroData stores edge feature tensors (`edge_attr`) only for edge types that received features. Edge types without features have no `edge_attr` attribute, allowing the GNN to use simpler message-passing layers for those types.

| Node type example | Typical nodes | Memory @ 1024-d | Memory @ 512-d | Key signals |
|-------------------|--------------|-----------------|----------------|-------------|
| cpi_Index | ~50K | ~200 MB | ~100 MB | CPI class hierarchy, index/change properties, BLS source |
| market_EquitySnapshot | ~500K-1M | ~2-4 GB | ~1-2 GB | Equity class, price/volume/52wk properties, market source |
| market_OptionSnapshot | ~1-2M | ~4-8 GB | ~2-4 GB | Option subclass, strike/expiry/greeks/underlying properties |
| jolts_JobOpeningsLevel | ~10K | ~40 MB | ~20 MB | JOLTS hierarchy, level/rate properties, BLS source |
| filings_Form4 | ~50K | ~200 MB | ~100 MB | SEC filing class, transaction properties, SEC source |
| unified_UnifiedMonth | ~12 | ~48 KB | ~24 KB | Temporal class, cross-source membership |

| Edge type example | Typical edges | Memory @ 32-d | Key signals |
|-------------------|--------------|---------------|-------------|
| (cpi_Index, precedes, cpi_Index) | ~50K | ~6 MB | Month delta, same-year, direction |
| (market_EquitySnapshot, precedes, market_EquitySnapshot) | ~500K-1M | ~64-128 MB | Intraday time delta, consecutive flag |
| (market_OptionSnapshot, hasUnderlyingEquity, market_EquitySnapshot) | ~500K-1M | ~64-128 MB | Moneyness, log-moneyness, strike-stock diff |
| (noaa_Alert, escalatesTo, noaa_Alert) | ~1K | ~128 KB | Severity delta |
| (cpi_Index, belongsToSector, unified_EconomicSector) | ~5K | — (no features) | Relation name sufficient |
| (cpi_Index, correlatesWith, ppi_Index) | ~10K | — (OFF by default) | Label similarity, cross-source flag |

## Memory Budget by Driver Size

The final `HeteroData` assembly happens on the driver, so driver memory
bounds the graph size. Approximate budgets:

```
~32 GB driver:
  JVM + Spark overhead:     ~8-10 GB
  Python interpreter:       ~1-2 GB
  Available for tensors:    ~20-22 GB
  → Node features: suitable for <1M total nodes at 1024-d
  → Edge features: adds ~0.5-1 GB for typical temporal + option edges at 32-d
  → Or <2M total nodes at 512-d with edge features
  → Metadata: <1 MB, negligible

~64 GB driver:
  JVM + Spark overhead:     ~10-12 GB
  Python interpreter:       ~1-2 GB
  Available for tensors:    ~50-52 GB
  → Node features: suitable for 2-5M total nodes at 1024-d
  → Edge features: adds ~1-3 GB for all featurized edge types at 32-d
  → Recommended for production with intraday market data
  → Metadata: <1 MB, negligible
```

Reducing `vector_dim` from 1024 to 512 **halves driver memory** for node feature tensors while preserving the same three-segment structure. Edge feature tensors at 32-d are already compact — reducing `edge_vector_dim` to 16 halves their memory but is rarely necessary since they are ~32× smaller per element than node features. This enables rapid experimentation at reduced resolution before committing to full-resolution production runs.

