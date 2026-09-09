# Memory and Scaling

What the driver holds while a job runs, and how that grows with the graph.

## Build / Train / Inference Lifecycle

**Build time** — this codebase, `pyg_builder`. Each metadata file and what writes it:

| file | written by |
|---|---|
| `graph_schema.json` | `constructor.py`, after HeteroData assembly |
| `feature_spec.json` | `VectorLayout.to_dict()` + `EdgeVectorLayout.to_dict()` |
| `normalization.json` | the `feature_extractor` normalization stats pass |
| `encoding_config.json` | `feature_extractor` + `edge_feature_extractor` configs |
| `ontology_schema.json` | `feature_extractor` ontology structure collection |
| `slot_mapping.json` | `feature_extractor` hash slot computation |

**Train time** — downstream GNN training code:

| reads | to |
|---|---|
| `graph_schema.json` | decide which node and edge types to include, set up data splits, and validate the `.pt` after loading |
| `feature_spec.json` | build the model architecture: layer dims, conv routing, segment projections |
| `hetero_data.pt` | load the graph, validated against `graph_schema.json` |
| — | train, then save a checkpoint referencing the metadata path |

**Inference time** — deployed model serving:

| reads | to |
|---|---|
| `feature_spec.json` | reconstruct an identical model architecture |
| `encoding_config.json` | configure the feature encoder with the same hash parameters |
| `normalization.json` | apply the same z-score stats to new data |
| `ontology_schema.json` | encode new nodes with the correct class hierarchy |
| `graph_schema.json` | validate that the new data produces compatible types |
| model checkpoint | load trained weights into the reconstructed architecture |
| — | encode new data, then run inference |

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

**Step 1: node ID table**

- Driver holds `node_counts`, a dict of about 1 KB.
- Executors hold `node_id_df`, cached.

**Step 2: edge indices**, collected one type at a time

- Driver holds the `edge_indices` dict, accumulating: `[2, N]` int64 per type, about 16 bytes an edge, so roughly 200-500 MB for 15-30M edges.
- Peak per type is a Pandas DataFrame plus the tensor, and the Pandas is freed immediately.
- Executors hold `edges_final_df`, cached for step 4.

**Step 3: feature tensors**, collected one type at a time, largest first

- Driver holds the `feature_tensors` dict, accumulating, plus `edge_indices`.
- Per type:
    1. Pre-allocate a dense numpy array of `num_nodes × vector_dim × 4` bytes.
    2. Collect sparse entries via `toPandas()` — types under 500K nodes in a single collect at about 120 MB peak, larger ones chunked by `node_id` range at about 120 MB a chunk.
    3. Scatter into the dense array in place, no copy.
    4. Delete the Pandas DataFrame and `gc.collect()`.
    5. Convert numpy to torch, zero-copy via `from_numpy`.

**Step 4: edge feature tensors**, collected one type at a time

- Driver holds the `edge_feature_tensors` dict, accumulating, plus `feature_tensors` and `edge_indices`.
- Per type:
    1. Pre-allocate a dense numpy array of `num_edges × edge_vector_dim × 4` bytes — much smaller than node features, 32-d against 1024-d.
    2. Collect sparse entries via `toPandas()` — types under 1M edges in a single collect, larger ones chunked by `edge_idx` range.
    3. Scatter into the dense array, delete the Pandas, `gc.collect()`.
    4. Convert numpy to torch, zero-copy via `from_numpy`.
- Typical total is 32 dims × 4 bytes × 2M edges, about 256 MB a type.
- Reuses the cached `edges_final_df`, so no additional executor memory.

`gc.collect()` is safe here because it runs on the driver process only — all
Spark executor work is complete before collection. It reclaims the Pandas and
numpy circular references that CPython's reference counting alone may not free.

**Step 5: assemble HeteroData**

- `HeteroData` stores references to the existing tensors, with no copy.
- Attach `edge_attr` for featurized edge types, again by reference.
- Delete the intermediate dicts, so only `HeteroData` holds references.
- Unpersist `edges_final_df`, freeing that executor cache.
- Unpersist `node_id_df`, freeing that one too.
- `gc.collect()` to reclaim the dict overhead.

**Post-construction: save the outputs**

- `torch.save()` writes the local `.pt` and, when archiving, a `BytesIO` buffer streamed to S3 via `upload_fileobj`. Peak is the `HeteroData` plus that buffer, the same size again; the buffer is freed after the upload.
- `MetadataCollector.to_metadata_files()` produces six JSON dicts, under 1 MB in total. `write_metadata_to_local()` writes the six files locally, and `write_metadata_to_s3()` adds six `put_object` calls when archiving. `MetadataCollector` holds only small Python dicts throughout.

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

| | ~32 GB driver | ~64 GB driver |
|---|---|---|
| JVM + Spark overhead | ~8-10 GB | ~10-12 GB |
| Python interpreter | ~1-2 GB | ~1-2 GB |
| Available for tensors | ~20-22 GB | ~50-52 GB |
| Node features | under 1M nodes at 1024-d | 2-5M nodes at 1024-d |
| Edge features | adds ~0.5-1 GB, typical temporal and option edges at 32-d | adds ~1-3 GB, every featurized edge type at 32-d |
| Metadata | under 1 MB, negligible | under 1 MB, negligible |

At ~32 GB, dropping to 512-d buys under 2M nodes with edge features still on.
~64 GB is the size to run intraday market data on.

Reducing `vector_dim` from 1024 to 512 **halves driver memory** for node feature tensors while preserving the same three-segment structure. Edge feature tensors at 32-d are already compact — reducing `edge_vector_dim` to 16 halves their memory but is rarely necessary since they are ~32× smaller per element than node features. This enables rapid experimentation at reduced resolution before committing to full-resolution production runs.

