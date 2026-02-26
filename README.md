# PyTorch Geometric Knowledge Graph Builder

> Serverless pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![AWS Glue](https://img.shields.io/badge/AWS-Glue-orange.svg)](https://aws.amazon.com/glue/)

## Overview

PyTorch Geometric Knowledge Graph Builder is a serverless pipeline that transforms raw RDF data from multiple heterogeneous sources into enriched knowledge graphs and constructs PyTorch Geometric `HeteroData` objects ready for Graph Neural Network (GNN) training.

The pipeline processes data from **100+ domain-specific ontologies** spanning economic indicators, financial filings, market data, and environmental alerts. All enrichment logic runs as **distributed PySpark DataFrame operations** on AWS Glue, enabling horizontal scaling across the cluster rather than bottlenecking on a single-threaded in-memory graph.

PyG construction also leverages Spark executors for all heavy computation (node ID assignment, edge resolution, feature extraction). Only compact integer and float tensors cross the Spark → driver boundary for final `HeteroData` assembly. All URI-to-name conversions use **pure Spark Column expressions** (JVM-native `WHEN` chains), not Python UDFs, eliminating serialization overhead.

Node feature vectors are **universal 1024-dimensional ontology-aware vectors** that encode three layers of information: ontology structure (class identity, hierarchy, source membership), property schema (presence, domain/range, property hierarchy), and literal values (numeric hashed slots, categorical multi-hot). All node types share the same vector width, enabling **shared GNN layers across heterogeneous types** and natural cross-type message passing. The vector dimension is configurable — all segment boundaries **scale proportionally** with `vector_dim`, so passing 512 produces a half-resolution vector with the same three-segment structure.

The pipeline supports three execution modes:

- **Full Pipeline**: End-to-end RDF enrichment and PyG graph construction
- **Enrichment Only**: Create reusable enriched Parquet artifacts
- **PyG Construction Only**: Rapidly experiment with different PyG graph structures from existing enriched Parquet

### Key Features

- **Large-Scale Integration**: Processes 100+ ontologies with tens of millions of triples per time period
- **Distributed Enrichment**: All enrichment runs as PySpark DataFrame operations across Spark executors
- **Distributed PyG Construction**: Node ID assignment, edge resolution, and feature extraction run on Spark executors — only compact tensors are collected to the driver
- **No Python UDFs in PyG Builder**: URI-to-name conversions use pure Spark `WHEN` expressions (JVM-native), not row-at-a-time Python UDFs
- **Ontology-Aware Feature Vectors**: Universal fixed-width vectors encoding class hierarchy, property schema, and literal values — not flat bags of literals
- **Universal Feature Width**: All node types share the same vector dimension, enabling shared GNN layers and cross-type message passing
- **Proportionally Scalable Dimensions**: Overriding `vector_dim` (e.g., 512, 2048) automatically rescales all segment and sub-segment boundaries via `VectorLayout` — no hardcoded dim indices
- **Driver Memory Safety**: Large node types use chunked collection with explicit memory management to prevent OOM
- **Temporal Unification**: Unified temporal entities across all data sources
- **Intra-Source Linking**: Automatic relationship discovery within data source families
- **Cross-Source Linking**: Automatic relationship discovery across heterogeneous datasets
- **PyTorch Geometric Output**: Native `HeteroData` objects with configurable node/edge types
- **Reusable Parquet Artifacts**: Enriched triples saved as Parquet for multiple PyG experiments without re-enrichment
- **Flexible Graph Construction**: Experiment with different graph structures from existing Parquet (5-10 min per experiment)
- **Serverless Architecture**: Fully managed AWS Glue, no infrastructure to maintain
- **Controlled Parquet Output**: Configurable partition count for optimal S3 file sizes
- **Canonical Namespace Registry**: Single source of truth for all namespace-to-prefix mappings in `rdf_utils.py`

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Raw Data Sources (S3) — N-Triples format                   │
│ ├── BLS Economic Data (10 categories, ~100 mappers) - RDF  │
│ ├── SEC Data (4 categories, 4 mappers) - RDF               │
│ ├── Market Data (1 mapper, intraday snapshots) - RDF       │
│ └── NOAA Weather Alerts (1 mapper) - RDF                   │
│                                                            │
│ Total: 100+ mappers and ontologies                         │
│ Volume: ~30-50M triples/month with intraday market data    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ AWS Glue Job: pyg-knowledge-graph-builder                  │
│                                                            │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   │
│  │ Parse        │──▶│ Enrichment    │──▶│ Build PyG    │   │
│  │ N-Triples    │   │ (PySpark      │   │ (PySpark     │   │
│  │ (Spark regex │   │  DataFrames   │   │  executors   │   │
│  │  on executors│   │  on executors)│   │  → driver    │   │
│  │  → triples   │   │               │   │  tensors)    │   │
│  │  DataFrame)  │   │               │   │              │   │
│  └──────────────┘   └───────────────┘   └──────────────┘   │
│                                                            │
│ Mode 1: Full Pipeline                                      │
│   N-Triples → triples_df → Enrich → Save Parquet           │
│   → Build PyG HeteroData → Save .pt                        │
│                                                            │
│ Mode 2: Enrichment Only                                    │
│   N-Triples → triples_df → Enrich → Save Parquet to S3     │
│                                                            │
│ Mode 3: PyG Only                                           │
│   Enriched Parquet (S3) → triples_df → Build PyG HeteroData│
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Outputs (S3)                                               │
│ ├── Enriched Triples (Parquet) - Reusable artifact         │
│ └── PyTorch Geometric HeteroData (.pt files) - GNN ready   │
└────────────────────────────────────────────────────────────┘
```

### Core Representation

All RDF data is parsed from N-Triples files into a single **triples DataFrame** that serves as the universal graph representation throughout the pipeline:

```
Schema: (subject: string, predicate: string, object: string)

┌─────────────────────────────────┬──────────────────────┬────────────────────┐
│ subject                         │ predicate            │ object             │
├─────────────────────────────────┼──────────────────────┼────────────────────┤
│ cpi:Food_Nov2024_Index          │ rdf:type             │ cpi:Index          │
│ cpi:Food_Nov2024_Index          │ cpi:indexValue       │ 295.8              │
│ cpi:Food_Nov2024_Index          │ cpi:hasMonth         │ cpi:November       │
│ cpi:Food_Nov2024_Index          │ cpi:hasCategory      │ cpi:Food_Entity    │
│ market:AAPL_Obs_20241115        │ rdf:type             │ market:PriceObs    │
│ market:AAPL_Obs_20241115        │ market:observedPrice │ 191.45             │
└─────────────────────────────────┴──────────────────────┴────────────────────┘
```

**N-Triples Parsing**: Raw `.nt` files are read as text by `spark.read.text()` and parsed on executors using Spark regex functions (`regexp_extract`). Subject and predicate URIs are extracted from angle brackets, and object values are cleaned (URI angle brackets stripped, literal datatype suffixes and language tags removed). No data passes through the driver during parsing.

Enrichment steps read from this DataFrame, produce new triples DataFrames, and union them back. The enriched DataFrame is saved as **Parquet** for reuse. PyG construction reads the enriched DataFrame, assigns integer node IDs, resolves edges, and extracts features — all on Spark executors. Only compact tensors cross to the driver for final `HeteroData` assembly.

### Why PySpark Instead of rdflib/SPARQL

| Aspect | rdflib + SPARQL | PySpark DataFrames |
|--------|----------------|-------------------|
| Execution | Single Python process on Glue driver | Distributed across all Spark executors |
| Memory | Entire graph must fit in driver RAM | Partitioned across cluster |
| Query optimization | None (sequential iteration) | Catalyst optimizer, predicate pushdown, broadcast joins |
| Parallelism | None | Automatic partitioning |
| Glue DPU utilization | Pays for cluster, uses 1 core | Uses all allocated DPUs |
| Join pattern | Python dict lookups or nested SPARQL | Distributed hash/sort-merge joins |

rdflib Namespace objects are used as **URI string constants** in the enrichment modules for readability — they produce plain strings and don't hold or query graph data. The PyG builder modules use **pure Spark Column expressions** for all URI-to-name conversions (no Python UDFs).

## Ontology-Aware Feature Vectors

### The Problem With Flat Literal Vectors

A naive approach encodes each node as a flat bag of its literal property values — `[indexValue, percentChange, relativeImportance, ...]`. This has fundamental limitations:

- **No structural context**: The vector for a `cpi:Index` node knows `indexValue = 295.8` but encodes nothing about what that node *is* in the ontology
- **Semantic collisions**: Two nodes from different ontologies sharing a property name (e.g., `hasValue`) get the same feature column despite completely different meanings
- **Ambiguous zeros**: A zero in a column is indistinguishable between "missing value" and "property doesn't apply to this type"
- **Per-type isolation**: Each node type has a different feature width, requiring type-specific linear layers in the GNN and preventing cross-type weight sharing

### The Solution: Ontology-Structure + Literal Hybrid Vector

Every node gets a fixed-width vector (default 1024-d) with three segments encoding progressively more specific information. All segment boundaries are computed proportionally by `VectorLayout`, so the structure scales to any `vector_dim`:

```
Default 1024-dimensional node feature vector
┌─────────────────────┬──────────────────────┬─────────────────────┐
│ Ontology Structure  │ Property Presence &  │ Literal Values      │
│ (class hierarchy,   │ Schema Signals       │ (numeric + encoded  │
│  type identity)     │ (which properties    │  categorical)       │
│                     │  are defined/present)│                     │
│ 25% of vector_dim   │ 37.5% of vector_dim  │ 37.5% of vector_dim │
│ (256 dims @ 1024)   │ (384 dims @ 1024)    │ (384 dims @ 1024)   │
└─────────────────────┴──────────────────────┴─────────────────────┘
```

#### Segment 1: Ontology Structure (25% of vector_dim)

Encodes **what the node is** in the ontology hierarchy — its class, its superclasses, and its ontology membership. Gives the GNN a structural fingerprint consistent across all nodes of the same type.

```
Segment 1: Ontology Structure [25% of vector_dim]
┌────────────────────┬────────────────────┬────────────────────┐
│ Class Identity     │ Class Hierarchy    │ Ontology/Source    │
│ (multi-hot hash    │ (rdfs:subClassOf   │ (which ontology    │
│  of rdf:type URIs) │  chain, depth-     │  namespace, multi- │
│                    │  weighted hashing) │  hot encoding)     │
│ 25% of segment     │ 50% of segment     │ 25% of segment     │
│ (64 dims @ 1024)   │ (128 dims @ 1024)  │ (64 dims @ 1024)   │
└────────────────────┴────────────────────┴────────────────────┘
```

- **Class Identity**: Each `rdf:type` URI is hashed into 4 deterministic slots. Nodes of the same type share identical bits.
- **Class Hierarchy**: `rdfs:subClassOf` chains are traversed (transitive closure up to depth 10). Superclass URIs are hashed with depth-weighted values (direct superclass = 1.0, grandparent = 0.5, etc.). Nodes sharing a superclass share bits in this segment.
- **Ontology/Source Membership**: Multi-hot encoding of which ontology namespace(s) the node belongs to, derived from both the type URI and the node URI itself. Uses the canonical namespace registry from `rdf_utils.py`.

#### Segment 2: Property Schema (37.5% of vector_dim)

Encodes **which ontology-defined properties are present** for this node, regardless of their values. This tells the GNN about schema conformance and distinguishes "missing because not observed" from "missing because inapplicable."

```
Segment 2: Property Schema [37.5% of vector_dim]
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Property Presence   │ Domain/Range Signals│ Property Hierarchy  │
│ (which properties   │ (rdfs:domain and    │ (rdfs:subPropertyOf │
│  this node has,     │  rdfs:range of      │  chains)            │
│  multi-hot hashed)  │  properties)        │                     │
│ 50% of segment      │ 29% of segment      │ 21% of segment      │
│ (192 dims @ 1024)   │ (112 dims @ 1024)   │ (80 dims @ 1024)    │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

- **Property Presence**: Each predicate URI the node has is hashed into 3 slots. A CPI Index node with `indexValue`, `percentChange`, `hasMonth` gets different bits than a JOLTS node with `jobOpeningsLevel`, `hasIndustry`.
- **Domain/Range Signals**: For each property this node has, the `rdfs:domain` and `rdfs:range` types declared in the ontology are hashed. This tells the GNN what types of relationships this node can participate in.
- **Property Hierarchy**: `rdfs:subPropertyOf` relationships are hashed, connecting specific properties to their abstract parents.

#### Segment 3: Literal Values (37.5% of vector_dim)

Carries the actual numeric and categorical values in a fixed-width format with proper encoding.

```
Segment 3: Literal Values [37.5% of vector_dim]
┌─────────────────────┬─────────────────────┐
│ Numeric Values      │ Categorical Values  │
│ (z-score normalized │ (multi-hot hash     │
│  into hashed slots) │  encoding)          │
│ 67% of segment      │ 33% of segment      │
│ (256 dims @ 1024)   │ (128 dims @ 1024)   │
└─────────────────────┴─────────────────────┘
```

- **Numeric Values**: Each numeric property's predicate URI hashes to a fixed slot. The value is z-score normalized (per-predicate stats computed in a single pass on executors) and placed at that slot. Hash collisions sum — rare with 256 dims and ~10 properties per type.
- **Categorical Values**: Multi-hot hash encoding instead of `dense_rank`. Each `(predicate, value)` pair hashes to 4 slots. No ordinal assumption.

### Proportional Dimension Scaling via VectorLayout

All segment and sub-segment boundaries are computed at runtime by the `VectorLayout` class from the configured `vector_dim`. No dim indices are hardcoded in the encoding logic. This means overriding `vector_dim` from a notebook invocation automatically produces a correctly structured vector at the requested resolution:

```
VectorLayout(1024) — default, production:
  Segment 1: Ontology Structure [0–255]     (256 dims)
    Class Identity:     [0–63]              (64 dims)
    Class Hierarchy:    [64–191]            (128 dims)
    Ontology Source:    [192–255]           (64 dims)
  Segment 2: Property Schema    [256–639]   (384 dims)
    Property Presence:  [256–447]           (192 dims)
    Domain/Range:       [448–559]           (112 dims)
    Property Hierarchy: [560–639]           (80 dims)
  Segment 3: Literal Values     [640–1023]  (384 dims)
    Numeric Values:     [640–895]           (256 dims)
    Categorical Values: [896–1023]          (128 dims)

VectorLayout(512) — half resolution, faster experiments:
  Segment 1: Ontology Structure [0–127]     (128 dims)
    Class Identity:     [0–31]              (32 dims)
    Class Hierarchy:    [32–95]             (64 dims)
    Ontology Source:    [96–127]            (32 dims)
  Segment 2: Property Schema    [128–319]   (192 dims)
    Property Presence:  [128–223]           (96 dims)
    Domain/Range:       [224–279]           (56 dims)
    Property Hierarchy: [280–319]           (40 dims)
  Segment 3: Literal Values     [320–511]   (192 dims)
    Numeric Values:     [320–448]           (129 dims)
    Categorical Values: [449–511]           (63 dims)

VectorLayout(256) — quarter resolution, rapid prototyping:
  Segment 1: Ontology Structure [0–63]      (64 dims)
    Class Identity:     [0–15]              (16 dims)
    Class Hierarchy:    [16–47]             (32 dims)
    Ontology Source:    [48–63]             (16 dims)
  Segment 2: Property Schema    [64–159]    (96 dims)
    Property Presence:  [64–111]            (48 dims)
    Domain/Range:       [112–139]           (28 dims)
    Property Hierarchy: [140–159]           (20 dims)
  Segment 3: Literal Values     [160–255]   (96 dims)
    Numeric Values:     [160–223]           (64 dims)
    Categorical Values: [224–255]           (32 dims)

VectorLayout(2048) — double resolution, maximum fidelity:
  Segment 1: Ontology Structure [0–511]     (512 dims)
    Class Identity:     [0–127]             (128 dims)
    Class Hierarchy:    [128–383]           (256 dims)
    Ontology Source:    [384–511]           (128 dims)
  Segment 2: Property Schema    [512–1279]  (768 dims)
    Property Presence:  [512–895]           (384 dims)
    Domain/Range:       [896–1119]          (224 dims)
    Property Hierarchy: [1120–1279]         (160 dims)
  Segment 3: Literal Values     [1280–2047] (768 dims)
    Numeric Values:     [1280–1793]         (514 dims)
    Categorical Values: [1794–2047]         (254 dims)
```

`VectorLayout` validates at construction time that all sub-segments are contiguous, non-overlapping, each has at least 1 dimension, and they sum exactly to `vector_dim`. If `vector_dim` is too small (< 32), it raises immediately with a clear error rather than silently producing a degenerate vector.

**Tradeoffs when reducing `vector_dim`:**

| vector_dim | Hash collision risk | Driver memory per 1M nodes | Use case |
|-----------|-------------------|--------------------------|----------|
| 2048 | Very low | ~8 GB | Maximum fidelity, large cluster |
| 1024 | Low (~10 properties/type vs 256 numeric slots) | ~4 GB | Production default |
| 512 | Moderate (128 numeric slots) | ~2 GB | Fast experiments on G.2X |
| 256 | Higher (64 numeric slots) | ~1 GB | Rapid prototyping, small datasets |

**Notebook invocation example:**

```python
# Quick experiment with half-resolution vectors on G.2X
config = {
    "feature_config": {
        "vector_dim": 512,
        "normalize": True
    }
}

# The Glue job parameter:
"--pyg_config": json.dumps(config)
```

### Why This Is Better for GNNs

```
OLD approach (flat literal vectors):
┌────────────────────────────────────────────────────────┐
│ cpi_Index:  [295.8, 0.3, 2.1, 0.05, 3.0, 1.0]          │  6 dims, only literals
│ ppi_Index:  [187.2, 0.1, 1.5, 0.03, 2.0, 1.0]          │  6 dims, only literals
│                                                        │
│ SEPARATE tensors per type (different widths)           │
│ GNN needs type-specific linear layers                  │
│ No cross-type weight sharing possible                  │
│ Zero = missing? or inapplicable? GNN can't tell        │
└────────────────────────────────────────────────────────┘

NEW approach (ontology-aware vectors):
┌────────────────────────────────────────────────────────┐
│ cpi_Index:  [ontology:25% | schema:37.5% | lit:37.5%]  │  1024-d, universal
│ ppi_Index:  [ontology:25% | schema:37.5% | lit:37.5%]  │  1024-d, universal
│                                                        │
│ SAME tensor width for ALL node types                   │
│ Shared ontology bits where types share ancestry        │
│ GNN can use SHARED layers across all types             │
│ Cross-type message passing works naturally             │
│ Property presence distinguishes missing vs N/A         │
│ Override vector_dim for memory/fidelity tradeoff       │
└────────────────────────────────────────────────────────┘
```

### All Encoding Runs on Spark Executors

Every encoding operation uses pure Spark expressions — no Python UDFs:

| Encoding | Spark Operation | Example |
|----------|----------------|---------|
| Class identity hash | `F.abs(F.hash(col, lit(seed))) % dim + offset` | `rdf:type` URI → 4 slots in class identity sub-segment |
| Hierarchy traversal | Iterative self-join on `rdfs:subClassOf` | Transitive closure up to depth 10 |
| Depth weighting | `F.lit(1.0) / F.col("depth").cast("double")` | Direct superclass = 1.0, grandparent = 0.5 |
| Namespace membership | `F.col("type_uri").startswith(namespace)` | Multi-hot encoding in ontology source sub-segment |
| Property presence hash | `F.abs(F.hash(col, lit(seed))) % dim + offset` | Predicate URI → 3 slots in property presence sub-segment |
| Numeric normalization | `F.broadcast(stats)` join + arithmetic | Per-predicate z-score in single pass |
| Numeric slot hashing | `F.abs(F.hash(predicate, lit(500))) % dim + offset` | Predicate → fixed slot in numeric sub-segment |
| Categorical multi-hot | `F.hash(F.concat(predicate, "::", value), lit(seed))` | (predicate, value) → 4 slots in categorical sub-segment |

All `dim` and `offset` values in the table above are read from `VectorLayout` at runtime — they scale with `vector_dim`.

## PyG Construction Pipeline

The PyG builder converts the enriched triples DataFrame into a PyTorch Geometric `HeteroData` object through four steps, with all heavy computation on Spark executors:

```
triples_df (enriched, on executors)
    │
    ├── Step 1: NodeMapper (on executors)
    │   ├── Discover node types from rdf:type triples
    │   ├── Filter out meta-ontology types (OWL, RDFS, RDF)
    │   ├── Convert type URIs to PyG names via pure Spark WHEN expressions
    │   ├── Assign canonical type per entity (most specific wins via type count)
    │   ├── Assign per-type 0-indexed integer IDs via Window functions
    │   ├── Cache and materialize node_id_df on executors
    │   └── Output: node_id_df (uri, node_id, node_type) — cached on executors
    │              node_counts Dict[str, int] — small collect to driver
    │
    ├── Step 2: EdgeMapper (on executors → driver tensors)
    │   ├── Exclude structural predicates (rdf:type, rdfs:label, etc.)
    │   ├── Double-join triples with node_id_df (subject → src_id, object → dst_id)
    │   ├── Inner join on object naturally filters out literal properties
    │   ├── Derive relation names via pure Spark WHEN expressions (no UDF)
    │   ├── Cache resolved edges, discover distinct edge types (small collect)
    │   ├── Collect per-edge-type [2, num_edges] int64 arrays via toPandas()
    │   ├── Release Pandas memory after each edge type conversion
    │   └── Output: Dict[(src_type, relation, dst_type) → LongTensor]
    │
    ├── Step 3: FeatureExtractor (on executors → driver tensors)
    │   ├── Compute VectorLayout from configured vector_dim (all boundaries
    │   │   scale proportionally — no hardcoded dim indices)
    │   ├── Extract ontology structure from triples (rdfs:subClassOf chains,
    │   │   rdfs:domain/range, rdfs:subPropertyOf) — all on executors
    │   ├── Compute per-node property presence via join — on executors
    │   ├── Extract numeric literals via anti-join + cast("double") — on executors
    │   ├── Extract categorical literals via anti-join + null cast filter — on executors
    │   ├── Compute per-predicate z-score stats (single-pass agg) — on executors
    │   ├── For each node type (largest first):
    │   │   ├── Encode Segment 1: class identity + hierarchy + source (hash-based)
    │   │   ├── Encode Segment 2: property presence + domain/range + prop hierarchy
    │   │   ├── Encode Segment 3: numeric hashed slots + categorical multi-hot
    │   │   ├── Union segments, aggregate (sum at same node_id+dim) — on executors
    │   │   ├── Pre-allocate dense numpy array on driver
    │   │   ├── Collect sparse entries via toPandas() (chunked for large types)
    │   │   ├── Scatter into dense array, delete Pandas, gc.collect()
    │   │   └── Convert numpy → torch (zero-copy via from_numpy)
    │   └── Output: Dict[node_type → FloatTensor[num_nodes, vector_dim]]
    │
    └── Step 4: Assemble HeteroData (on driver)
        ├── Only compact tensors on driver — no URI strings
        ├── Attach feature tensors per type (same width for all)
        ├── Attach edge_index tensors per (src, rel, dst) type
        ├── Release intermediate dicts, gc.collect()
        ├── Release node_id_df from executor cache
        └── Output: HeteroData ready for torch.save() and GNN training
```

### Driver Memory Safety

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

  gc.collect() is safe here because it runs on the driver process
  only — all Spark executor work is complete before collection.
  It reclaims Pandas/numpy circular references that CPython's
  reference counting alone may not free.

Step 4: Assemble HeteroData
  HeteroData stores references to existing tensors (no copy)
  Delete intermediate dicts → only HeteroData holds references
  gc.collect() to reclaim dict overhead

Step 5: Save to S3 (in build_graph.py)
  torch.save() to BytesIO buffer → upload_fileobj streams to S3
  Peak: HeteroData + serialized buffer (same size)
  Buffer freed after upload
```

**Chunked collection for large types**: When a node type has more than 500K nodes (configurable via `chunk_node_threshold`), the sparse `(node_id, dim, value)` entries are collected in chunks by node_id range. Each chunk's Pandas DataFrame is scattered into the pre-allocated dense array and immediately freed. This bounds peak Pandas memory to ~120 MB per chunk regardless of total type size.

**Largest types processed first**: Types are sorted by node count (descending) so that if a type is too large for available driver memory, the job fails fast rather than after processing all smaller types.

### Scaling Characteristics

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
| Hash-based encoding | Spark executors | `F.hash()`, `F.abs()`, `F.lit()` — JVM-native, no Python UDF |
| Feature normalization | Spark executors | Single-pass `agg()` for per-predicate stats, broadcast joined |
| Sparse entry aggregation | Spark executors | `groupBy(node_id, dim).agg(sum)` — handles hash collisions |
| Edge index collection | Driver | Per-edge-type [2, N] int64 — ~16 bytes/edge |
| Feature collection | Driver | Per-type sparse entries → dense [N, vector_dim] float32, chunked for large types |
| HeteroData assembly | Driver | Only compact tensors, no strings |
| Enriched Parquet write | Spark executors | `repartition` + `write.parquet` — executors write directly to S3 |
| PyG .pt upload | Driver | `upload_fileobj` streams buffer to S3 (no extra copy) |

**Universal feature width**: HeteroData stores feature tensors of the same `vector_dim` for every node type. The ontology-aware encoding keeps vectors informative even for types with few literal properties — the ontology structure and property schema segments still carry meaningful signal.

| Node type example | Typical nodes | Memory @ 1024-d | Memory @ 512-d | Key signals |
|-------------------|--------------|-----------------|----------------|-------------|
| cpi_Index | ~50K | ~200 MB | ~100 MB | CPI class hierarchy, index/change properties, BLS source |
| market_PriceObservation | ~500K-1M | ~2-4 GB | ~1-2 GB | Market class, price/volume properties, ticker source |
| market_options_OptionQuote | ~1-2M | ~4-8 GB | ~2-4 GB | Options subclass, strike/expiry/greeks properties |
| jolts_JobOpeningsLevel | ~10K | ~40 MB | ~20 MB | JOLTS hierarchy, level/rate properties, BLS source |
| filings_Form4 | ~50K | ~200 MB | ~100 MB | SEC filing class, transaction properties, SEC source |
| unified_UnifiedMonth | ~12 | ~48 KB | ~24 KB | Temporal class, cross-source membership |

### Memory Budget by Glue Worker Type

```
Glue G.2X (32 GB driver):
  JVM + Spark overhead:     ~8-10 GB
  Python interpreter:       ~1-2 GB
  Available for tensors:    ~20-22 GB
  → Suitable for <1M total nodes at 1024-d
  → Or <2M total nodes at 512-d

Glue G.4X (64 GB driver):
  JVM + Spark overhead:     ~10-12 GB
  Python interpreter:       ~1-2 GB
  Available for tensors:    ~50-52 GB
  → Suitable for 2-5M total nodes at 1024-d
  → Recommended for production with intraday market data
```

Reducing `vector_dim` from 1024 to 512 **halves driver memory** for feature tensors while preserving the same three-segment structure. This enables running on G.2X workers for rapid experimentation before committing to full-resolution production runs on G.4X.

### PyG Configuration

The PyG builder accepts an optional configuration dict:

```json
{
    "node_types": ["cpi_Index", "ppi_MonthlyChange", "market_PriceObservation"],
    "edge_types": ["bls_enrichment_precedes", "bls_enrichment_correlatesWith"],
    "feature_config": {
        "normalize": true,
        "vector_dim": 1024,
        "chunk_node_threshold": 500000
    },
    "include_temporal_nodes": true,
    "include_sector_nodes": true
}
```

| Config key | Default | Description |
|-----------|---------|-------------|
| `node_types` | All rdf:type classes | Whitelist of PyG node type names to include |
| `edge_types` | All entity-to-entity predicates | Whitelist of relation names to include |
| `feature_config.normalize` | `true` | Z-score normalize numeric features (single-pass per-predicate) |
| `feature_config.vector_dim` | `1024` | Feature vector dimension — all segments scale proportionally. Minimum 32. |
| `feature_config.chunk_node_threshold` | `500000` | Node count above which chunked collection is used |
| `include_temporal_nodes` | `true` | Include Month/Year/Quarter node types |
| `include_sector_nodes` | `true` | Include EconomicSector node types |

When config is empty, sensible defaults are inferred from the data.

### Glue Job Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--mode` | Yes | `full` | `full`, `enrichment_only`, or `pyg_only` |
| `--source_bucket` | Modes 1,2 | — | S3 bucket containing raw N-Triples files |
| `--source_prefix` | Modes 1,2 | — | S3 prefix for raw N-Triples files |
| `--output_bucket` | Yes | — | S3 bucket for all outputs |
| `--enriched_parquet_prefix` | Mode 3 | `enriched/{time_period}/triples/` | S3 prefix for enriched Parquet |
| `--pyg_output_key` | Modes 1,3 | `pyg/{time_period}/hetero_data.pt` | S3 key for PyG output |
| `--enable_ontology_mapping` | No | `false` | Enable ontology equivalence mapping |
| `--time_period` | No | Current `YYYY-MM` | Time period label for output paths |
| `--pyg_config` | No | `{}` | JSON string with PyG construction config |
| `--parquet_partitions` | No | `200` | Number of Parquet output partitions |

**Example Glue job parameters:**

```json
{
    "--mode": "full",
    "--source_bucket": "my-data-lake",
    "--source_prefix": "rdf/monthly/2024-12/",
    "--output_bucket": "my-data-lake",
    "--enriched_parquet_prefix": "enriched/2024-12/triples/",
    "--pyg_output_key": "pyg/2024-12/hetero_data.pt",
    "--enable_ontology_mapping": "true",
    "--time_period": "2024-12",
    "--parquet_partitions": "200",
    "--pyg_config": "{\"feature_config\": {\"normalize\": true, \"vector_dim\": 1024}}"
}
```

**Notebook experiment with reduced dimensions:**

```json
{
    "--mode": "pyg_only",
    "--output_bucket": "my-data-lake",
    "--enriched_parquet_prefix": "enriched/2024-12/triples/",
    "--pyg_output_key": "pyg/2024-12/hetero_data_512d.pt",
    "--time_period": "2024-12",
    "--pyg_config": "{\"feature_config\": {\"vector_dim\": 512, \"normalize\": true}}"
}
```

## Knowledge Graph Enrichment

The enrichment pipeline creates a unified knowledge graph by establishing relationships at two levels across **100+ data sources and ontologies**. Each enrichment step is a PySpark transformation that reads the triples DataFrame, computes new relationship triples, and unions them back.

### Enrichment Pipeline Flow

```
triples_df (raw)
    │
    ├── BLS Intra-Source Enricher
    │   ├── Temporal sequences (precedes links)
    │   ├── Sector classification (belongsToSector)
    │   ├── Cross-dataset correlations (correlatesWith)
    │   └── Hierarchical enrichment (hasParent chains)
    │
    ├── SEC Intra-Source Enricher
    │   ├── Company unification (owl:sameAs by CIK)
    │   ├── Person unification (owl:sameAs by CIK)
    │   ├── Filing sequences (precedes by date)
    │   ├── Sector classification (belongsToSector)
    │   └── Violation type linking (hasViolationType)
    │
    ├── Market Intra-Source Enricher
    │   ├── Ticker unification (owl:sameAs across sources)
    │   ├── Price sequences (precedes by timestamp)
    │   ├── Option-stock linking (hasUnderlyingPriceObservation)
    │   ├── Option strategy detection (straddleWith, spreadWith)
    │   └── Sector classification (belongsToSector)
    │
    ├── NOAA Intra-Source Enricher
    │   ├── Alert temporal sequences (precedes by sent time)
    │   ├── Geographic linking (affectsSameRegion via SAME codes)
    │   ├── Event type linking (sameEventType)
    │   └── Severity escalation detection (escalatesTo)
    │
    ├── Temporal Unifier (cross-source)
    │   └── Unified months/years/quarters (owl:sameAs)
    │
    ├── Cross-Source Linker
    │   ├── Sector-based linking across sources
    │   ├── Company/ticker linking (SEC ↔ Market)
    │   ├── Geographic linking (BLS ↔ NOAA)
    │   ├── Causal relationships (BLS → Market, NOAA → Market)
    │   └── Measurement type alignment
    │
    └── Ontology Mapper (optional)
        ├── owl:equivalentProperty mappings
        └── owl:equivalentClass mappings
    │
    ▼
triples_df (enriched) → Parquet (S3) + PyG HeteroData (.pt)
```

### Intra-Source Linking

Discovers and creates relationships within each data source family:

**Within BLS Economic Data** (10 categories, ~100 mappers)
- Links related indicators across CPI, PPI, ECI, EMPSIT, JOLTS, LAUS, METRO, REALER, WKYENG, XIMPIM
- Connects hierarchical category structures (e.g., All Items → Food → Food at Home)
- Establishes temporal sequences within each indicator
- Correlates related measurements (e.g., CPI Food ↔ PPI Food Manufacturing)

**Within SEC Data** (4 categories, 4 mappers)
- Unifies company entities across filings, proceedings, and suspensions by CIK
- Unifies person entities across filings and proceedings by CIK
- Links filings in chronological sequences per company/owner
- Classifies entities by sector and violation type

**Within Market Data** (1 mapper, intraday snapshots every 10-30 minutes)
- Links price observations in chronological sequences per ticker
- Links option contracts to underlying stock price observations
- Identifies option strategies (straddles, vertical spreads, strangles)
- Classifies tickers by sector
- Links multi-source observations of same ticker/contract

**Within NOAA Weather Data** (1 mapper)
- Links alerts in chronological sequences per geographic area
- Connects alerts affecting same regions via SAME geocodes
- Links alerts of the same event type
- Detects severity escalations within same area over time

### Enrichment as PySpark Operations

Each enrichment step follows the same pattern — filter the triples DataFrame to extract relevant entities, join to discover relationships, and produce new triples:

```python
def link_options_to_stocks(self, triples_df):
    """Example: Link option contracts to underlying stock prices"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Extract option contracts (filter + self-join to pivot properties)
    contracts_df = (
        triples_df.filter(F.col("predicate") == RDF_TYPE)
                  .filter(F.col("object") == OPTION_CONTRACT_TYPE)
                  .select(F.col("subject").alias("contract"))
        .join(
            triples_df.filter(F.col("predicate") == UNDERLYING_TICKER)
                      .select(F.col("subject").alias("contract"),
                              F.col("object").alias("ticker")),
            "contract"
        )
    )

    # Extract price observations — one representative per ticker
    w = Window.partitionBy("ticker").orderBy("obs")
    prices_df = (
        triples_df.filter(...)  # similar filter+join pattern
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
    )

    # Distributed join — Spark handles partitioning and optimization
    joined = contracts_df.join(prices_df, on="ticker", how="inner")

    # Produce new triples
    return joined.select(
        F.col("contract").alias("subject"),
        F.lit(HAS_UNDERLYING_OBS).alias("predicate"),
        F.col("price_obs").alias("object")
    )
```

**Key patterns used across all enrichers:**

| Pattern | PySpark Operation | Example |
|---------|------------------|---------|
| Entity extraction | `filter` + `select` | Find all CPI Index entities |
| Property pivot | Self-join on subject | Get (entity, month, year, value) rows |
| Temporal sequencing | `Window.partitionBy().orderBy()` + `lag`/`lead` | Link consecutive measurements |
| Cross-dataset correlation | `join` on normalized keywords | CPI Food ↔ PPI Food Manufacturing |
| Sector classification | `join` with broadcast pattern dict | Classify entities by sector keywords |
| Entity unification | `groupBy` + `collect_list` + `explode` | Unify companies by CIK across SEC datasets |
| Existence check | `left_anti` join | Only add triples that don't already exist |

### Cross-Source Linking

Discovers and creates relationships across different data source families:

**Linking Strategies** (applied across 100+ ontologies):

1. **Temporal Alignment** — Unifies temporal entities across all sources
```turtle
# Before: each source has its own temporal entities
cpi:November, ppi:November, jolts:November, sec:November, market:November

# After: single unified temporal entity
unified:November2024 a bls:UnifiedMonth ;
    owl:sameAs cpi:November, ppi:November, jolts:November,
               sec:November, market:November, noaa:November .
```

2. **Sector-Based Linking** — Links entities sharing economic sectors
```turtle
unified:EnergySector a bls:EconomicSector .

cpi:EnergyEntity bls:belongsToSector unified:EnergySector .
ppi:EnergyGoodsEntity bls:belongsToSector unified:EnergySector .
market:XOM_Ticker bls:belongsToSector unified:EnergySector .
sec:EnergyCompanyFiling bls:belongsToSector unified:EnergySector .
```

3. **Company/Ticker-Based Linking** — Links entities referencing same companies
```turtle
unified:Company_AAPL a bls:UnifiedCompany ;
    bls:ticker "AAPL" .

sec:AAPL_10K_Filing bls:refersToCompany unified:Company_AAPL .
market:AAPL_Ticker bls:refersToCompany unified:Company_AAPL .
```

4. **Geographic/Regional Linking** — Links entities by geographic region
```turtle
unified:CaliforniaRegion a bls:GeographicRegion .

laus:California_LaborForce bls:hasRegion unified:CaliforniaRegion .
noaa:California_HeatAlert bls:affectsRegion unified:CaliforniaRegion .
```

5. **Causal/Impact Relationships** — Discovers potential causal links
```turtle
ppi:EnergyGoods bls:leadsTo cpi:EnergyConsumer .
noaa:HurricaneAlert bls:impacts market:EnergyTicker .
sec:Form10K_Filing bls:affects market:StockTicker .
```

6. **Measurement Type Alignment** — Links similar measurement types
```turtle
cpi:IndexMeasurement a bls:PriceIndex .
ppi:IndexMeasurement a bls:PriceIndex .
jolts:RateMeasurement a bls:RateMeasurement .
laus:UnemploymentRate a bls:RateMeasurement .
```

### Example Intra-Source Patterns

```turtle
# ============================================
# Pattern 1: Hierarchical Relationships
# ============================================
# CPI category hierarchy (captured in raw RDF by mappers)
cpi:AllItems_Entity a cpi:AllItems ;
    rdfs:label "All items" .

cpi:AllItems_Food_Entity a cpi:Food ;
    rdfs:label "Food" ;
    cpi:hasParent cpi:AllItems_Entity .

cpi:AllItems_Food_FoodAtHome_Entity a cpi:FoodAtHome ;
    rdfs:label "Food at home" ;
    cpi:hasParent cpi:AllItems_Food_Entity .

# ============================================
# Pattern 2: Temporal Sequences (enrichment adds)
# ============================================
cpi:AllItems_Food_November2024_Index a cpi:Index ;
    cpi:indexValue "295.8"^^xsd:decimal ;
    cpi:hasCategory cpi:AllItems_Food_Entity ;
    cpi:hasMonth cpi:November ;
    cpi:hasYear cpi:2024 .

cpi:AllItems_Food_December2024_Index a cpi:Index ;
    cpi:indexValue "296.2"^^xsd:decimal ;
    cpi:hasCategory cpi:AllItems_Food_Entity ;
    cpi:hasMonth cpi:December ;
    cpi:hasYear cpi:2024 .

# Enrichment adds temporal ordering
cpi:AllItems_Food_November2024_Index bls:precedes
    cpi:AllItems_Food_December2024_Index .

# ============================================
# Pattern 3: Intra-Source Correlations (enrichment adds)
# ============================================
cpi:AllItems_Food_Entity bls:correlatesWith
    ppi:FinalDemand_FoodManufacturing_12345_Entity .

jolts:Industry_LeisureAndHospitality_FoodServices_Industry
    bls:correlatesWith empsit:LeisureAndHospitality_Employment_Entity .
```

### Enrichment Statistics (typical for 1-month dataset)

| Enrichment Type | Triples Added | Example |
|----------------|---------------|---------|
| Temporal Unification | ~50,000 | All sources → unified months/years |
| Sector-Based Links | ~10,000 | Energy entities across CPI/PPI/JOLTS/Market |
| Company/Ticker Links | ~5,000 | SEC filings ↔ Stock prices |
| Geographic Links | ~3,000 | Regional employment ↔ Weather ↔ Market |
| Causal Relationships | ~8,000 | PPI → CPI, JOLTS → CPI, Weather → Market |
| Hierarchical Enrichment | ~15,000 | Parent-child relationships across sources |
| **Total Enrichment** | **~91,000** | Added to ~500,000 raw triples |

### Benefits for GNN Training

This enriched structure combined with ontology-aware feature vectors enables GNNs to learn:
- **Temporal Patterns**: How indicators evolve and correlate over time across 100+ sources
- **Cross-Domain Relationships**: How economic, financial, employment, and environmental factors interact
- **Sector Dynamics**: How sector-wide shocks propagate across different data types
- **Lead-Lag Relationships**: Which indicators predict changes in others
- **Geographic Effects**: How regional factors affect economic and market outcomes
- **Company-Specific Patterns**: How company fundamentals relate to market performance
- **Intraday Dynamics**: How market prices and options evolve within trading sessions
- **Ontology-Aware Similarity**: Nodes sharing superclasses or property schemas are naturally similar in feature space, even before training
- **Cross-Type Reasoning**: Universal feature width enables shared GNN layers that learn patterns across all 100+ ontologies simultaneously

## Data Sources

The pipeline ingests RDF data from multiple heterogeneous sources, all in **N-Triples format**:

**BLS Economic Data** (10 categories, ~100 mappers)
- CPI (Consumer Price Index) — 8 tables
- PPI (Producer Price Index) — 7 tables
- ECI (Employment Cost Index) — 14 tables
- EMPSIT (Employment Situation) — 27 tables
- JOLTS (Job Openings and Labor Turnover) — 15 tables
- LAUS (Local Area Unemployment Statistics) — 3 tables
- METRO (Metropolitan Area Statistics) — 4 tables
- REALER (Real Earnings) — 2 tables
- WKYENG (Weekly Earnings) — 6 tables
- XIMPIM (Import/Export Price Indexes) — 11 tables

**SEC Data** (4 categories, 4 mappers)
- Company filings (10-K, 10-Q, 8-K, Forms 3/4/5)
- Administrative proceedings
- Litigation releases
- Trading suspensions

**Market Data** (1 mapper, intraday snapshots every 10-30 minutes)
- Stock prices with options chains (select tickers)
- ~39 snapshots/day at 10-min intervals during trading hours
- ~1-1.5M triples/day, ~30-35M triples/month

**NOAA Weather Data** (1 mapper)
- US weather alerts (CAP format)

> **Total: 100+ mappers and ontologies** covering economic, financial, employment, and environmental data

> **Typical monthly volume: ~30-50M triples** (dominated by intraday market snapshots)

> **Note:** Raw RDF data is generated by separate Lambda scraper functions (not part of this repository). This pipeline assumes RDF data is already available in S3 in N-Triples format conforming to 100+ domain-specific ontologies.

## Project Structure

```
pyg-knowledge-graph-builder/
├── glue_jobs/
│   ├── build_graph.py                      # Main Glue job entry point
│   ├── enrichment/                         # RDF enrichment modules (PySpark)
│   │   ├── __init__.py
│   │   ├── pipeline.py                     # Main enrichment orchestrator
│   │   ├── temporal_unifier.py             # Temporal entity unification
│   │   ├── cross_source_linker.py          # Cross-source linking (BLS↔SEC↔Market↔NOAA)
│   │   ├── intra_source_linker.py          # Main intra-source entry point
│   │   ├── ontology_mapper.py              # Ontology mapping utilities
│   │   └── intra_source/                   # Intra-source enrichment modules
│   │       ├── __init__.py
│   │       ├── base.py                     # Base classes/interfaces
│   │       ├── bls_linker.py               # BLS orchestrator
│   │       ├── sec_linker.py               # SEC orchestrator
│   │       ├── market_linker.py            # Market orchestrator
│   │       ├── noaa_linker.py              # NOAA orchestrator
│   │       ├── bls/                        # BLS-specific components
│   │       │   ├── __init__.py
│   │       │   ├── patterns.py             # BLS_SECTOR_PATTERNS
│   │       │   ├── correlations.py         # KNOWN_CORRELATIONS
│   │       │   ├── measurements.py         # MEASUREMENT_TYPES
│   │       │   └── base_enricher.py        # Dataset-specific enrichers
│   │       ├── sec/                        # SEC-specific components
│   │       │   ├── __init__.py
│   │       │   ├── patterns.py             # SEC_SECTOR_PATTERNS, SEC_VIOLATION_PATTERNS
│   │       │   └── correlations.py         # SEC KNOWN_CORRELATIONS
│   │       ├── market/                     # Market-specific components
│   │       │   ├── __init__.py
│   │       │   ├── patterns.py             # MARKET_SECTOR_PATTERNS, MARKET_OPTION_PATTERNS
│   │       │   ├── correlations.py         # Market KNOWN_CORRELATIONS
│   │       │   └── measurements.py         # Market MEASUREMENT_TYPES
│   │       └── noaa/                       # NOAA-specific components
│   │           ├── __init__.py
│   │           └── patterns.py             # NOAA alert patterns
│   ├── pyg_builder/                        # PyG construction modules
│   │   ├── __init__.py
│   │   ├── constructor.py                  # Orchestrates HeteroData construction
│   │   ├── node_mapper.py                  # Assigns per-type integer node IDs on executors
│   │   ├── edge_mapper.py                  # Resolves edges to integer index tensors on executors
│   │   └── feature_extractor.py            # Ontology-aware feature vectors with VectorLayout
│   └── utils/
│       ├── __init__.py
│       └── rdf_utils.py                    # Namespace constants, URI helpers, canonical
│                                           # namespace registry (NAMESPACE_PREFIXES,
│                                           # ONTOLOGY_NAMESPACE_INDICES)
├── notebooks/
│   ├── utils/
│   │   └── invoke_helpers.py               # Helper functions
│   ├── quick_experiment.ipynb              # Quick start
│   ├── multi_experiment.ipynb              # Multi-graph workflow
│   └── experiments/
│       ├── node_types.ipynb                # Experiment with node types
│       ├── edge_types.ipynb                # Experiment with edge types
│       └── features.ipynb                  # Experiment with features
├── tests/                                  # Unit and integration tests
├── deployment/                             # Deployment scripts
│   └── cdk/
│       ├── app.py
│       ├── cdk.json
│       ├── requirements.txt
│       ├── README.md
│       └── stacks/
│           ├── __init__.py
│           ├── glue_stack.py
│           ├── s3_stack.py
│           └── iam_stack.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

### Module Roles

| Module | Role | Uses PySpark? |
|--------|------|--------------|
| `rdf_utils.py` | Namespace constants, URI string helpers, canonical `NAMESPACE_PREFIXES` and `ONTOLOGY_NAMESPACE_INDICES` registries (single source of truth for all PyG builder modules) | No (pure Python) |
| `patterns.py` / `correlations.py` / `measurements.py` | Configuration dictionaries (sector keywords, correlation definitions) | No (pure Python) |
| `pipeline.py` | Orchestrates enrichment steps, manages triples DataFrame | Yes |
| `temporal_unifier.py` | Produces unified month/year/quarter triples | Yes |
| `bls_linker.py`, `sec_linker.py`, `market_linker.py`, `noaa_linker.py` | Produce intra-source enrichment triples | Yes |
| `cross_source_linker.py` | Produces cross-source enrichment triples | Yes |
| `ontology_mapper.py` | Produces equivalence mapping triples | Yes |
| `build_graph.py` | Parses N-Triples, orchestrates pipeline modes, saves Parquet and .pt | Yes (orchestration) |
| `constructor.py` | Orchestrates PyG HeteroData construction from triples DataFrame | Yes (orchestration) |
| `node_mapper.py` | Discovers node types, assigns per-type integer IDs via Window functions. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `edge_mapper.py` | Double-joins triples with node IDs, collects edge index tensors. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `feature_extractor.py` | Builds ontology-aware vectors via `VectorLayout` (proportionally scaled segments): extracts class hierarchy, property schema, and literal values on executors; collects sparse entries (chunked for large types); scatters into dense tensors on driver. Imports `ONTOLOGY_NAMESPACE_INDICES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |

### Scalability

The pipeline is designed to handle:
- **100+ ontologies** with different schemas and vocabularies
- **30-50M triples per month** with intraday market snapshots
- **Heterogeneous data types** (prices, rates, levels, changes, categorical)
- **Multiple temporal granularities** (intraday, daily, weekly, monthly, quarterly)
- **Dynamic schema evolution** as new data sources are added
- **Horizontal scaling** by adding Glue DPUs — enrichment and PyG construction work distributes automatically
- **Bounded driver memory** — PyG construction collects only compact integer/float tensors, not URI strings. Chunked collection for large node types bounds peak Pandas memory. `gc.collect()` between types reclaims fragmented memory (safe because it runs on the driver process only, after all Spark executor work is complete).
- **Configurable vector dimension** — reducing `vector_dim` from 1024 to 512 halves driver memory for feature tensors while preserving the same three-segment structure via proportional `VectorLayout` scaling
- **No Python UDFs in the hot path** — URI-to-name conversions, hash-based encoding, and numeric parsing use pure Spark expressions (JVM-native), avoiding Python serialization overhead on 30-50M rows
- **Controlled Parquet output** — configurable partition count prevents thousands of tiny files or few huge files on S3
- **Efficient literal isolation** — anti-join against node_id_df filters out edge triples before numeric parsing, avoiding wasted computation on URI-valued objects
- **Canonical namespace registry** — `NAMESPACE_PREFIXES` and `ONTOLOGY_NAMESPACE_INDICES` in `rdf_utils.py` are the single source of truth, imported by `node_mapper.py`, `edge_mapper.py`, and `feature_extractor.py` to eliminate duplication