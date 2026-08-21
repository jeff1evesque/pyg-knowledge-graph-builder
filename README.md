# PyTorch Geometric Knowledge Graph Builder

> GPU-accelerated Apache Spark pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

[![tests](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/tests.yml)
[![unicode](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/unicode.yml/badge.svg)](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/unicode.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Apache Spark](https://img.shields.io/badge/Apache-Spark-orange.svg)](https://spark.apache.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-Accelerator-green.svg)](https://nvidia.github.io/spark-rapids/)

## Overview

PyTorch Geometric Knowledge Graph Builder is an Apache Spark pipeline that transforms raw RDF data from multiple heterogeneous sources into enriched knowledge graphs and constructs PyTorch Geometric `HeteroData` objects ready for Graph Neural Network (GNN) training.

The pipeline processes data from **100+ domain-specific ontologies** spanning economic indicators, financial filings, market data, and environmental alerts. All enrichment logic runs as **distributed PySpark DataFrame operations** on a Spark standalone cluster accelerated by the **RAPIDS Accelerator for Apache Spark** (GPU), enabling horizontal scaling across the cluster rather than bottlenecking on a single-threaded in-memory graph. Because the pipeline is UDF-free except for one small parsing step, the compute-heavy DataFrame operators (regex parsing, joins, hashing, window functions, aggregations) execute on GPU.

PyG construction also leverages Spark executors for all heavy computation (node ID assignment, edge resolution, feature extraction). Only compact integer and float tensors cross the Spark → driver boundary for final `HeteroData` assembly. All URI-to-name conversions use **pure Spark Column expressions** (JVM-native `WHEN` chains), not Python UDFs, eliminating serialization overhead.

Node feature vectors are **universal 1024-dimensional ontology-aware vectors** that encode three layers of information: ontology structure (class identity, hierarchy, source membership), property schema (presence, domain/range, property hierarchy), and literal values (numeric hashed slots, categorical multi-hot). All node types share the same vector width, enabling **shared GNN layers across heterogeneous types** and natural cross-type message passing. The vector dimension is configurable — all segment boundaries **scale proportionally** with `vector_dim`, so passing 512 produces a half-resolution vector with the same three-segment structure.

Edge feature vectors are **selective 32-dimensional derived vectors** that encode per-instance signals for high-value edge types. Only edges with meaningful per-instance variation (temporal sequences, option-stock links, severity escalations) receive features — structural edges like `belongsToSector` and `owl:sameAs` are left featureless. Edge features encode three layers: temporal signals (time delta, period flags, direction), numeric contrast (differences, ratios, magnitudes between endpoints), and relational context (namespace, label similarity, relation identity). The edge vector dimension is configurable via `edge_vector_dim`, and all segment boundaries **scale proportionally** via `EdgeVectorLayout`. Edge features are derived entirely from endpoint node properties already present in the triples — **no enrichment changes are required**.

After each PyG build, the pipeline writes **six metadata JSON files** alongside the `.pt` file. These files capture the complete graph inventory, feature vector structure, normalization statistics, encoding parameters, ontology structure, and dimension-to-meaning mappings needed for downstream GNN training and inference.

The pipeline supports three execution modes:

- **Full Pipeline**: End-to-end RDF enrichment and PyG graph construction
- **Enrichment Only**: Create reusable enriched Parquet artifacts
- **PyG Construction Only**: Rapidly experiment with different PyG graph structures from existing enriched Parquet

### Key Features

- **Large-Scale Integration**: Processes 100+ ontologies with tens of millions of triples per time period
- **Distributed Enrichment**: All enrichment runs as PySpark DataFrame operations across Spark executors
- **Distributed PyG Construction**: Node ID assignment, edge resolution, and feature extraction run on Spark executors — only compact tensors are collected to the driver
- **No Python UDFs in PyG Builder**: URI-to-name conversions use pure Spark `WHEN` expressions (JVM-native), not row-at-a-time Python UDFs
- **Ontology-Aware Node Feature Vectors**: Universal fixed-width vectors encoding class hierarchy, property schema, and literal values — not flat bags of literals
- **Derived Edge Feature Vectors**: Selective fixed-width vectors encoding temporal signals, numeric contrast, and relational context between edge endpoints — no enrichment changes required
- **Universal Node Feature Width**: All node types share the same vector dimension, enabling shared GNN layers and cross-type message passing
- **Selective Edge Featurization**: Only high-value edge types receive feature vectors; structural edges use simpler GNN message-passing layers
- **Proportionally Scalable Dimensions**: Overriding `vector_dim` or `edge_vector_dim` automatically rescales all segment and sub-segment boundaries via `VectorLayout` / `EdgeVectorLayout` — no hardcoded dim indices
- **No Double-Join for Edge Features**: Edge features reuse the cached resolved edges DataFrame from EdgeMapper — the expensive double-join runs exactly once
- **Driver Memory Safety**: Large node types use chunked collection with explicit memory management to prevent OOM
- **Six Metadata Files Per Build**: `graph_schema.json`, `feature_spec.json`, `normalization.json`, `encoding_config.json`, `ontology_schema.json`, and `slot_mapping.json` written alongside every `.pt` file (locally, and mirrored to S3 when an archive is configured) — enabling consistent training, inference, and experiment tracking
- **Node Index Per Build**: a `node_index/` Parquet dataset mapping every `(node_type, node_id)` back to its source entity URI — the `.pt` holds only feature tensors, so this is what makes the graph joinable to training labels and lets a prediction be attributed to a real entity
- **Temporal Unification**: Unified temporal entities across all data sources
- **Intra-Source Linking**: Automatic relationship discovery within data source families
- **Cross-Source Linking**: Automatic relationship discovery across heterogeneous datasets
- **PyTorch Geometric Output**: Native `HeteroData` objects with configurable node/edge types and optional edge features
- **Reusable Parquet Artifacts**: Enriched triples saved as Parquet for multiple PyG experiments without re-enrichment
- **Flexible Graph Construction**: Experiment with different graph structures from existing Parquet (5-10 min per experiment)
- **GPU-Accelerated Spark**: Runs on an Apache Spark standalone cluster with the RAPIDS Accelerator; DataFrame operators execute on GPU
- **Local-First Storage**: Interim enriched Parquet stays on a shared local filesystem; final artifacts are written locally and optionally mirrored to S3 as a durable catalog
- **Controlled Parquet Output**: Configurable partition count for optimal file sizes
- **Canonical Namespace Registry**: Single source of truth for all namespace-to-prefix mappings in `rdf_utils.py`

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Raw Data Sources (local filesystem or S3 via s3a://)       │
│                                                            │
│ N-Triples format (.nt files):                              │
│ ├── BLS Economic Data (10 categories, ~100 mappers)        │
│ ├── Market Data (1 mapper, intraday snapshots)             │
│ └── NOAA Weather Alerts (1 mapper)                         │
│                                                            │
│ Turtle Parquet format (column of Turtle blobs):            │
│ └── SEC Data (4 categories, 4 mappers)                     │
│     (and any other source whose scraper writes Parquet)    │
│                                                            │
│ Total: 100+ mappers and ontologies                         │
│ Volume: ~30-50M triples/month with intraday market data    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Spark job (spark-submit): pyg-knowledge-graph-builder      │
│ Spark standalone cluster + RAPIDS Accelerator (GPU)        │
│                                                            │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────┐   │
│  │ Parse        │──▶│ Enrichment    │──▶│ Build PyG    │   │
│  │ Source RDF   │   │ (PySpark      │   │ (PySpark     │   │
│  │ (N-Triples:  │   │  DataFrames   │   │  executors   │   │
│  │  Spark regex │   │  on executors,│   │  on GPU      │   │
│  │  on executors│   │  GPU via      │   │  → driver    │   │
│  │ Turtle Parq: │   │  RAPIDS)      │   │  tensors)    │   │
│  │  rdflib UDF  │   │               │   │              │   │
│  │  on executors│   │               │   │              │   │
│  │  → triples   │   │               │   │              │   │
│  │  DataFrame)  │   │               │   │              │   │
│  └──────────────┘   └───────────────┘   └──────────────┘   │
│                                                            │
│ Mode 1: Full Pipeline                                      │
│   Source RDF → triples_df → Enrich → Save Parquet (local)  │
│   → Build PyG HeteroData → Save .pt + metadata JSON        │
│                                                            │
│ Mode 2: Enrichment Only                                    │
│   Source RDF → triples_df → Enrich → Save Parquet (local)  │
│                                                            │
│ Mode 3: PyG Only                                           │
│   Enriched Parquet (local) → triples_df → Build PyG        │
│   HeteroData → Save .pt + metadata JSON                    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ Outputs                                                    │
│ ├── Enriched Triples (Parquet) - local, reusable artifact  │
│ ├── PyTorch Geometric HeteroData (.pt) - local + optional  │
│ │     S3 archive - GNN ready                               │
│ └── Metadata JSON files (6 files per build) - local +      │
│     optional S3 archive - training / inference support     │
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
│ market:AAPL_20241115T143000Z    │ rdf:type             │ market:EquitySnap  │
│ market:AAPL_20241115T143000Z    │ market:lastPrice     │ 191.45             │
│ market:AAPL_20241115T143000Z    │ market:symbol        │ AAPL               │
│ market:AAPL_20241115T143000Z    │ market:captureTime   │ 2024-11-15T14:30Z  │
└─────────────────────────────────┴──────────────────────┴────────────────────┘
```

**N-Triples Parsing**: Raw `.nt` files are read as text by `spark.read.text()` and parsed on executors using Spark regex functions (`regexp_extract`). Subject and predicate URIs are extracted from angle brackets, and object values are cleaned (URI angle brackets stripped, literal datatype suffixes and language tags removed). No data passes through the driver during parsing.

Enrichment steps read from this DataFrame, produce new triples DataFrames, and union them back. The enriched DataFrame is saved as **Parquet** for reuse. PyG construction reads the enriched DataFrame, assigns integer node IDs, resolves edges, extracts node and edge features — all on Spark executors. Only compact tensors cross to the driver for final `HeteroData` assembly. After the `.pt` file is saved, six metadata JSON files are written to a `metadata/` subdirectory alongside it.

### Why PySpark Instead of rdflib/SPARQL

| Aspect | rdflib + SPARQL | PySpark DataFrames |
|--------|----------------|-------------------|
| Execution | Single Python process on the driver | Distributed across all Spark executors (GPU via RAPIDS) |
| Memory | Entire graph must fit in driver RAM | Partitioned across cluster |
| Query optimization | None (sequential iteration) | Catalyst optimizer, predicate pushdown, broadcast joins |
| Parallelism | None | Automatic partitioning |
| Hardware utilization | Uses 1 core | Uses all executor cores and GPUs |
| Join pattern | Python dict lookups or nested SPARQL | Distributed hash/sort-merge joins |

rdflib Namespace objects are used as **URI string constants** in the enrichment modules for readability — they produce plain strings and don't hold or query graph data. The PyG builder modules use **pure Spark Column expressions** for all URI-to-name conversions (no Python UDFs).

### Namespaces: whose terms are whose

`rdf_utils.py` holds two kinds of namespace, and the distinction is not cosmetic — a URI names the authority for the term.

**Publishers' vocabularies** (`cpi:`, `ppi:`, `jolts:`, `cap:`, `nws:`, `sec.gov/filings#`, …) stay on their own domains. Those really are their terms.

**Terms this project invents** all live under one base we control, sub-pathed by concern:

```
https://jefflevesque.com/ontology/bls/          RateMeasurement, PriceIndex, coversMonth, …
https://jefflevesque.com/ontology/sec/
https://jefflevesque.com/ontology/noaa/         EmergencyAlert, AlertArea, AlertInfo
https://jefflevesque.com/ontology/market/
https://jefflevesque.com/ontology/unified/      hasMonth, hasYear, measurementValue, …
https://jefflevesque.com/ontology/temporal/     SourceMonth, SourceYear, SourceQuarter
https://jefflevesque.com/ontology/provenance/   derivedBy, route markers (never encoded)
```

These previously sat under the publishers' domains (`bls.gov/enrichment/`, `sec.gov/enrichment/`, `noaa.gov/enrichment/`, `financial-data.org/enrichment/`) and under `example.org`. Both were wrong, differently: a URI under `bls.gov` asserts BLS defined `RateMeasurement` — nobody there did, and federating this graph with real BLS-published RDF would merge our invention into their vocabulary. `example.org` is reserved by RFC 2606 for documentation, so it cannot misattribute, but it belongs to nobody and reads as an unfinished placeholder.

The base is a single constant, `ONTOLOGY_BASE`, so re-homing the vocabulary is a one-line edit — but not a free one: **these URIs are hashed into feature slots**, so moving them moves every slot and changes `encoding_config.json`'s contract digest. That is the intended signal (graphs built either side of the move are not comparable), not a side effect. `slot_mapping.json` records the URI→slot mapping per build, so older artifacts stay interpretable.

**Prefixes were deliberately left unchanged** (`bls_enrichment`, `unified`, `temporal`, …), so node type names, `graph_schema.json`, `node_index/` and every edge-type triple are identical across the move. Only the URIs — and therefore the slots — differ.

`tests/test_namespaces.py` asserts the invariants that would otherwise fail silently: no minted namespace under a publisher's or a reserved domain; prefixes unique; where one namespace is a string prefix of another the longer is ordered first (`startsWith` matching would otherwise name the node type after the wrong vocabulary); and the literal type names in `_CANONICAL_TYPE_PRIORITY` are still producible by the naming rule from a registered namespace.

## Ontology-Aware Node Feature Vectors

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
│ (160 dims @ 1024)  │ (48 dims @ 1024)   │ (48 dims @ 1024)   │
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
- **Domain/Range Signals**: For each property this node has, its `rdfs:domain` and `rdfs:range` are hashed. This tells the GNN what types of relationships this node can participate in. No source declares either, so both are **derived** — the domain from the observed `rdf:type` of the property's subjects, the range from its objects' types and from the XSD datatype the source declared on its literals. A property used on more than one class gets neither, because `rdfs:domain` is an intersection; see [`ontology_schema.json`](#ontology_schemajson) for the provenance and coverage this publishes.
- **Property Hierarchy**: `rdfs:subPropertyOf` relationships are hashed, connecting specific properties to their abstract parents. Also derived — from the `PROPERTY_MAPPINGS` entries whose target several properties share (nine point at `unified:measurementValue`, and a rate is not a price, so they are sub-properties rather than equivalents).

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

#### Numeric vs. categorical is decided per property, not per value

**A property is numeric or categorical, never both.** A predicate is numeric only when *more than* `feature_config.numeric_predicate_min_share` (default `0.5`) of its literal values parse as a number; otherwise every one of its values — including any that happen to parse — is encoded as a category label.

Classifying each *value* independently splits one property across both sub-segments whenever its labels are not uniformly shaped. SEC `hasDocumentType` is the motivating case: 315 of its 2,372 values (13.3%) are bare-digit form types — Form `4`, `144`, `3`, `425`, `497`, `487`, `25` — while the rest are hyphenated (`10-K`, `8-K`, `S-1`). Those 315 were z-scored into the numeric sub-segment as if a form number were a magnitude (mean 62.24, std 128.64), inventing a continuous ordering over labels, while the other 2,057 were correctly multi-hot encoded.

A simple majority is deliberate: it is the least presumptuous rule that fixes the above, and it lets a genuinely numeric measurement carry a minority of unparseable sentinels (`"N/A"`, `"unknown"`) without demoting the whole property out of the numeric sub-segment — those sentinels are then dropped as missing data rather than re-encoded as labels, exactly as an absent property is. The threshold is recorded in `encoding_config.json` under `numeric_values.predicate_min_numeric_share`, since it determines which sub-segment a property is encoded into.

### Proportional Dimension Scaling via VectorLayout

All segment and sub-segment boundaries are computed at runtime by the `VectorLayout` class from the configured `vector_dim`. No dim indices are hardcoded in the encoding logic. This means overriding `vector_dim` from a notebook invocation automatically produces a correctly structured vector at the requested resolution:

```
VectorLayout(1024) — default, production:
  Segment 1: Ontology Structure [0–255]     (256 dims)
    Class Identity:     [0–159]             (160 dims)
    Class Hierarchy:    [160–207]           (48 dims)
    Ontology Source:    [208–255]           (48 dims)
  Segment 2: Property Schema    [256–639]   (384 dims)
    Property Presence:  [256–447]           (192 dims)
    Domain/Range:       [448–558]           (111 dims)
    Property Hierarchy: [559–639]           (81 dims)
  Segment 3: Literal Values     [640–1023]  (384 dims)
    Numeric Values:     [640–896]           (257 dims)
    Categorical Values: [897–1023]          (127 dims)

VectorLayout(512) — half resolution, faster experiments:
  Segment 1: Ontology Structure [0–127]     (128 dims)
    Class Identity:     [0–79]              (80 dims)
    Class Hierarchy:    [80–103]            (24 dims)
    Ontology Source:    [104–127]           (24 dims)
  Segment 2: Property Schema    [128–319]   (192 dims)
    Property Presence:  [128–223]           (96 dims)
    Domain/Range:       [224–279]           (56 dims)
    Property Hierarchy: [280–319]           (40 dims)
  Segment 3: Literal Values     [320–511]   (192 dims)
    Numeric Values:     [320–448]           (129 dims)
    Categorical Values: [449–511]           (63 dims)

VectorLayout(256) — quarter resolution, rapid prototyping:
  Segment 1: Ontology Structure [0–63]      (64 dims)
    Class Identity:     [0–39]              (40 dims)
    Class Hierarchy:    [40–51]             (12 dims)
    Ontology Source:    [52–63]             (12 dims)
  Segment 2: Property Schema    [64–159]    (96 dims)
    Property Presence:  [64–111]            (48 dims)
    Domain/Range:       [112–139]           (28 dims)
    Property Hierarchy: [140–159]           (20 dims)
  Segment 3: Literal Values     [160–255]   (96 dims)
    Numeric Values:     [160–223]           (64 dims)
    Categorical Values: [224–255]           (32 dims)

VectorLayout(2048) — double resolution, maximum fidelity:
  Segment 1: Ontology Structure [0–511]     (512 dims)
    Class Identity:     [0–319]             (320 dims)
    Class Hierarchy:    [320–415]           (96 dims)
    Ontology Source:    [416–511]           (96 dims)
  Segment 2: Property Schema    [512–1279]  (768 dims)
    Property Presence:  [512–895]           (384 dims)
    Domain/Range:       [896–1118]          (223 dims)
    Property Hierarchy: [1119–1279]         (161 dims)
  Segment 3: Literal Values     [1280–2047] (768 dims)
    Numeric Values:     [1280–1794]         (515 dims)
    Categorical Values: [1795–2047]         (253 dims)
```

`VectorLayout` validates at construction time that all sub-segments are contiguous, non-overlapping, each has at least 1 dimension, and they sum exactly to `vector_dim`. If `vector_dim` is too small (< 32), it raises immediately with a clear error rather than silently producing a degenerate vector.

**Tradeoffs when reducing `vector_dim`:**

| vector_dim | Hash collision risk | Class ceiling (class_identity dim) | Driver memory per 1M nodes | Use case |
|-----------|-------------------|---------------------------------|--------------------------|----------|
| 2048 | Very low | 320 | ~8 GB | Maximum fidelity, large cluster |
| 1024 | Low (~10 properties/type vs 256 numeric slots) | 160 | ~4 GB | Production default |
| 512 | Moderate (128 numeric slots) | 80 | ~2 GB | Fast experiments |
| 256 | Higher (64 numeric slots) | 40 | ~1 GB | Rapid prototyping, small datasets |

`class_identity` gets 15.625% of `vector_dim` (62.5% of segment 1), and it holds **at most** that many linearly independent class codes — so **the class count, not the property count, is what `vector_dim` has to clear** for class identity to stay recoverable. A full-source fixture build produces 96 classes, which fits the 1024-d default with 64 classes of nominal headroom.

Treat that ceiling as an upper bound, not a target. It is what pigeonhole forbids exceeding, not what the hashing achieves: measured, `d` 4-hot codes drawn into `d` dims come out rank `d−2`, so separability has to be **measured** below the ceiling rather than assumed. The build does exactly that — see `code_matrix_rank` in the slot-mapping collision report.

Two levers when the class count grows past what the default carries:

- **`feature_config.class_identity_dim`** — sets the sub-segment width directly, taken from the rest of segment 1, so `vector_dim` and driver memory do not move. Rejected (not clamped) if it does not fit, since the point of the override is to guarantee a budget.
- **`feature_config.vector_dim`** — scales every segment, at proportional memory cost.

Segment 1 is a quarter of the vector, so no split of it exceeds ~192 classes at the 1024-d default; past that, `vector_dim` is the only lever.

A build whose classes are **not separable** now **fails** with `ClassIdentityCapacityError` rather than logging a warning — over-subscribed, sharing an identical code, or linearly dependent. Set `feature_config.allow_class_identity_oversubscription=true` to build anyway. The build still only *warns* when the class count passes 85% of the segment, which is a nudge rather than a fault.

> **Changing `class_identity_dim` or `vector_dim` invalidates trained models.** Slots are `hash % dim`, so a different width re-maps every class. This is why the width is a published tuning constant in `encoding_config.json` and not derived per build from the observed class count — a width that moved whenever a new class appeared would silently re-map every existing one.

**Invocation example:**

```python
# Quick experiment with half-resolution vectors
config = {
    "feature_config": {
        "vector_dim": 512,
        "normalize": True
    }
}

# Passed to the job as:
#   --pyg_config "$(python -c 'import json,sys; ...')"
"--pyg_config": json.dumps(config)
```

### Why This Is Better for GNNs

```
OLD approach (flat literal vectors):
┌────────────────────────────────────────────────────────────┐
│ cpi_Index:  [295.8, 0.3, 2.1, 0.05, 3.0, 1.0]              │  6 dims, only literals
│ ppi_Index:  [187.2, 0.1, 1.5, 0.03, 2.0, 1.0]              │  6 dims, only literals
│                                                            │
│ SEPARATE tensors per type (different widths)               │
│ GNN needs type-specific linear layers                      │
│ No cross-type weight sharing possible                      │
│ Zero = missing? or inapplicable? GNN can't tell            │
└────────────────────────────────────────────────────────────┘

NEW approach (ontology-aware vectors):
┌────────────────────────────────────────────────────────────┐
│ cpi_Index:  [ontology:25% | schema:37.5% | lit:37.5%]      │  1024-d, universal
│ ppi_Index:  [ontology:25% | schema:37.5% | lit:37.5%]      │  1024-d, universal
│                                                            │
│ SAME tensor width for ALL node types                       │
│ Shared ontology bits where types share ancestry            │
│ GNN can use SHARED layers across all types                 │
│ Cross-type message passing works naturally                 │
│ Property presence distinguishes missing vs N/A             │
│ Override vector_dim for memory/fidelity tradeoff           │
└────────────────────────────────────────────────────────────┘
```

### All Node Encoding Runs on Spark Executors

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

## Derived Edge Feature Vectors

### The Problem With Featureless Edges

In a basic heterogeneous graph, edges carry only their type label (e.g., `("cpi_Index", "bls_enrichment_precedes", "cpi_Index")`). The GNN learns a single weight matrix per edge type, applied identically to all edges of that type. This has limitations:

- **No per-instance variation**: A `precedes` edge spanning 1 month is treated identically to one spanning 12 months
- **No endpoint contrast**: An option-stock edge where the option is deep in-the-money looks the same as one that is far out-of-the-money
- **No cross-source signal**: A `correlatesWith` edge between two CPI categories is indistinguishable from one linking CPI to PPI
- **Wasted information**: Endpoint node properties that could inform message passing are ignored until the GNN aggregates them — edge features let the GNN modulate messages *before* aggregation

### The Solution: Selective Derived Edge Feature Vectors

Edge features are **selective** — only edge types with meaningful per-instance variation receive feature vectors. Edge types where the relation name alone carries sufficient signal are left featureless. This is a deliberate design choice: adding features to `belongsToSector` or `owl:sameAs` edges would waste memory on constant vectors that carry no information beyond what the edge type already encodes.

Every featurized edge gets a fixed-width vector (default 32-d) with three segments derived entirely from endpoint node properties. All segment boundaries are computed proportionally by `EdgeVectorLayout`, so the structure scales to any `edge_vector_dim`:

```
Default 32-dimensional edge feature vector
┌─────────────────────┬──────────────────────┬─────────────────────┐
│ Temporal Signals    │ Numeric Contrast     │ Relational Context  │
│ (time delta,        │ (differences, ratios,│ (namespace, label   │
│  period flags,      │  magnitudes between  │  similarity,        │
│  direction)         │  endpoints)          │  relation identity) │
│                     │                      │                     │
│ 37.5% of edge_dim   │ 37.5% of edge_dim    │ 25% of edge_dim     │
│ (12 dims @ 32)      │ (12 dims @ 32)       │ (8 dims @ 32)       │
└─────────────────────┴──────────────────────┴─────────────────────┘
```

#### Which Edge Types Get Features

Edge types are classified by relation name into categories. Only categories in the enabled set receive feature vectors:

| Category | Example Relations | Default | Key Signals |
|----------|------------------|---------|-------------|
| **temporal** | `precedes`, `follows`, `hasNext` | **ON** | Month delta, same-year flag, consecutive-month flag, direction |
| **option_stock** | `hasUnderlyingPriceObservation` | **ON** | Moneyness (strike/stock), log-moneyness, strike-stock difference |
| **escalation** | `escalatesTo`, `severityChange` | **ON** | Severity delta between alerts |
| **correlation** | `correlatesWith`, `relatedTo`, `*Correlation` | OFF | Label Jaccard similarity, same-namespace flag |
| **causal** | `leadsTo`, `impacts`, `affects` | OFF | Label similarity, cross-source indicator |
| **strategy** | `straddleWith`, `spreadWith` | OFF | Strike distance, same-expiry flag |
| **generic** | anything no fragment matched | OFF | Category indicator hash, relational context |
| **skip** (never featurized) | `belongsToSector`, `owl:sameAs`, `hasParent` | — | Relation name alone is sufficient |

The `*Correlation` fragment matters more than it looks: the cross-source linkers emit one
relation per sector (`energySectorCorrelation`, `employmentSizeSectorCorrelation`, …), so
matching only the literal names `correlatesWith` / `relatedTo` classified all of them
**generic** — which no `enabled_categories` value could select. Every such edge type was
silently dropped from featurization while the job logged `Building 32-d edge feature
vectors` and exited 0.

`generic` is the fallback for relations no fragment matched, so enabling it featurizes
essentially every non-skip edge type in the graph. It is selectable, but off by default.
A category name that is not in this table now raises at construction rather than producing
an empty result an hour later, and a run that featurizes **nothing** while edge features
are enabled logs a `WARNING` naming the categories the graph actually contains.

#### Segment 1: Temporal Signals (37.5% of edge_vector_dim)

Encodes **when** the edge endpoints exist relative to each other. For temporal edges, this captures the time gap, periodicity, and direction. For non-temporal edges, a category indicator hash is placed in this segment so the GNN can still distinguish edge categories in this segment.

```
Segment 1: Temporal Signals [37.5% of edge_vector_dim]
┌────────────────────┬────────────────────┬────────────────────┐
│ Time Delta         │ Period Flags       │ Direction          │
│ (signed normalized │ (same-year,        │ (forward/backward  │
│  month delta,      │  consecutive-month,│  temporal direction│
│  absolute delta)   │  same-quarter)     │  indicator)        │
│ 40% of segment     │ 35% of segment     │ 25% of segment     │
│ (5 dims @ 32)      │ (4 dims @ 32)      │ (3 dims @ 32)      │
└────────────────────┴────────────────────┴────────────────────┘
```

- **Time Delta**: Month delta between endpoints computed as `(dst_year - src_year) * 12 + (dst_month - src_month)`, normalized by dividing by 12. Both signed and absolute values are encoded in hashed slots. A 1-month gap = 0.083, a 1-year gap = 1.0.
- **Period Flags**: Binary indicators — same-year (1.0 if both endpoints share the same year), consecutive-month (1.0 if exactly 1 month apart), same-quarter (1.0 if same calendar quarter and year).
- **Direction**: +1.0 if destination is later in time, -1.0 if earlier, 0.0 if same or unknown.

#### Segment 2: Numeric Contrast (37.5% of edge_vector_dim)

Encodes **how** the numeric properties of the two endpoints differ. For edges where both endpoints share the same predicate URI (e.g., two `cpi:Index` nodes both having `indexValue`), computes differences, ratios, and magnitudes. For edges with semantically related but differently-named properties (e.g., option `strikePrice` vs. stock `observedPrice`), uses cross-property derivation.

```
Segment 2: Numeric Contrast [37.5% of edge_vector_dim]
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Difference          │ Ratio               │ Magnitude           │
│ (dst_val - src_val  │ (dst_val / src_val, │ (average absolute   │
│  per shared         │  clamped to         │  value of both      │
│  property)          │  [-10, 10])         │  endpoints)         │
│ 40% of segment      │ 35% of segment      │ 25% of segment      │
│ (5 dims @ 32)       │ (4 dims @ 32)       │ (3 dims @ 32)       │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

- **Difference**: `dst_value - src_value` for each shared numeric property, hashed into a fixed slot by predicate URI. For temporal edges, this captures how much an indicator changed. For option-stock edges (cross-property), this encodes the strike-stock price difference.
- **Ratio**: `dst_value / src_value`, clamped to [-10, 10] to avoid extreme values from near-zero denominators. For option-stock edges, this encodes moneyness (strike/stock_price) and log-moneyness.
- **Magnitude**: Average absolute value of both endpoints — provides scale context so the GNN can distinguish a 1-point change on a 300-point index from a 1-point change on a 10-point index.

**Cross-property derivation** for specific edge categories:

| Category | Source Property | Destination Property | Derived Signals |
|----------|----------------|---------------------|-----------------|
| option_stock | `strikePrice` | `observedPrice` | Moneyness, log-moneyness, strike-stock difference |
| escalation | `severity` / `severityLevel` | `severity` / `severityLevel` | Severity delta (positive = escalation) |

#### Segment 3: Relational Context (25% of edge_vector_dim)

Encodes **what kind** of relationship this edge represents and whether it crosses ontology boundaries.

```
Segment 3: Relational Context [25% of edge_vector_dim]
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Namespace Signals   │ Label Similarity    │ Relation Identity   │
│ (same-namespace     │ (Jaccard word       │ (relation name +    │
│  flag, cross-source │  overlap of         │  category hash)     │
│  flag, ns hashes)   │  endpoint labels)   │                     │
│ 40% of segment      │ 35% of segment      │ 25% of segment      │
│ (3 dims @ 32)       │ (3 dims @ 32)       │ (2 dims @ 32)       │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

- **Namespace Signals**: Same-namespace flag (1.0 if both endpoints are from the same ontology, derived from PyG node type prefix), cross-source flag (inverse), and hashed namespace identity for finer-grained source encoding. These are driver-side string operations on type names, broadcast as literals — not Spark UDFs.
- **Label Similarity**: For correlation and causal edges, Jaccard word overlap between endpoint `rdfs:label` values computed via Spark array functions (`array_intersect`, `array_union`). Distinguishes strong correlations (exact keyword match like "Energy" ↔ "Energy") from weak ones ("Food at Home" ↔ "Food Manufacturing"). For other edge types, a category indicator hash is used instead.
- **Relation Identity**: The relation name and category are each hashed into fixed slots (relation at weight 1.0, category at weight 0.5). Edges of the same specific relation share strong bits; edges of the same category share weaker bits.

### Proportional Dimension Scaling via EdgeVectorLayout

All segment and sub-segment boundaries are computed at runtime by the `EdgeVectorLayout` class from the configured `edge_vector_dim`. No dim indices are hardcoded in the encoding logic:

```
EdgeVectorLayout(32) — default, production:
  Segment 1: Temporal Signals  [0–11]    (12 dims)
    Time Delta:        [0–4]             (5 dims)
    Period Flags:      [5–8]             (4 dims)
    Direction:         [9–11]            (3 dims)
  Segment 2: Numeric Contrast  [12–23]   (12 dims)
    Difference:        [12–16]           (5 dims)
    Ratio:             [17–20]           (4 dims)
    Magnitude:         [21–23]           (3 dims)
  Segment 3: Relational Context [24–31]  (8 dims)
    Namespace:         [24–26]           (3 dims)
    Label Similarity:  [27–29]           (3 dims)
    Relation Identity: [30–31]           (2 dims)

EdgeVectorLayout(64) — double resolution:
  Segment 1: Temporal Signals  [0–23]    (24 dims)
    Time Delta:        [0–9]             (10 dims)
    Period Flags:      [10–17]           (8 dims)
    Direction:         [18–23]           (6 dims)
  Segment 2: Numeric Contrast  [24–47]   (24 dims)
    Difference:        [24–33]           (10 dims)
    Ratio:             [34–41]           (8 dims)
    Magnitude:         [42–47]           (6 dims)
  Segment 3: Relational Context [48–63]  (16 dims)
    Namespace:         [48–53]           (6 dims)
    Label Similarity:  [54–59]           (6 dims)
    Relation Identity: [60–63]           (4 dims)

EdgeVectorLayout(16) — half resolution, minimal overhead:
  Segment 1: Temporal Signals  [0–5]     (6 dims)
    Time Delta:        [0–1]             (2 dims)
    Period Flags:      [2–3]             (2 dims)
    Direction:         [4–5]             (2 dims)
  Segment 2: Numeric Contrast  [6–11]    (6 dims)
    Difference:        [6–7]             (2 dims)
    Ratio:             [8–9]             (2 dims)
    Magnitude:         [10–11]           (2 dims)
  Segment 3: Relational Context [12–15]  (4 dims)
    Namespace:         [12–12]           (1 dim)
    Label Similarity:  [13–13]           (1 dim)
    Relation Identity: [14–15]           (2 dims)
```

`EdgeVectorLayout` validates at construction time that all sub-segments are contiguous, non-overlapping, each has at least 1 dimension, and they sum exactly to `edge_vector_dim`. If `edge_vector_dim` is too small (< 9), it raises immediately with a clear error.

**Tradeoffs when adjusting `edge_vector_dim`:**

| edge_vector_dim | Driver memory per 1M edges | Hash collision risk | Use case |
|----------------|---------------------------|-------------------|----------|
| 64 | ~256 MB | Very low | Maximum edge signal fidelity |
| 32 | ~128 MB | Low | Production default |
| 16 | ~64 MB | Moderate | Minimal overhead, large edge counts |

### Why Edge Features Require Zero Enrichment Changes

Edge features are derived entirely from properties already present in the enriched triples:

1. **EdgeMapper** resolves each edge's subject and object URIs to integer node IDs via a double-join against the node ID table. This produces a cached resolved edges DataFrame with `(src_type, src_id, relation, dst_type, dst_id)`.

2. **EdgeFeatureExtractor** receives this cached DataFrame directly from the constructor — **no double-join replay**. It joins endpoint node IDs against the literal triples to access numeric properties and labels, all on Spark executors.

3. Temporal signals are derived from month/year properties that nodes already have (e.g., `cpi:hasMonth`, `cpi:hasYear`). Numeric contrast is derived from literal properties (e.g., `cpi:indexValue`, `market:strikePrice`). Label similarity uses existing `rdfs:label` values.

No new triples, no new predicates, no changes to any enrichment module.

### All Edge Encoding Runs on Spark Executors

Every edge encoding operation uses pure Spark expressions — no Python UDFs:

| Encoding | Spark Operation | Example |
|----------|----------------|---------|
| Month delta | `(dst_year - src_year) * 12 + (dst_month - src_month)` | Signed month distance between endpoints |
| Delta normalization | `F.col("month_delta") / F.lit(12.0)` | Normalize to ~[-1, 1] range |
| Period flags | `F.when(condition, F.lit(1.0)).otherwise(F.lit(0.0))` | Same-year, consecutive-month, same-quarter |
| Direction indicator | `F.when(delta > 0, 1.0).when(delta < 0, -1.0).otherwise(0.0)` | Forward/backward temporal direction |
| Numeric difference | `F.col("dst_val") - F.col("src_val")` | Per-property difference between endpoints |
| Safe ratio | `F.greatest(F.lit(-10.0), F.least(F.lit(10.0), dst/src))` | Clamped ratio avoiding division by zero |
| Moneyness | `F.col("strike_price") / F.col("stock_price")` | Option-stock cross-property derivation |
| Log-moneyness | `F.log(F.greatest(moneyness, F.lit(1e-8)))` | Symmetric around ATM (log(1) = 0) |
| Namespace flag | `F.lit(1.0)` / `F.lit(0.0)` (driver-side string compare, broadcast) | Same-namespace indicator |
| Label Jaccard | `F.size(F.array_intersect(...)) / F.size(F.array_union(...))` | Word overlap between endpoint labels |
| Relation hash | `F.abs(F.hash(F.lit(relation), F.lit(seed))) % dim + offset` | Relation name → fixed slots |
| Category hash | `F.abs(F.hash(F.lit(category), F.lit(seed))) % dim + offset` | Category → fixed slots at half weight |
| Edge idx assignment | `F.row_number().over(Window.partitionBy(...).orderBy("src_id", "dst_id")) - 1` | Deterministic alignment with edge_index |

All `dim` and `offset` values are read from `EdgeVectorLayout` at runtime — they scale with `edge_vector_dim`.

### Why This Is Better for GNNs

```
WITHOUT edge features:
┌────────────────────────────────────────────────────────────┐
│ precedes edge (1 month gap):   no features                 │
│ precedes edge (12 month gap):  no features                 │
│ → GNN treats both identically during message passing       │
│                                                            │
│ option→stock (deep ITM):       no features                 │
│ option→stock (far OTM):        no features                 │
│ → GNN cannot modulate messages by moneyness                │
│                                                            │
│ correlatesWith (CPI↔CPI):      no features                 │
│ correlatesWith (CPI↔PPI):      no features                 │
│ → GNN cannot distinguish intra-source from cross-source    │
└────────────────────────────────────────────────────────────┘

WITH selective edge features:
┌────────────────────────────────────────────────────────────┐
│ precedes edge (1 month):  [delta=0.08 | same_yr=1 | ...]   │  32-d
│ precedes edge (12 month): [delta=1.00 | same_yr=0 | ...]   │  32-d
│ → GNN can learn time-decay attention weights               │
│                                                            │
│ option→stock (deep ITM):  [moneyness=0.7 | log_m=-0.36]    │  32-d
│ option→stock (far OTM):   [moneyness=1.5 | log_m=0.41]     │  32-d
│ → GNN can modulate option-stock messages by moneyness      │
│                                                            │
│ correlatesWith (CPI↔CPI): [same_ns=1 | sim=0.8 | ...]      │  32-d
│ correlatesWith (CPI↔PPI): [same_ns=0 | sim=0.3 | ...]      │  32-d
│ → GNN can weight intra-source correlations differently     │
│                                                            │
│ belongsToSector:           no features (not needed)        │
│ owl:sameAs:                no features (not needed)        │
│ → Structural edges use simpler message-passing layers      │
└────────────────────────────────────────────────────────────┘
```

## PyG Construction Pipeline

The PyG builder converts the enriched triples DataFrame into a PyTorch Geometric `HeteroData` object through five steps, with all heavy computation on Spark executors. After the `.pt` file is saved, six metadata JSON files are written alongside it (locally, and mirrored to S3 when an archive is configured):

```
triples_df (enriched, on executors)
    │
    ├── Step 1: NodeMapper (on executors)
    │   ├── Discover node types from rdf:type triples
    │   ├── Filter out meta-ontology types (OWL, RDFS, RDF)
    │   ├── Convert type URIs to PyG names via pure Spark WHEN expressions
    │   ├── Assign canonical type per entity (pinned temporal types first, then
    │   │   most specific wins via type count)
    │   ├── Assign per-type 0-indexed integer IDs via Window functions
    │   ├── Cache and materialize node_id_df on executors
    │   ├── Collect type URI mapping for metadata (small collect, <500 rows)
    │   └── Output: node_id_df (uri, node_id, node_type) — cached on executors
    │              node_counts Dict[str, int] — small collect to driver
    │              node_type_uris Dict[str, str] — deposited into MetadataCollector
    │
    ├── Step 2: EdgeMapper (on executors → driver tensors + cached DataFrame)
    │   ├── Exclude structural predicates (rdf:type, rdfs:label, etc.)
    │   ├── Double-join triples with node_id_df (subject → src_id, object → dst_id)
    │   ├── Inner join on object naturally filters out literal properties
    │   ├── Derive relation names via pure Spark WHEN expressions (no UDF)
    │   ├── Cache resolved edges DataFrame (reused by EdgeFeatureExtractor)
    │   ├── Discover distinct edge types (small collect)
    │   ├── Collect per-edge-type [2, num_edges] int64 arrays via toPandas()
    │   │   in deterministic order (src_id ASC, dst_id ASC)
    │   ├── Release Pandas memory after each edge type conversion
    │   ├── Collect predicate URI mapping for metadata (small collect, <100 rows)
    │   └── Output: Dict[(src_type, relation, dst_type) → LongTensor]
    │              edges_final_df — cached on executors for Step 4
    │              edge_predicate_uris Dict[str, str] — deposited into MetadataCollector
    │
    ├── Step 3: FeatureExtractor (on executors → driver tensors)
    │   ├── Compute VectorLayout from configured vector_dim (all boundaries
    │   │   scale proportionally — no hardcoded dim indices)
    │   ├── Extract ontology structure from triples (rdfs:subClassOf chains,
    │   │   rdfs:domain/range, rdfs:subPropertyOf) — all on executors
    │   ├── Compute per-node property presence via join — on executors
    │   ├── Isolate literal triples via anti-join — on executors
    │   ├── Classify each predicate numeric or categorical by the share of its
    │   │   values that cast("double") (small collect, one row per predicate)
    │   ├── Route each predicate's literals to its one sub-segment — on executors
    │   ├── Compute per-predicate z-score stats (single-pass agg) — on executors
    │   ├── Collect normalization stats for metadata (small collect, <200 rows)
    │   ├── Collect ontology schema snapshot for metadata (small collects:
    │   │   type URIs ~500 rows, class hierarchy ~5000 rows, property
    │   │   schema ~500 rows)
    │   ├── Compute slot mapping on driver (hash approximation, <1000 entries)
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
    │              VectorLayout.to_dict() — deposited into MetadataCollector
    │              normalization_stats, ontology_schema, slot_mapping
    │              — all deposited into MetadataCollector
    │
    ├── Step 4: EdgeFeatureExtractor (on executors → driver tensors)
    │   ├── Compute EdgeVectorLayout from configured edge_vector_dim
    │   │   (all boundaries scale proportionally)
    │   ├── Classify each edge type by relation name into categories
    │   │   (temporal, option_stock, escalation, correlation, causal,
    │   │    strategy, skip, generic)
    │   ├── Filter to eligible edge types (enabled categories only)
    │   ├── Assign deterministic edge_idx via Window functions on executors
    │   │   (same sort order as EdgeMapper: src_id ASC, dst_id ASC)
    │   ├── Extract endpoint numeric properties via anti-join — on executors
    │   ├── Extract endpoint labels (rdfs:label) — on executors
    │   ├── For each eligible edge type:
    │   │   ├── Filter resolved edges to this type (on executors)
    │   │   ├── Encode Segment 1: temporal signals (time delta, period flags,
    │   │   │   direction) or category indicator for non-temporal edges
    │   │   ├── Encode Segment 2: numeric contrast (difference, ratio,
    │   │   │   magnitude) or cross-property derivation (moneyness, severity)
    │   │   ├── Encode Segment 3: namespace signals + label similarity +
    │   │   │   relation identity hash
    │   │   ├── Union segments, aggregate (sum at same edge_idx+dim) — on executors
    │   │   ├── Pre-allocate dense numpy array on driver
    │   │   ├── Collect sparse entries via toPandas() (chunked for large types)
    │   │   ├── Scatter into dense array, delete Pandas, gc.collect()
    │   │   └── Convert numpy → torch (zero-copy via from_numpy)
    │   ├── Reuses cached edges_final_df from EdgeMapper — no double-join replay
    │   ├── Deposit EdgeVectorLayout.to_dict(), encoding config, and edge
    │   │   classification into MetadataCollector
    │   └── Output: Dict[(src_type, rel, dst_type) → FloatTensor[num_edges, edge_vector_dim]]
    │              Only contains entries for edge types that received features
    │
    ├── Step 5: Assemble HeteroData (on driver)
    │   ├── Only compact tensors on driver — no URI strings
    │   ├── Attach node feature tensors per type (same width for all)
    │   ├── Attach edge_index tensors per (src, rel, dst) type
    │   ├── Attach edge_attr tensors for featurized edge types
    │   ├── Release intermediate dicts, gc.collect()
    │   ├── Release edges_final_df from executor cache
    │   ├── Release node_id_df from executor cache
    │   └── Output: HeteroData ready for torch.save() and GNN training
    │
    └── Post-construction: Save outputs (build_graph.py)
        ├── torch.save() → BytesIO → fs_utils.write_bytes() → work dir (.pt file)
        │   (local path → open(); s3a:// URI → Hadoop FileSystem)
        ├── MetadataCollector.to_metadata_files() → six JSON dicts
        └── write_metadata_to_local() → fs_utils.write_bytes() → six JSON files
            (same scheme routing; write_metadata_to_s3() adds the boto3
             mirror when an S3 archive is configured)
```

## Metadata Files

Every PyG build produces six JSON metadata files written alongside the `.pt` file (locally, and mirrored to S3 when an archive is configured). These files enable downstream training and inference code to consistently use the `HeteroData` object without re-running the pipeline.

### Output Location

Written under `--local_work_dir` (and mirrored under the S3 archive bucket/key when `--s3_archive_bucket` is set).

The period is written as Hive-style partition directories (`year=2024/month=12`) derived from `--time_period`. Everything a build produces stays under that one directory, so a period can be copied, archived or deleted as a unit:

```
<local_work_dir>/pyg/year=2024/month=12/
├── hetero_data.pt
├── metadata/
│   ├── graph_schema.json
│   ├── feature_spec.json
│   ├── normalization.json
│   ├── encoding_config.json
│   ├── ontology_schema.json
│   └── slot_mapping.json
└── node_index/
    └── part-*.parquet
```

Spark's partition discovery reads `key=value` directory names into real columns, so the tabular artifacts read as partitioned tables across every period they hold — a filter on `year`/`month` becomes a PartitionFilter and unmatched periods are never opened:

```python
# enriched triples: the subtree is uniformly Parquet, so read it directly
spark.read.parquet(f"{work_dir}/enriched")
# columns: [subject, predicate, object, year, month]

# node_index: pass basePath, since the period directory also holds the
# .pt blob and the JSON metadata
spark.read.option("basePath", f"{work_dir}/pyg") \
     .parquet(f"{work_dir}/pyg/year=*/month=*/node_index")
# columns: [node_type, node_id, uri, year, month]
```

A `--time_period` that is not `YYYY-MM` is written as a single path segment instead, unpartitioned. There is no `day=` level: `--time_period` is monthly, so it would carry one value per month — path depth with no pruning benefit.

For experiment variants (non-default `--pyg_filename`), the metadata and node-index directories are named after the output file stem, so two variants in one period cannot overwrite each other:

```
<local_work_dir>/pyg/year=2024/month=12/
├── hetero_data_512d.pt
├── hetero_data_512d_metadata/
│   ├── graph_schema.json
│   └── ...
└── hetero_data_512d_node_index/
    └── part-*.parquet
```

The metadata and node-index directories are derived automatically from the `.pt` filename (`--pyg_filename`, and the S3 `--s3_pyg_key` when archiving) by `derive_metadata_prefix()` / `derive_node_index_prefix()` in `metadata_writer.py`. No additional configuration is required.

### File Descriptions

#### `graph_schema.json`

Complete inventory of every node type and edge type in the graph. The entry point for any consumer of the graph.

**Schema version: `1.2`.** 1.2 adds `relation_groups` and the `index` / `relation_group` / `src_type_index` / `dst_type_index` fields — additive, every 1.1 field keeps its name and meaning. Node-type `has_features` changed meaning in 1.1 — see below. Check the `version` field before relying on it.

**Contents:**
- Every node type with its count, source ontology URI, category tag, and `has_features`
  - `has_features` means the type carries **literal-value features**: a non-zero `literal_values` segment (the last of the three node-vector segments). On the e2e fixtures 76 of 100 node types qualify; the other 24 are pure taxonomy types (`EconomicSector`, `GeographicRegion`, `TimePeriod`, …) that carry ontology structure but no measurements.
  - It cannot usefully mean "a feature tensor exists" — `constructor.py` gives *every* node type an `x`, falling back to a zeros placeholder — nor "any non-zero value", since the `ontology_structure` segment is populated for every typed node. Literal values are the segment that actually varies between types.
  - **Through schema `1.0` this field was `count > 0`** — the node count, not features at all. It was therefore true for essentially every type, and `summary.node_types_with_literal_features` was identical to `total_node_types` by construction (a real build reported 100 and 100). A consumer filtering node types on `has_features` against a 1.0 artifact gets every type back.
  - Caveat: a type whose literal values all normalize to exactly 0.0 (a constant numeric property under z-score) reads as `false`. That is the honest answer to "does this type carry usable literal signal", but it is not the same question as "were literals present in the source".
- Every edge type as a full three-part tuple with its count, predicate URI, origin, whether it has edge features, and if so the feature dimension
  - Edge-type `has_features` was always correct — it comes from `edge_feature_flags`, which reflects whether `edge_attr` was actually produced (23 of 499 on the same build).
  - `origin` is one of **`raw`** (the relationship was stated in the source RDF), **`enrichment`** (this pipeline inferred it), or **`unification`** (a cross-source identity link). Enrichment adds ~91,000 triples on top of the raw data, so some edges are reported facts and others are derived — a distinction a model consumer should weight differently. Classified by `classify_edge_origin()` in `rdf_utils.py` from the predicate namespace **and both endpoint node types**, since the pipeline marks its output in two places: inferred links carry a minted *predicate* (under `jefflevesque.com/ontology/bls/`, …), while unification links carry a minted *node* but a standard predicate (`unified:November owl:sameAs cpi:November`). Checking the predicate alone reports every unification edge as `raw`.
  - Deliberately three values rather than separating intra- from cross-source enrichment: those share a namespace, so nothing at this layer can tell them apart, and a field promising a distinction it never emits is worse than a narrower honest one. It records *that* an edge was derived, not *why* — per-edge lineage is a much larger change, worth building only once a model's predictions need explaining.
- **`relation_groups`** (1.2): which edge types are the same relation and may **share GNN weights**. A PyG edge type is `(src_type, relation, dst_type)` and a heterogeneous conv allocates one weight matrix per edge type, so a relation spanning many endpoint pairs multiplies out — on the e2e fixtures **69 relations produce 770 edge types**, and a single `HeteroConv` 1024→128 over that is ~101M parameters for ~10k edges, with 67 edge types holding exactly one edge.
  - The `.pt` cannot collapse them: `edge_index` values are node IDs **local to their node type**, so `(cpi_Category, r, X)` and `(eci_Industry, r, X)` sharing a key would put two ID spaces in the same row. The multiplicity is forced by the container, so the pipeline publishes the grouping instead of leaving a consumer to guess it by string-splitting edge-type keys.
  - Each group carries `edge_types` (the keys it covers), `edge_type_count`, summed `count`, `predicate_uri`, `origin`, `has_features` and `feature_dim`. The grouping **partitions** the edge types — every key is in exactly one group.
  - Nothing is lost by tying: each edge type also carries `src_type_index` / `dst_type_index` into the node-type table (whose entries now carry a stable `index`, assigned by sorted name), so a shared relation weight can still condition on endpoint type through a node-type embedding — one table of *N* types rather than *N×M* matrices.
- Summary statistics: total node types, total edge types, total relation groups, total nodes, total edges, edge types with features
- Build metadata: time period, build timestamp, pipeline config

**Generated by:** `constructor.py` after HeteroData assembly, from `node_mapper.node_counts`, `node_mapper.get_type_uri_mapping()`, `edge_mapper.build_edge_indices()`, and `edge_feature_extractor.get_edge_classification()`. The per-type `has_features` flag is read off the assembled feature tensors themselves (`MetadataCollector.register_node_literal_features`), so it cannot drift from the `.pt` it describes.

**Changes between builds:** Yes — counts change every time period; new types may appear when new data sources are added.

---

#### `feature_spec.json`

Defines the structure of the 1024-d node feature vector and the 32-d edge feature vectors. Tells training code what each segment means and how to route dimensions through the model architecture.

**Contents:**
- Total node feature dimension with all segment and sub-segment boundaries (start index, end index, dim, name, type)
- Flag indicating structural dimensions are shared within a node type
- Total edge feature dimension with all segment and sub-segment boundaries
- List of edge types that carry features and list that do not
- Per-relation derivation method (temporal, option_stock, escalation, correlation)

**Generated by:** `constructor.py` from `feature_extractor.get_layout().to_dict()` and `edge_feature_extractor.get_layout().to_dict()`

**Changes between builds:** Rarely — only when the feature vector design changes (new segment layout, different dimensions).

---

#### `normalization.json`

Per-property normalization statistics used to z-score numeric literal values during feature encoding. Required to encode new data into the same feature space the model was trained on.

**Contents:**
- Normalization method (z-score)
- Per-property statistics: predicate URI, mean, standard deviation, count of non-null values
- List of zero-variance properties (sigma was 0, set to constant 1.0)

**Precision:** Statistics are rounded to 12 significant digits on write. `stddev` is a parallel reduction, and parallel float reductions are not order-deterministic — two runs over identical data could otherwise differ in the last ULP of a float64, making the file non-reproducible byte-for-byte. The rounding is far below float32 feature precision, so feature tensors are unaffected (they compared equal even when this file did not).

**Generated by:** `feature_extractor._collect_normalization_metadata()` during the stats aggregation pass — a single-pass `groupBy().agg()` on the numeric literals DataFrame, collecting one row per predicate (typically <200 rows)

**Changes between builds:** Yes — distribution statistics shift every time period as new data arrives. A model trained on December 2024 normalization stats expects inference data normalized with those same stats.

---

#### `encoding_config.json`

Every parameter needed to deterministically reproduce the hash-based encoding. If any of these values change, the same ontology class or property hashes to different vector positions and the trained model breaks.

**Contents:**
- Hash algorithm name (`spark_murmur3`)
- Per-segment encoding parameters: dimension, number of hash functions, seed values
- Class identity seeds, class hierarchy seeds and decay function, ontology membership method
- Property presence seeds and encoding convention (1.0 present, -1.0 absent, 0.0 not in schema)
- Domain/range seeds, numeric value hashing seed, categorical value hashing seeds
- Edge feature encoding parameters: relation classification fragments, temporal normalization divisor, ratio clamp value, cross-property derivation seeds
- Total node and edge feature dimensions (`node_features.total_dim`, `edge_features.total_dim`)
- `checksum` — a **SHA-256 digest of the encoding contract**: everything above, hashed together. This is a *contract* hash, not a data hash, so rebuilding a different time period with the same settings yields the same digest, while changing any seed, dimension, segment boundary or namespace table changes it. A deployed model can compare the digest it was trained against with the one shipped alongside a graph and refuse to run on a mismatch — otherwise it would load cleanly and return plausible, silently wrong numbers, with every feature in a different slot than the weights expect.

> **Note:** this field previously held only `{"total_node_feature_dim": N}` — a dimension, not a checksum, which detected nothing (two builds with different seeds but the same vector width compared equal). It was also lost to a key collision when the node and edge configs were merged. The digest is now computed once over the merged config, so both halves contribute.

**Generated by:** `constructor.py` by merging `feature_extractor.get_encoding_config()` and `edge_feature_extractor.get_encoding_config()`; the digest is stamped by `MetadataCollector`, the only place that sees the whole contract

**Changes between builds:** Rarely — only when the encoding scheme is redesigned. Should be identical across all time periods that feed the same model.

---

#### `ontology_schema.json`

Frozen snapshot of the ontology structure at build time. Contains the class hierarchies, property definitions, domain/range declarations, and namespace mappings used to compute the structural and schema segments of the feature vector.

**Contents:**
- Per node type (keyed by PyG name): source type URI, ordered superclass chain with depths, namespace, defined properties with their range types
- URI-to-PyG-name mapping for all type URIs encountered
- Namespace prefix table
- `ontology_mapping_enabled` / `ontology_mapping_evidence` — whether the ontology-mapping phase ran over the triples this build read, and what that verdict was based on
- `hierarchy_source` / `property_schema_source` / `property_hierarchy_source` — where each axiom set came from, or why there is none
- `provenance` — per axiom set, whether it was **declared by the source**, **curated**, or **observed**
- `property_schema_coverage` — how many predicates got a domain and a range, and which ones did not
- `derived_axioms` — per axiom set: how many axioms each derivation route produced (including `declared`), and which axioms those were

**Provenance: derived is not declared.** No source in this project declares `rdfs:subClassOf`, `rdfs:domain`, `rdfs:range` or `rdfs:subPropertyOf` — verified across the real 130k-triple run and both fixture sets. All four are therefore *derived* by `OntologyMapper`, and once in the graph a derived axiom is shaped exactly like a declared one. They do not license the same reasoning: "this property is used on class C in this month's data" is weaker than "this property's domain is C", and a property seen with one class here could be broader in general.

| Axiom set | Derivation routes |
|---|---|
| `class_hierarchy` | `curated class mappings` + `class naming` |
| `property_hierarchy` | `curated property mappings` (shared `PROPERTY_MAPPINGS` targets) |
| `property_domain` | `observed subject types` |
| `property_range` | `observed object types` + `declared literal datatype` |

**The `*_source` fields name the route, not the predicate.** `hierarchy_source` used to read `"rdfs:subClassOf"` — true of the predicate the encoder consumed, and false as an answer to "did a source declare this hierarchy". It now reads like `derived: curated class mappings (24), class naming (10)`, or `mixed: …` when a source genuinely declares some. Only a set with no derivation markers at all still reports the bare predicate name. Empty sets keep their existing "no rdfs:subClassOf in source data…" reason — that is about *absent data*, which provenance does not change.

**Counting axioms in an enriched graph does not answer "does any source declare them."** Since the mapper emits them, an enriched build contains 34 `rdfs:subClassOf` triples and zero of them came from a source. `derived_axioms.<set>.counts.declared` is the field that answers it, and it is `0` on every build since the derivations landed.

**Curated and guessed edges are separated.** The two are not equally trustworthy: a person chose each curated edge, while the naming rules produced `AllItemsLessShelter -> Shelter` and three more inversions before the negating-qualifier guard caught them. `derived_axioms.class_hierarchy.axioms` lists the edges of each route, and every entry in a node type's `superclass_chain` carries its own `provenance` — so chasing a wrong superclass tells you whether to edit `CLASS_MAPPINGS` or fix a naming rule, without reading the code. Links at `depth > 1` report `"transitive closure of the direct edges"`, since they are this pipeline's closure rather than asserted edges with a route of their own.

**Why some predicates get no domain or range.** `rdfs:domain` is an axiom with *intersection* semantics: asserting `p rdfs:domain A` and `p rdfs:domain B` says every subject of `p` is both an A and a B. Measured over the fixtures, 77 of 160 predicates are used on more than one class (`market:askPrice` on `EquitySnapshot` and `OptionSnapshot`; `rdfs:label` on 68), so emitting both would state something false. A predicate with more than one candidate therefore gets **no axiom**, and `property_schema_coverage` names it. Range applies the same rule across both routes at once, which also rules out the two predicates used with entities *and* literals (`cpi:hasMonth` is 189 typed URIs and 11 literals).

Skipping costs less than the count suggests: a predicate used on one class carries real signal about that class, while one used on 22 says little that `class_identity` does not already encode about the node itself.

**Measured coverage** over the full committed fixture set (all four sources, both loaders — 169 data predicates after enrichment):

| | Covered | Gap | Why the gap |
|---|---|---|---|
| `rdfs:domain` | 83 (49.1%) | 86 | 77 predicates are used on more than one class; the rest carry no typed subject |
| `rdfs:range` | 135 (79.9%) | 34 | 16 ambiguous (several object classes, or entity *and* literal use); the remainder are CAP predicates whose URI objects are never typed in these fixtures |

Every one of those gaps is listed by URI in `property_schema_coverage.without_domain` / `without_range`, so it is a known set rather than something to be inferred from a thin sub-segment. Resulting occupancy, against 0.00% for all three before this: `class_hierarchy` 1.25%, `domain_range` 2.83% (carried by 77 of 102 node types), `property_hierarchy` 2.31%.

**Reading an empty hierarchy.** When every `superclass_chain` is `[]`, `ontology_mapping_enabled` tells you which of the two causes you are looking at, because they demand opposite responses:

| `ontology_mapping_enabled` | Meaning | What to do |
|---|---|---|
| `true` | The mapping phase ran and the sources genuinely declare no subsumption | Nothing — that is the data |
| `false` | The hierarchy was never computed; `class_hierarchy` (64 of 1024 dims, 6.25% of every node vector) is structurally zero | Re-run enrichment with `--enable_ontology_mapping true` |

The flag arrived in schema `version` `1.1`, `provenance` / `property_schema_coverage` in `1.2`, and the route-aware `*_source` strings plus `derived_axioms` in `1.3`; a `1.0` file predates all of them and its empty hierarchy stays ambiguous. It is detected from the triples — the presence of `owl:equivalentProperty` / `owl:equivalentClass`, which only `OntologyMapper` emits — rather than read off a config flag. That is the only signal that stays truthful in `pyg_only` mode, where enrichment ran in a separate job and this job's own `--enable_ontology_mapping` describes a phase it never reaches. For the same reason a `pyg_only` job manifest records `"enable_ontology_mapping": null` rather than the inert flag — that field says nothing about the enriched Parquet the run consumed, and this file is what answers the question for it.

**Generated by:** `feature_extractor._collect_ontology_schema_metadata()` — collects from small distinct/aggregated DataFrames: type URIs (~500 rows), class hierarchy transitive closure (~5000 rows), property schema (~500 rows). All collect calls target aggregated DataFrames, never raw triples.

**Changes between builds:** Sometimes — when new data sources or ontologies are added. If the ontology structure changes between the training build and an inference build, the structural segment of the feature vector will differ. This file lets you detect that.

---

#### `slot_mapping.json`

Maps specific vector dimensions back to their semantic meaning. Purely for interpretability — no training or inference code depends on this file.

**Contents:**
- Per numeric property: predicate URI, local name, hash slot within the numeric sub-segment, global dimension index
- Per categorical property: predicate URI, local name, hash slots (multiple due to multi-hot), global dimension indices
- Per class: class URI, PyG name, hash slots in the class identity sub-segment, global dimension indices
- Per superclass: which slots in the hierarchy sub-segment each superclass contributes to
- Per namespace: which slot in the ontology source sub-segment each namespace occupies
- Hash collision report: collision counts and rates per sub-segment. For `class_identity` (slot mapping **1.1**) this reports *separability* rather than slot occupancy — see below.

> **`class_identity` in the collision report (slot mapping 1.1).** Class identity is a **multi-hot** code: each class occupies `num_hashes` slots and is identified by the *set*, not by any single slot. Slot reuse is therefore not identity loss. Through slot mapping 1.0 this sub-segment reported `collisions` / `collision_rate` computed as raw slot occupancy, which reads as lost identity and is misleading: with more hash entries than dimensions, pigeonhole forces a high value however healthy the code is. A real build with 44 classes × 4 hashes into 64 dims reported `collision_rate: 0.67` while every one of the 44 codes was distinct, the code matrix was full rank, and its condition number was ~12 — identity fully recoverable.
>
> Those keys are now `slot_reuse` / `slot_reuse_rate`, and the fields that do bound identity are reported alongside: `distinct_codes`, `classes_sharing_a_code` (classes that are genuinely indistinguishable), `max_pairwise_slot_overlap`, `capacity_classes`, `headroom_classes`, `code_matrix_rank`, `rank_deficiency`, and `linearly_separable`.
>
> **`linearly_separable` is measured, not inferred.** It was previously computed as `num_classes <= dim and not classes_sharing_a_code` — two conditions that are each *necessary* but not together *sufficient*, reported under the name of a rank test. Distinct codes inside the ceiling can still be linearly dependent: the 4-hot codes `{0,1,2,3}`, `{0,1,4,5}`, `{2,3,6,7}`, `{4,5,6,7}` are pairwise distinct and fit in 8 dims, yet A + D − B − C = 0, so one class is exactly a blend of the others and no linear readout can recover it. That build reported `linearly_separable: true` and shipped. The field is now the measured rank of the `num_classes × dim` code matrix (`code_matrix_rank == total_classes`), with `rank_deficiency` giving how many classes are unrecoverable.
>
> This also means `capacity_classes` is a **ceiling, not a guarantee** — `d` 4-hot codes drawn into `d` dims measure rank `d−2`, so plan headroom rather than aiming at the limit.
>
> A build that is not separable — over-subscribed, sharing a code, or rank-deficient — now **raises `ClassIdentityCapacityError`** instead of logging. A warning was the wrong severity: the artifact ships, every consistency check passes, and the only symptom is a model that never learns to tell two classes apart, weeks downstream with nothing pointing back at the build. `feature_config.allow_class_identity_oversubscription=true` restores warn-and-continue. Approaching the limit (past 85% of the segment width) still only warns.

**Generated by:** `feature_extractor._collect_slot_mapping_metadata()` — computes approximate slot assignments on the driver using a Python hash approximation of Spark's murmur3. The approximation may not match exactly for all inputs; this file is for interpretability only and is never used by training or inference code.

**Changes between builds:** Only when the encoding config or ontology schema changes. If neither changes, the slot mapping is identical across time periods.

#### `node_index/` (Parquet)

The identity map: which real-world entity each row of the graph is. `hetero_data.pt` stores only `x` and `num_nodes` per node type, so without this the graph is **anonymous** — row 5 of `cpi_Index` is a specific CPI series and nothing else on disk records which one.

The six JSON files above are all schema-level: together they answer *"what does column 37 mean?"*. This is the one artifact that answers *"who is row 5?"*.

**Contents:** one row per node — `node_type`, `node_id`, `uri`. Sorted by `(node_type, node_id)` and coalesced to a single file so the artifact is content-stable run to run.

**Why Parquet, not a seventh JSON:** production volume is ~30-50M triples/month, so this can reach millions of rows. A single JSON would be hundreds of MB and would need parsing in full to resolve one entity; Parquet supports predicate pushdown and matches how the rest of the pipeline stores bulk data.

**Needed for:**
- **Training** — labels arrive keyed by entity; without this there is nothing to join them on, so a target tensor aligned to the graph cannot be built
- **Inference** — to look up a specific entity's row, and to attribute a prediction ("row 5 scores 0.93") back to something meaningful
- **Rebuilds** — node IDs come from `row_number()` over a `uri`-ordered window, so they are deterministic for a given input, but adding or removing one entity shifts every row below it. A model trained on one period cannot be applied to the next without each build's own map.

**Generated by:** `constructor._node_index()`, written by `save_node_index()` in `build_graph.py`. Written by Spark directly from the executors, so it lands in object storage when `--local_work_dir` is an `s3a://` URI (as on the cluster) — it is not part of the boto3 mirror, being a distributed dataset rather than a driver-side blob.

**Changes between builds:** Every build — it is per-period by nature.

---

### Consumer Matrix

| File | Training (model init) | Inference (encode new data) | Inference (model load) | Human exploration | Experiment tracking |
|---|---|---|---|---|---|
| `graph_schema.json` | **Yes** — architecture decisions, data splits | **Yes** — validate compatible types | No | **Yes** — first file to read | **Yes** — what each experiment contained |
| `feature_spec.json` | **Yes** — layer construction, conv routing | No | **Yes** — reconstruct same architecture | Occasionally | **Yes** — compare feature designs |
| `normalization.json` | No | **Yes** — scale new data identically | No | No | Occasionally — detect distribution drift |
| `node_index/` | **Yes** — join labels to rows | No | **Yes** — resolve entity ↔ row, attribute predictions | **Yes** — the only way to tell what a row is | No |
| `encoding_config.json` | No | **Yes** — hash to same slots | **Yes** — compare `checksum.contract_digest` and refuse a mismatch | No | **Yes** — the digest identifies the encoding contract exactly |
| `ontology_schema.json` | No | **Yes** — encode new nodes with correct hierarchy | No | Occasionally | **Yes** — detect schema drift |
| `slot_mapping.json` | No | No | No | **Yes** — interpret model attention | No |

---

### Build / Train / Inference Lifecycle

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

### Driver Memory Impact

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

### Memory Budget by Driver Size

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
    "edge_feature_config": {
        "enabled": true,
        "edge_vector_dim": 32,
        "chunk_edge_threshold": 1000000,
        "max_edge_types_per_batch": 8,
        "enabled_categories": ["temporal", "option_stock", "escalation"]
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
| `feature_config.vector_dim` | `1024` | Node feature vector dimension — all segments scale proportionally. Minimum 32. |
| `feature_config.chunk_node_threshold` | `500000` | Node count above which chunked collection is used |
| `--class_mappings` (CLI) | built-in table | JSON object (or path to a JSON file) of `{source_class_uri: target_class_uri}` merged over `ontology_mapper.CLASS_MAPPINGS`. A `null` target drops a built-in entry. Lets a new source declare its class semantics without editing the module. Where two or more sources share a target the relationship is published as `rdfs:subClassOf`, not `owl:equivalentClass`. |
| `feature_config.numeric_predicate_min_share` | `0.5` | Share of a property's literal values that must parse as numbers for the property to be treated as numeric. At or below it, every value is a category label. Lower it to admit sparsely numeric properties; raise it toward `1.0` to demand near-uniformly numeric ones. |
| `edge_feature_config.enabled` | `true` | Enable/disable edge feature extraction entirely |
| `edge_feature_config.edge_vector_dim` | `32` | Edge feature vector dimension — all segments scale proportionally. Minimum 9. |
| `edge_feature_config.chunk_edge_threshold` | `1000000` | Edge count above which chunked collection is used |
| `edge_feature_config.max_edge_types_per_batch` | `8` | Maximum edge types unioned into one batched collect. Caps the width of the batch's Catalyst plan, which the edge budget alone does not — many tiny types satisfy any edge budget while still producing a plan large enough to exhaust the driver during plan rendering. |
| `edge_feature_config.enabled_categories` | `["temporal", "option_stock", "escalation"]` | Which edge categories receive features. Options: `temporal`, `option_stock`, `escalation`, `correlation`, `causal`, `strategy` |
| `include_temporal_nodes` | `true` | Include Month/Year/Quarter node types |
| `include_sector_nodes` | `true` | Include EconomicSector node types |

When config is empty, sensible defaults are inferred from the data.

### Job Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--mode` | Yes | `full` | `full`, `enrichment_only`, or `pyg_only` |
| `--source_paths` | Modes 1,2 | — | Comma-separated source path(s)/URI(s): local directories or `s3a://...`. Each is loaded independently and the results are unioned into a single triples DataFrame before enrichment. A path naming the archive's `source=sec` partition must also name `feed=filings`: that is the only SEC feed carrying RDF, and the job rejects the other seven up front rather than failing later on a missing Turtle column |
| `--local_work_dir` | Yes | — | Working directory for the interim enriched Parquet and the final artifacts. Must be reachable by every worker — a shared mount (e.g. NFS) or a URI on shared storage (`s3a://...`); on a multi-node cluster a driver-local path won't do |
| `--s3_archive_bucket` | No | `""` | Optional S3 bucket to *additionally* mirror the final artifacts (`.pt` + metadata + manifest). When empty, artifacts are written only to `--local_work_dir` (which may itself be an `s3a://` path) |
| `--s3_pyg_key` | No | `pyg/{time_period}/{pyg_filename}` | Optional S3 key for the archived `.pt`; the metadata prefix is derived from it |
| `--pyg_filename` | No | `hetero_data.pt` | Local `.pt` filename (override for experiment variants, e.g. `hetero_data_512d.pt`); determines the metadata directory name |
| `--enable_ontology_mapping` | No | `true` | Run the ontology-mapping phase: equivalences, predicate folding, and the derived `rdfs:subClassOf` hierarchy that fills the `class_hierarchy` sub-segment. Applies to modes `full` and `enrichment_only`; `pyg_only` never reaches this phase, so the flag is inert there (and meaningless in that mode's manifest — see [`ontology_schema.json`](#ontology_schemajson)) |
| `--time_period` | No | Current `YYYY-MM` | Time period label for output paths |
| `--pyg_config` | No | `{}` | JSON string with PyG construction config |
| `--parquet_partitions` | No | `200` | Number of Parquet output partitions |
| `--source_format` | No | `ntriples` | Source RDF format: `ntriples` (one triple per line in `.nt` files) or `turtle_parquet` (self-contained Turtle blobs in a Parquet column). Applies to modes `full` and `enrichment_only` only — `pyg_only` always reads enriched Parquet written by this pipeline |
| `--turtle_column` | No | *(auto)* | Column name containing Turtle strings when `--source_format=turtle_parquet`. Ignored for `ntriples` format. Left unset, the column is resolved **per source** against `TURTLE_COLUMN_CANDIDATES` (`triples`, then `rdf_turtle`), so one run can span sources whose schemas disagree; set it to force a single name everywhere |
| `--market_sector_definitions_bucket` | No | `""` | S3 bucket containing the S&P 500 tickers CSV for dynamic sector classification. If empty, falls back to hardcoded sector patterns |
| `--market_sector_definitions_key` | No | `""` | S3 key for the tickers CSV (e.g., `market/sp500/tickers/latest.csv`). Tickers are grouped by `GICS Sector` column to build sector patterns at runtime |

Metadata files are always written when mode is `full` or `pyg_only`. Mode `enrichment_only` does not produce metadata files (no PyG graph is built in that mode).

Jobs are launched with `bin/submit_spark_job.sh`, which packages the code
and submits to the Spark standalone master with the RAPIDS Accelerator
enabled. Set `SPARK_MASTER_URL` (and optionally `RAPIDS_JAR`); see the
launcher header for all environment variables.

**Example — full pipeline (local source, local + S3 archive):**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode full \
    --source_paths /data/rdf/monthly/2024-12/ \
    --local_work_dir /data \
    --s3_archive_bucket my-archive \
    --s3_pyg_key pyg/year=2024/month=12/hetero_data.pt \
    --enable_ontology_mapping true \
    --time_period 2024-12 \
    --parquet_partitions 200 \
    --pyg_config '{"feature_config": {"normalize": true, "vector_dim": 1024}, "edge_feature_config": {"enabled": true, "edge_vector_dim": 32}}'
```

Outputs (local; and mirrored to `s3://my-archive/...` because an archive
bucket was given):
```
/data/enriched/year=2024/month=12/triples/            # interim, local only
/data/pyg/year=2024/month=12/hetero_data.pt
/data/pyg/year=2024/month=12/metadata/graph_schema.json
/data/pyg/year=2024/month=12/metadata/feature_spec.json
/data/pyg/year=2024/month=12/metadata/normalization.json
/data/pyg/year=2024/month=12/metadata/encoding_config.json
/data/pyg/year=2024/month=12/metadata/ontology_schema.json
/data/pyg/year=2024/month=12/metadata/slot_mapping.json
```

**Example — reduced-dimension experiment from existing enriched Parquet:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_512d.pt \
    --time_period 2024-12 \
    --pyg_config '{"feature_config": {"vector_dim": 512, "normalize": true}, "edge_feature_config": {"edge_vector_dim": 16}}'
```

Outputs:
```
/data/pyg/year=2024/month=12/hetero_data_512d.pt
/data/pyg/year=2024/month=12/hetero_data_512d_metadata/graph_schema.json
/data/pyg/year=2024/month=12/hetero_data_512d_metadata/feature_spec.json
... (remaining four files)
```

**Example — additional edge feature categories:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_full_edge_features.pt \
    --time_period 2024-12 \
    --pyg_config '{"edge_feature_config": {"enabled_categories": ["temporal", "option_stock", "escalation", "correlation", "causal"]}}'
```

**Example — edge features disabled:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode pyg_only \
    --local_work_dir /data \
    --pyg_filename hetero_data_no_edge_features.pt \
    --time_period 2024-12 \
    --pyg_config '{"edge_feature_config": {"enabled": false}}'
```

**Turtle Parquet source (SEC filings), reading from S3 via s3a://:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode enrichment_only \
    --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \
    --local_work_dir /data \
    --source_format turtle_parquet \
    --turtle_column triples \
    --time_period 2024-12 \
    --parquet_partitions 200
```

If your Parquet column is named something other than `triples`, nothing needs to be
said: with `--turtle_column` unset the loader resolves the name against
`TURTLE_COLUMN_CANDIDATES` for each source path independently, so
`triples` and `rdf_turtle` sources can be submitted together and enrich into a single
graph. Pass `--turtle_column rdf_turtle` only to force one name across every source —
useful when a schema carries both columns and only one of them is meant, and the source
of truth when a third name appears.

Source Parquet written by pandas/pyarrow commonly carries **nanosecond** timestamps,
which Spark's Parquet reader rejects outright (`Illegal Parquet type: INT64
(TIMESTAMP(NANOS,true))`) during schema conversion — before a single triple is read, and
regardless of the fact that this pipeline goes on to select nothing but the Turtle
column. [`bin/submit_spark_job.sh`](bin/submit_spark_job.sh) therefore sets
`spark.sql.legacy.parquet.nanosAsLong=true` by default; set `PARQUET_NANOS_AS_LONG=false`
to restore Spark's behavior.

**Full pipeline from a Turtle Parquet source:**

```bash
SPARK_MASTER_URL=spark://<host>:7077 \
  bin/submit_spark_job.sh \
    --mode full \
    --source_paths s3a://my-data-lake/raw/sec/filings/2024-12/ \
    --local_work_dir /data \
    --source_format turtle_parquet \
    --turtle_column triples \
    --time_period 2024-12 \
    --parquet_partitions 200 \
    --pyg_config '{"feature_config": {"normalize": true, "vector_dim": 1024}}'
```

### Cluster prerequisites for GPU runs

Four cluster-side settings decide whether this job runs at all — or whether it
merely *appears* to. They share an unpleasant property: when any of them is wrong,
the job **hangs indefinitely with no error message** (or, for the fourth, silently
runs on one node) rather than failing, so they are worth checking before you
conclude the job itself is slow.

**1. Workers must advertise their GPUs.** The RAPIDS configuration makes every
executor request a GPU (`spark.executor.resource.gpu.amount`). On a standalone
cluster the worker must independently declare that it *has* one, or the master
can never satisfy the request and the application waits forever without ever
scheduling a task:

```bash
# in $SPARK_HOME/conf/spark-env.sh on each worker
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 \
  -Dspark.worker.resource.gpu.discoveryScript=$SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
```

**2. Buckets whose names contain dots need path-style S3A access.** With S3A's
default virtual-host addressing the request is sent to
`<bucket>.s3.<region>.amazonaws.com`. If the bucket name contains dots (for
example a bucket named after a domain), that hostname has more labels than the
`*.s3.<region>.amazonaws.com` wildcard certificate covers, and TLS fails with
`SSLPeerUnverifiedException` before any S3 call is made:

```
spark.hadoop.fs.s3a.path.style.access    true
```

Note that a working `aws s3 ls` proves nothing here: `boto3` and the AWS CLI
switch to path-style automatically for dotted buckets, while S3A does not.

**3. Cap the RAPIDS pool on GPUs whose memory is shared with the host.** RAPIDS
defaults to pooling essentially all available GPU memory, which is correct for a
discrete card with dedicated VRAM. On integrated or unified-memory GPUs, that
memory is the host's RAM — the default can reserve nearly all of it, starve the
OS and the JVM, and drive the machine into swap:

```
spark.rapids.memory.gpu.allocFraction    0.25
```

**4. On a multi-homed node, bind Spark to the interface the cluster actually uses.**
If the nodes have more than one network — say a management LAN plus a dedicated
fabric — every Spark process advertises whichever non-loopback interface it finds
first unless told otherwise. Pointing the master at the fabric is not enough:
`SPARK_MASTER_HOST` only decides where the master *listens*, while the addresses
the driver, the executors and their block managers hand out to *each other* come
from `SPARK_LOCAL_IP`:

```bash
# in $SPARK_HOME/conf/spark-env.sh, per node — the node's OWN address on the fabric
export SPARK_LOCAL_IP=10.0.0.7
```

This one does not hang so much as **lie**. Executors on every node *except* the
driver's cannot reach the driver's advertised address, time out after 120s, exit 1,
and are relaunched forever; the executor that happens to be co-located with the
driver reaches it over loopback and quietly runs the entire job. The master reports
every worker healthy, the job succeeds, the tests pass — and the cluster is running
at the capacity of a single node. If those executors do survive long enough to
shuffle, peer block fetches hang instead, and the job stops making progress with no
error at all. Confirm the fix by checking that the master's worker list shows fabric
addresses, not LAN ones.

**Verify GPU execution; don't assume it.** A `count()` on Parquet can be answered
from file metadata without touching the GPU. Set `spark.rapids.sql.explain=ALL`,
or confirm that `Gpu*` operators (e.g. `GpuFileSourceScanExec`) appear in the
physical plan.

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
    │   ├── Transaction sequences (precedes by transaction date, within
    │   │   a reporting owner and instrument class)
    │   ├── Sector classification (belongsToSector)
    │   └── Violation type linking (hasViolationType)
    │
    ├── Market Intra-Source Enricher
    │   ├── Snapshot temporal sequences (precedes by captureTime)
    │   ├── Option-to-underlying equity linking (hasUnderlyingEquity)
    │   ├── Option strategy detection (straddleWith, spreadWith, strangleWith)
    │   ├── Sector classification (belongsToSector via symbol; loads
    │   │   GICS sectors from S3 tickers CSV with hardcoded fallback)
    │   └── Moneyness computation (hasMoneyness: ATM/ITM/OTM)
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
    └── Ontology Mapper (optional; --enable_ontology_mapping, default true)
        ├── owl:equivalentProperty / owl:equivalentClass  (one-to-one pairs only)
        ├── predicate folding to the unified vocabulary
        ├── skos:prefLabel normalization
        ├── rdfs:subClassOf      ← curated CLASS_MAPPINGS + class naming
        ├── rdfs:subPropertyOf   ← curated PROPERTY_MAPPINGS shared targets
        ├── rdfs:domain/range    ← observed usage + declared XSD datatypes
        └── prov:derivedBy       ← how each of the above was arrived at
    │
    ▼
triples_df (enriched) → Parquet (local) + PyG HeteroData (.pt) + Metadata JSON (local + optional S3)
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
- Links equity and option snapshots in chronological sequences per symbol
- Links option snapshots to their underlying equity snapshots
- Identifies option strategies (straddles, vertical spreads, strangles)
- Classifies snapshots by sector (via symbol and underlyingSymbol)
- Computes moneyness classification (ATM/ITM/OTM) for option snapshots

**Within NOAA Weather Data** (1 mapper)
- Links alerts in chronological sequences per geographic area
- Connects alerts affecting same regions via SAME geocodes
- Links alerts of the same event type
- Detects severity escalations within same area over time

### Enrichment as PySpark Operations

Each enrichment step follows the same pattern — filter the triples DataFrame to extract relevant entities, join to discover relationships, and produce new triples:

```python
def link_options_to_underlying(self, triples_df):
    """Example: Link option snapshots to underlying equity snapshots"""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Extract option snapshots with their underlying symbol and capture time
    options_df = (
        triples_df.filter(F.col("predicate") == RDF_TYPE)
                  .filter(F.col("object") == OPTION_SNAPSHOT_TYPE)
                  .select(F.col("subject").alias("option"))
        .join(
            triples_df.filter(F.col("predicate") == UNDERLYING_SYMBOL_PRED)
                      .select(F.col("subject").alias("option"),
                              F.col("object").alias("underlying_symbol")),
            "option"
        )
        .join(
            triples_df.filter(F.col("predicate") == CAPTURE_TIME_PRED)
                      .select(F.col("subject").alias("option"),
                              F.col("object").alias("option_time")),
            "option"
        )
    )

    # Extract equity snapshots with their symbol and capture time
    equities_df = (
        triples_df.filter(F.col("predicate") == RDF_TYPE)
                  .filter(F.col("object") == EQUITY_SNAPSHOT_TYPE)
                  .select(F.col("subject").alias("equity"))
        .join(...)  # similar pattern for symbol + capture_time
    )

    # Join: option's underlyingSymbol = equity's symbol AND same captureTime
    joined = options_df.join(equities_df,
        (options_df.underlying_symbol == equities_df.equity_symbol)
        & (options_df.option_time == equities_df.equity_time),
        "inner"
    )

    # Produce new triples
    return joined.select(
        F.col("option").alias("subject"),
        F.lit(HAS_UNDERLYING_EQUITY_PRED).alias("predicate"),
        F.col("equity").alias("object")
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

> **Note:** temporal alignment is **not** part of cross-source linking. It runs earlier, as its own enrichment phase (`TemporalUnifier`), and its output is merged before this stage runs. Cross-source linking previously duplicated it with a matcher anchored only at the end of the URI, so anything merely *ending* with a month name (e.g. `...PercentChange_..._2025_September`) was asserted to **be** that month via `owl:sameAs`. That step has been removed; the example below is what `TemporalUnifier` produces correctly.

**Temporal Alignment** (enrichment phase 2, `TemporalUnifier`) — unifies temporal entities across all sources
```turtle
# Before: each source has its own temporal entities
cpi:November, ppi:November, jolts:November, sec:November, market:November

# After: single unified temporal entity
unified:November2024 a bls:UnifiedMonth ;
    owl:sameAs cpi:November, ppi:November, jolts:November,
               sec:November, market:November, noaa:November .

# ...and the SOURCE-side periods are given a type of their own, which is what
# makes them nodes at all (see below)
cpi:November a temporal:SourceMonth ; rdfs:label "November" .
```

> **Source temporal URIs are typed here too.** Sources reference periods as bare URIs — `cpi:February`, `eci:2024`, `jolts:August` — carrying no `rdf:type`. `node_mapper` only creates nodes for typed URIs, so those periods were not nodes and *every* triple pointing at them was dropped during edge resolution: `hasMonth`, `hasYear`, `hasStartMonth`/`hasEndMonth`, `hasStartYear`/`hasEndYear` (~1,205 on the e2e fixtures). The graph had no temporal dimension — nothing recorded *when* a measurement happened — and the `owl:sameAs` links above, pointing at the same untyped URIs, were dropped as well, leaving `UnifiedMonth`/`UnifiedYear` as isolated nodes. `TemporalUnifier` now emits `temporal:Source{Month,Year,Quarter}` for exactly the set of temporal URIs it already collects, so both hops of the bridge resolve:
>
> ```
> cpi measurement → cpi:February → unified:February ← eci:February ← eci measurement
> ```
>
> The type is deliberately **not** in an `*/enrichment/` namespace: `classify_edge_origin()` reads a minted endpoint type as a pipeline-derived edge, and a measurement's link to its own period is an observed source fact — only the type is ours. It is deliberately distinct from `UnifiedMonth` as well, so `unified:February owl:sameAs cpi:February` still says which node is canonical.
>
> **These types are pinned as canonical.** Many source periods already carry a source type — `cpi:2024` is both `cpi:Year` and `temporal:SourceYear`. `node_mapper`'s default rule (fewest instances wins) picks the *source* type, because it is per-namespace and therefore rarer, which shards one concept across every namespace that names it: measured on the e2e fixtures, 37 months split over `cpi_Month`/`jolts_Month`/`empsit_Month`/`eci_Month`/`temporal_SourceMonth` and 14 years likewise, leaving `temporal_SourceYear` holding a single node. The `owl:sameAs` edges then land on whichever shard a period fell into, and a heterogeneous GNN sees unrelated node types with no path between them. `node_mapper._CANONICAL_TYPE_PRIORITY` pins `temporal_Source*` ahead of the count heuristic so every period lands in one node type per granularity. The source type is not lost — it remains an `rdf:type` triple and appears in `ontology_schema.json`; only the canonical type used for graph *structure* is overridden. Predicates stay per-source (`cpi_hasYear`, `jolts_hasYear`, …), so the sources agree on what a year *is* without being forced to share measurement semantics.

> **Cross-source paths are four hops, so size the model accordingly.** Every
> route between sources goes through a hub rather than a direct edge, and the
> temporal spine is the longest of them:
>
> ```
> SEC filing → temporal/sec/July → unified:July → cpi:July → cpi measurement
>      1              2                 3            4
> ```
>
> The company hub is shorter but the same shape
> (`filing → issuer → unified:Company_X ← quote snapshot`). This is the design —
> hub-and-spoke is what lets *N* sources agree on a period without *N²* joins —
> but it has a direct consequence for training: **a message-passing depth of
> fewer than 4 layers cannot propagate any signal between two sources.** A
> 2-layer model trained on this graph learns within-source structure only, no
> matter how many cross-source edges the enrichment produced. Measured on the
> e2e fixtures, all six source-family pairs (bls/sec/market/noaa) are connected,
> and every one of them at distance 4.

**Linking Strategies** (applied across 100+ ontologies):

1. **Sector-Based Linking** — Links entities sharing economic sectors
```turtle
unified:EnergySector a bls:EconomicSector .

cpi:EnergyEntity bls:belongsToSector unified:EnergySector .
ppi:EnergyGoodsEntity bls:belongsToSector unified:EnergySector .
market:XOM_Ticker bls:belongsToSector unified:EnergySector .
sec:EnergyCompanyFiling bls:belongsToSector unified:EnergySector .
```

2. **Company/Ticker-Based Linking** — Links entities referencing same companies
```turtle
unified:Company_AAPL a bls:UnifiedCompany ;
    bls:ticker "AAPL" .

sec:AAPL_10K_Filing bls:refersToCompany unified:Company_AAPL .
market:AAPL_20241115T143000Z bls:refersToCompany unified:Company_AAPL .
```

3. **Geographic/Regional Linking** — Links entities by geographic region
```turtle
unified:CaliforniaRegion a bls:GeographicRegion .

laus:California_LaborForce bls:hasRegion unified:CaliforniaRegion .
noaa:California_HeatAlert bls:affectsRegion unified:CaliforniaRegion .
```

4. **Causal/Impact Relationships** — Discovers potential causal links
```turtle
ppi:EnergyGoods bls:leadsTo cpi:EnergyConsumer .
noaa:HurricaneAlert bls:impacts market:EnergyTicker .
sec:Form10K_Filing bls:affects market:StockTicker .
```

5. **Measurement Type Alignment** — Links similar measurement types
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

This enriched structure combined with ontology-aware node feature vectors, derived edge feature vectors, and the six metadata files enables GNNs to learn:
- **Temporal Patterns**: How indicators evolve and correlate over time across 100+ sources, with edge features encoding the exact time gap and direction
- **Cross-Domain Relationships**: How economic, financial, employment, and environmental factors interact, with edge features distinguishing intra-source from cross-source correlations
- **Sector Dynamics**: How sector-wide shocks propagate across different data types
- **Lead-Lag Relationships**: Which indicators predict changes in others, with temporal edge features encoding the lag magnitude
- **Geographic Effects**: How regional factors affect economic and market outcomes
- **Company-Specific Patterns**: How company fundamentals relate to market performance
- **Intraday Dynamics**: How market prices and options evolve within trading sessions, with edge features encoding moneyness and time-to-expiry signals
- **Ontology-Aware Similarity**: Nodes sharing superclasses or property schemas are naturally similar in feature space, even before training
- **Cross-Type Reasoning**: Universal node feature width enables shared GNN layers that learn patterns across all 100+ ontologies simultaneously
- **Edge-Modulated Message Passing**: Edge features allow the GNN to modulate messages based on per-instance signals (time gap, moneyness, severity delta) rather than treating all edges of the same type identically
- **Severity Escalation Detection**: Edge features on escalation edges encode the severity delta, enabling the GNN to learn escalation patterns in weather alert sequences
- **Consistent Inference**: The six metadata files ensure that new data is encoded into the same feature space the model was trained on — same normalization stats, same hash seeds, same ontology structure

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

**Market Data** (1 mapper, intraday equity + option snapshots every 20 minutes)
- Flat snapshot model: EquitySnapshot and OptionSnapshot with all fields as direct properties
- ~500+ tickers with full options chains (~500K+ symbols per snapshot)
- ~39 snapshots/day at 20-min intervals during market hours
- ~1-1.5M triples/day, ~30-35M triples/month

**NOAA Weather Data** (1 mapper)
- US weather alerts (CAP format)

> **Total: 100+ mappers and ontologies** covering economic, financial, employment, and environmental data

> **Typical monthly volume: ~30-50M triples** (dominated by intraday market snapshots)

> **Note:** Raw RDF data is generated by separate Lambda scraper functions (not part of this repository). This pipeline assumes RDF data is already available in S3 in N-Triples format conforming to 100+ domain-specific ontologies.

## Project Structure

```
pyg-knowledge-graph-builder/
├── bin/
│   └── submit_spark_job.sh                 # spark-submit launcher (RAPIDS/GPU)
├── conf/
│   └── spark-rapids.conf.template          # reference RAPIDS spark-defaults
├── spark_jobs/
│   ├── build_graph.py                      # Main Spark job entry point
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
│   │       ├── market/                     # Market-specific components (flat snapshot model)
│   │       │   ├── __init__.py
│   │       │   ├── patterns.py             # MARKET_SECTOR_PATTERNS, MARKET_OPTION_STRATEGY_PATTERNS
│   │       │   ├── correlations.py         # Market KNOWN_CORRELATIONS
│   │       │   └── measurements.py         # Market MEASUREMENT_TYPES
│   │       └── noaa/                       # NOAA-specific components
│   │           ├── __init__.py
│   │           └── patterns.py             # NOAA alert patterns
│   ├── pyg_builder/                        # PyG construction modules
│   │   ├── __init__.py
│   │   ├── constructor.py                  # Orchestrates HeteroData construction
│   │   │                                   # (5 steps) and MetadataCollector;
│   │   │                                   # returns (HeteroData, MetadataCollector)
│   │   ├── node_mapper.py                  # Assigns per-type integer node IDs on
│   │   │                                   # executors; get_type_uri_mapping() for
│   │   │                                   # metadata
│   │   ├── edge_mapper.py                  # Resolves edges to integer index tensors
│   │   │                                   # on executors; returns cached resolved
│   │   │                                   # edges for edge feature reuse;
│   │   │                                   # get_predicate_uri_mapping() for metadata
│   │   ├── feature_extractor.py            # Ontology-aware node feature vectors with
│   │   │                                   # VectorLayout; collects normalization stats,
│   │   │                                   # ontology schema, and slot mapping for
│   │   │                                   # metadata during build_features()
│   │   ├── edge_feature_extractor.py       # Derived edge feature vectors with
│   │   │                                   # EdgeVectorLayout; reuses cached resolved
│   │   │                                   # edges from EdgeMapper; provides encoding
│   │   │                                   # config and edge classification for metadata
│   │   └── metadata_writer.py              # MetadataCollector (accumulates artifacts
│   │                                       # during construction steps);
│   │                                       # write_metadata_to_local() / _to_s3()
│   │                                       # (serialize the six JSON files);
│   │                                       # derive_metadata_prefix() (naming convention)
│   └── utils/
│       ├── __init__.py
│       └── rdf_utils.py                    # Namespace constants, URI helpers, canonical
│                                           # namespace registry (NAMESPACE_PREFIXES,
│                                           # ONTOLOGY_NAMESPACE_INDICES)
├── tests/                                  # Unit and integration tests
├── .gitignore
├── README.md
└── requirements.txt
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
| `build_graph.py` | Parses source RDF into triples DataFrame (`load_ntriples_to_dataframe()` for `.nt` files, `load_turtle_parquet_to_dataframe()` for Turtle Parquet blobs); dispatches via `load_source_triples()`; orchestrates pipeline modes; writes enriched Parquet locally and the `.pt` + metadata JSON files locally (mirroring the final artifacts to S3 when an archive is configured). `--source_format` and `--turtle_column` parameters control which loader is used | Yes (orchestration) |
| `constructor.py` | Orchestrates PyG HeteroData construction from triples DataFrame (5 steps: node IDs, edge indices, node features, edge features, assembly); initializes `MetadataCollector`; calls `register_*` methods after each step; returns `(HeteroData, MetadataCollector)` | Yes (orchestration) |
| `node_mapper.py` | Discovers node types, assigns per-type integer IDs via Window functions. `get_type_uri_mapping()` provides a small collect for metadata. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `edge_mapper.py` | Double-joins triples with node IDs, collects edge index tensors. Returns cached resolved edges DataFrame for reuse by `edge_feature_extractor.py`. `get_predicate_uri_mapping()` provides a small collect for metadata. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `feature_extractor.py` | Builds ontology-aware node feature vectors via `VectorLayout` (proportionally scaled segments): extracts class hierarchy, property schema, and literal values on executors; collects sparse entries (chunked for large types); scatters into dense tensors on driver. During `build_features()`, collects normalization stats, ontology schema snapshot, and slot mapping into small Python objects via `_collect_*` methods. `get_metadata_artifacts()` returns these for `MetadataCollector`. Imports `ONTOLOGY_NAMESPACE_INDICES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `edge_feature_extractor.py` | Builds derived edge feature vectors via `EdgeVectorLayout` (proportionally scaled segments): classifies edge types by category, extracts endpoint properties, encodes temporal signals / numeric contrast / relational context on executors; collects sparse entries per edge type; scatters into dense tensors on driver. Reuses cached resolved edges from `edge_mapper.py` — no double-join replay. `get_encoding_config()` and `get_edge_classification()` provide metadata for `MetadataCollector`. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `metadata_writer.py` | `MetadataCollector` accumulates metadata artifacts deposited by `constructor.py` during each step; `to_metadata_files()` produces six JSON-serializable dicts; `write_metadata_to_local()` writes them to the local metadata directory and `write_metadata_to_s3()` mirrors them to S3; `derive_metadata_prefix()` computes the metadata directory from the `.pt` filename/key | No (pure Python) |

### Scalability

The pipeline is designed to handle:
- **100+ ontologies** with different schemas and vocabularies
- **30-50M triples per month** with intraday market snapshots
- **Heterogeneous data types** (prices, rates, levels, changes, categorical)
- **Multiple temporal granularities** (intraday, daily, weekly, monthly, quarterly)
- **Dynamic schema evolution** as new data sources are added
- **Horizontal scaling** by adding Spark workers (and GPUs) — enrichment and PyG construction work distributes automatically
- **Bounded driver memory** — PyG construction collects only compact integer/float tensors, not URI strings. Chunked collection for large node types bounds peak Pandas memory. Chunked collection for large edge types follows the same pattern. `gc.collect()` between types reclaims fragmented memory (safe because it runs on the driver process only, after all Spark executor work is complete).
- **Configurable node vector dimension** — reducing `vector_dim` from 1024 to 512 halves driver memory for node feature tensors while preserving the same three-segment structure via proportional `VectorLayout` scaling
- **Configurable edge vector dimension** — reducing `edge_vector_dim` from 32 to 16 halves driver memory for edge feature tensors while preserving the same three-segment structure via proportional `EdgeVectorLayout` scaling
- **Selective edge featurization** — only high-value edge types receive feature vectors, avoiding wasted memory on constant vectors for structural edges
- **No double-join for edge features** — the expensive double-join (triples × node_id_df) runs exactly once in EdgeMapper; EdgeFeatureExtractor reuses the cached result
- **No Python UDFs in the hot path** — URI-to-name conversions, hash-based encoding, and numeric parsing use pure Spark expressions (JVM-native), avoiding Python serialization overhead on 30-50M rows
- **Controlled Parquet output** — configurable partition count prevents thousands of tiny files or few huge files
- **Efficient literal isolation** — anti-join against node_id_df filters out edge triples before numeric parsing, avoiding wasted computation on URI-valued objects
- **Canonical namespace registry** — `NAMESPACE_PREFIXES` and `ONTOLOGY_NAMESPACE_INDICES` in `rdf_utils.py` are the single source of truth, imported by `node_mapper.py`, `edge_mapper.py`, `feature_extractor.py`, and `edge_feature_extractor.py` to eliminate duplication
- **Negligible metadata overhead** — all metadata collect calls target small aggregated DataFrames (<5000 rows each); total metadata memory is under 1 MB; six JSON files are written after the `.pt` file with no impact on tensor collection or HeteroData assembly

## Testing

Live pass/fail status is the **tests** badge at the top of this README, which reflects the latest [GitHub Actions](.github/workflows/tests.yml) run on the default branch.

The suite runs entirely against a **local `SparkSession`** — no Spark cluster and no RAPIDS Accelerator are required. The same application code runs unchanged on the GPU cluster (RAPIDS is a drop-in SQL plugin), so plain local Spark is a valid way to test the enrichment and PyG logic.

Tests are split into three **run groups** by pytest marker — how expensive a test is and therefore where it runs. (Depth of coverage is a separate axis; see [Test tiers](#test-tiers) below, which maps onto these three groups.)

- **Fast suite** — everything *unmarked*, selected with `-m "not e2e"`. Runs on a stock `ubuntu-latest` runner (Java 17 + Python 3.12) on **every push and pull request** ([`tests.yml`](.github/workflows/tests.yml)); this is what the **tests** badge reflects.
- **End-to-end smoke** — marked `e2e`. Runs the real `build_graph` over small fixtures for all sources. It's heavy (~1,300 Spark stages regardless of data size) and does **not** reliably finish on a 7 GB GitHub runner, so it is **manual-only** ([`e2e.yml`](.github/workflows/e2e.yml), `workflow_dispatch`) and best run locally / on a capable machine.
- **Cluster submit smoke** — marked `cluster` (and `e2e`, so `-m "not e2e"` excludes it too). Submits the real job to a real standalone cluster. Never runs in CI, and **skips itself** unless `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set, so it costs contributors without a cluster nothing. See [Cluster smoke test](#cluster-smoke-test-a-real-cluster-not-local) below.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-test.txt          # runtime deps + pyspark + pytest

# Fast suite (what CI runs on every push/PR):
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest tests/ -m "not e2e"

# ...optionally in parallel (local machines only — NOT what CI runs). Each xdist
# worker is its own process and so builds its OWN SparkSession, trading memory
# for wall-clock: on a 20-core box, 8 workers cut the fast suite from ~248s to
# ~90s (2.8x). Returns flatten quickly because per-worker session startup is a
# fixed cost. Pick a number; avoid `-n auto`, which silently scales with
# whatever machine it lands on:
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest tests/ -m "not e2e" -n 8

# End-to-end pipeline smoke test (heavy; run locally / on a capable machine).
# bin/run_e2e_tests.sh wraps the env boilerplate below — SPARK_LOCAL_IP, the
# raised driver heap (the full pipeline fans out into ~1,300 stages and OOMs the
# default heap), and the RAPIDS toggle:
bin/run_e2e_tests.sh          # CPU (default)
bin/run_e2e_tests.sh gpu      # GPU via the RAPIDS Accelerator (auto-finds the jar)

# ...or invoke pytest directly. CPU:
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" \
  .venv/bin/python -m pytest tests/ -m e2e -s

# ...the same e2e test on GPU via the RAPIDS Accelerator (requires a GPU + the
# RAPIDS jar). SPARK_RAPIDS=1 enables the plugin; point RAPIDS_JAR at the jar:
SPARK_RAPIDS=1 RAPIDS_JAR=/path/to/rapids-4-spark_2.12-<version>.jar \
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell" \
  .venv/bin/python -m pytest tests/ -m e2e -s
```

The `gpu` mode of [`bin/run_e2e_tests.sh`](bin/run_e2e_tests.sh) resolves the RAPIDS jar via a glob (`/opt/spark/jars/rapids-4-spark_*.jar`, overridable with `RAPIDS_JAR` / `RAPIDS_JAR_DIR`), so a version bump on the host needs no edit.

The e2e test runs on **CPU by default**; set `SPARK_RAPIDS=1` to run it through the **RAPIDS Accelerator** on GPU. RAPIDS is a drop-in SQL plugin, so the application logic — and therefore every *pipeline* assertion — is identical on CPU and GPU; the toggle only changes where the DataFrame operators execute. (The one Python parsing UDF always runs on CPU under both.) The GPU settings mirror [`conf/spark-rapids.conf.template`](conf/spark-rapids.conf.template) / [`bin/submit_spark_job.sh`](bin/submit_spark_job.sh), minus the cluster-only GPU resource-scheduling confs that don't apply in `local[*]` mode.

Under `SPARK_RAPIDS=1` the suite additionally asserts that the query **really ran on the GPU** ([`tests/e2e/test_gpu_placement.py`](tests/e2e/test_gpu_placement.py); skipped on CPU runs). This matters because a plugin that fails to load — a missing or mismatched jar, say — does not raise: Spark carries on correctly on the CPU and every pipeline assertion still passes, so the suite would report a green *GPU* run while the GPU sat idle. The check disables adaptive query execution before reading the plan: under AQE the plan renders as CPU (`isFinalPlan=false`) even when the GPU is executing it, so a plan inspected with AQE on cannot prove GPU placement either way.

> **Note — the GPU run is *slower* on these fixtures, and that is expected.** The `gpu` mode is a **correctness / plumbing sanity check** (does the pipeline produce the same graph through RAPIDS?), not a benchmark. On the tiny e2e fixtures GPU wall-clock is higher than CPU for two reasons: (1) a **one-time JIT compile** of GPU kernels when the RAPIDS jar ships no precompiled binaries for the local GPU architecture, and (2) the fixtures are so small that **per-operator GPU launch and host↔device transfer overhead dominates** any compute savings. GPU acceleration only pays off at cluster data scale, where those fixed costs amortize. Do not read the local e2e timing as a CPU-vs-GPU verdict. (You may also see recoverable `RMM ... maximum pool size exceeded` messages — that is the deliberately conservative `spark.rapids.memory.gpu.allocFraction` cap being hit; raise it for real workloads.)

`pyspark` is cluster-provided in production and is therefore not in `requirements.txt`; `requirements-test.txt` layers it (and `pytest`) on top for local and CI runs.

### Cluster smoke test (a real cluster, not `local[*]`)

Everything above runs against a local `SparkSession`. That leaves one path untested: **the job as actually submitted to a cluster**. Local mode reads none of the cluster's `spark-defaults.conf`, has no master or workers (so executor/GPU resource negotiation — the thing that decides whether a GPU job is ever *scheduled* — never happens), and imports the job in-process instead of zipping it and shipping it to executors. A cluster whose workers advertise no GPU will accept the job and simply never schedule it, hanging forever while the entire local suite stays green.

[`tests/e2e/test_cluster_submit.py`](tests/e2e/test_cluster_submit.py) (marker: `cluster`) submits the real job through [`bin/submit_spark_job.sh`](bin/submit_spark_job.sh) and asserts it completes, writes its artifacts to shared storage, **produces a structurally valid graph**, and **actually placed operators on the GPU**. The submission is bounded by a timeout, because an unschedulable GPU request hangs rather than failing. It is **skipped unless `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set**, so CI and contributors without a cluster are unaffected.

It submits **two** jobs. `--mode enrichment_only` is the fast leg — the quick signal on launcher, submit, and IAM wiring. `--mode full` additionally runs PyG construction, which is the only place the **driver-side artifact writers** meet real object storage. That distinction is not academic: `save_pyg_local()` and `write_metadata_to_local()` both used `os.makedirs` + plain `open()`, and plain Python I/O treats `s3a://bucket/key` as a *relative path* — it creates a junk `./s3a:/bucket/key` tree under the driver's working directory, logs `Saved ... to s3a://...`, and exits 0. The graph and its metadata never reach the object store, and nothing raises. Because the cluster leg only ever ran `enrichment_only`, neither writer was ever invoked with a non-local URI and the defect survived undetected (it is the same defect `#197` fixed for the job manifest). All three writers now share one scheme-aware implementation, [`spark_jobs/utils/fs_utils.py`](spark_jobs/utils/fs_utils.py) — **any new driver-side write must go through `write_bytes()`**, which routes local paths to direct I/O, URIs through the Hadoop FileSystem API, and *raises* rather than silently localizing a URI it cannot reach.

The `full` leg then **downloads its own artifacts back out of object storage and validates them**, using the very same `_assert_valid_graph_and_metadata` helper the local e2e suite uses — edge indices in range, tensor dtypes, finiteness, metadata/tensor agreement, node_index coverage, temporal structure, edge origins. Without this the cluster leg only ever asserted that *bytes arrived*: a run that produced a structurally broken graph passed every check, because the file was the right size in the right place. The helper is imported rather than reimplemented, so the cluster's graph is held to exactly the standard the local one is and the two cannot drift apart. The S3 tree is mirrored into a temp dir with its relative layout intact and handed a real `JobConfig`, so path derivation is the pipeline's own code rather than a second copy of it. Cost is ~2.5s of download and validation against a ~3-minute submission.

```bash
export SPARK_HOME=/opt/spark
export SPARK_MASTER_URL=spark://<master>:7077
export CLUSTER_SMOKE_OUTPUT_PATH=s3a://<bucket>/<prefix>   # must be reachable by EVERY node
.venv/bin/python -m pytest tests/e2e/test_cluster_submit.py -m cluster -q
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARK_MASTER_URL` | — | Required; the test skips without it. |
| `CLUSTER_SMOKE_OUTPUT_PATH` | — | Required. Where the job writes. On a cluster this must be shared storage (`s3a://`): executors are spread across machines, so a `file://` path resolves to a different local disk on each one and the commit protocol cannot assemble the output. |
| `CLUSTER_SMOKE_SOURCE_PATH` | staged fixtures | Source data. If unset, the repo's own e2e fixtures are uploaded under the output path, so a fresh clone can run this against any cluster with nothing copied onto the nodes by hand. |
| `CLUSTER_SMOKE_TIMEOUT` | `900` | Seconds before a hung submission is killed (with its driver). |
| `CLUSTER_SMOKE_MAX_SOURCES` | all | Cap the number of sources, for a faster signal. The pipeline's cost scales with **source count**, not data size — one source ≈ 30s, all seven ≈ 6min. Applies to both legs, so it roughly halves the suite's wall-clock. |
| `CLUSTER_SMOKE_EXPECT_GPU` | `1` | Set `0` to skip the GPU-placement assertion (e.g. a CPU-only cluster). |

Three failure modes this test exists to catch, all of which otherwise produce a **green** suite: RAPIDS silently falling back to CPU (the job still succeeds and still produces a correct graph), a job that writes a structurally broken graph to the right place under the right name (caught only by opening the `.pt`, which is what the validation step above added), and the driver OOMing while *planning* the multi-source query — see [`bin/submit_spark_job.sh`](bin/submit_spark_job.sh) on `spark.sql.maxPlanStringLength`, and `_settle()` in [`spark_jobs/enrichment/pipeline.py`](spark_jobs/enrichment/pipeline.py) on why the enrichment phases truncate their logical plans.

### Test tiers

Test depth is calibrated to risk rather than applied uniformly — deeper coverage only where the logic is genuinely subtle, to keep maintenance debt proportional to value.

Tiers are a **second, finer axis** than the three run groups above, not a competing one. A marker answers *where and when a test runs*; a tier answers *how deep it cuts*. The two line up cleanly — tiers **1–4 together are the fast suite**, and each of the last two tiers is one marker-gated group of its own:

| Run group (from above) | Tiers it contains | Marker |
|---|---|---|
| Fast suite — every push/PR | 1, 2, 3, 4 | *(none)* — selected with `-m "not e2e"` |
| End-to-end smoke — manual-only | 5 | `e2e` |
| Cluster submit smoke — opt-in | 6 | `cluster` (+ `e2e`) |

Each tier below repeats its run group, so no row has to be cross-referenced against the table above:

| Tier | Scope | Examples |
|------|-------|----------|
| **1 — pure / no-Spark**<br>*fast suite* | Import-time integrity, vector geometry, hand-maintained pattern tables, and RDF parse determinism. Sub-second. | `test_imports.py` (imports every `spark_jobs` module), `test_vector_layout.py` / `test_edge_vector_layout.py` (`VectorLayout` / `EdgeVectorLayout` boundaries), `test_source_patterns.py` (NOAA/market/SEC pattern-dict integrity), `test_bnode_determinism.py` (blank-node labels are content-derived, so the same Turtle parses identically every time — rdflib's own labels are random per parse) |
| **2 — linker smokes**<br>*fast suite* | Each enrichment module's `enrich()` driven end-to-end over tiny in-memory triples: one happy path + one short-circuit (foreign input for the intra-source linkers; a single detected source, and two sources with nothing linkable, for the cross-source linker; non-temporal input for the temporal unifier). | `test_{bls,noaa,market,sec}_linker.py`, `test_cross_source_linker.py`, `test_temporal_unifier.py` |
| **3 — targeted deep**<br>*fast suite* | One focused test on each module's trickiest computation (including the negative case), where a silent regression would be costly. | severity escalation (NOAA), option moneyness (market), CIK unification (SEC), temporal sequencing (BLS), state-FIPS geographic chain (cross-source), expiration-date period derivation (temporal unifier) |
| **4 — construction internals**<br>*fast suite* | Value-level unit tests of the PyG construction modules — exact node IDs, edge-index contents + `(src_id, dst_id)` ordering, config filters, determinism, the node/edge feature **encoding** (a known triple lands in the layout-reserved vector slot with the expected value: class-identity/categorical multi-hots, depth-weighted `subClassOf` class hierarchy, property-schema presence/domain-range/property-hierarchy slots, z-score numeric normalization, edge temporal/numeric-contrast/moneyness signals, label-similarity Jaccard on correlation edges, the escalation severity-delta fallback, plus `_classify_relation` per category and a categorical-determinism guard), the final `build_hetero_data` assembly (feature↔node-ID alignment via encoding-independent sentinels, edge endpoints within per-type ID ranges, per-type counts), and the six metadata JSON files' content (keys, counts, type names, feature-segment structure) — asserting what the `e2e` smoke only checks structurally or for presence. Runs in the fast suite; `test_metadata_writer.py` is pure-Python (no `SparkSession`). | `test_node_mapper.py`, `test_edge_mapper.py`, `test_feature_extractor.py`, `test_edge_feature_extractor.py`, `test_constructor.py`, `test_metadata_writer.py` |
| **5 — pipeline smoke**<br>*`e2e` marker — manual-only* | The real `build_graph` end-to-end over tiny committed RDF fixtures: both source loaders (`.nt` and turtle-parquet) × all three modes (`full`, and the `enrichment_only` → `pyg_only` split), asserting a valid `.pt` + all six metadata JSONs and layout-consistent tensor shapes. Also a **reproducibility** check that runs the pipeline twice (each in its own process, to isolate the per-run driver heap; the two independent runs are launched **concurrently**, and started early so they overlap the rest of the e2e suite rather than adding to it — cutting the twin-run cost from ~103s to ~21s of join time, at the price of two extra driver JVMs alive during the overlap. Set `TWIN_RUNS_SEQUENTIAL=1` to run them one-at-a-time on a memory-constrained host) and asserts the two graphs, feature tensors, and metadata are identical — an integration property no unit test can see. The check is split in two: `test_output_is_reproducible` compares the node/edge inventory, per-type counts, `edge_index`, edge features, **every** node type's feature tensors, and each metadata file exactly (modulo the allow-listed `build_timestamp`); `test_jolts_features_reproducible` additionally compares the metadata in **serialized** form, which catches dict key/insertion-order drift the parsed comparison cannot. Both carried concessions to a known non-determinism until it was fixed — `jolts_*` tensors were skipped and metadata was compared order-insensitively — so the guard is now strictly stronger than before that bug existed. The cause was twofold: rdflib assigns blank nodes a fresh **random** label on every parse (which then hash into different feature slots), and the metadata builders used unordered `collect()` (Spark returns rows in task-completion order). Blank-node determinism is additionally guarded by fast unit tests that *do* run in CI (`tests/test_bnode_determinism.py`), since this e2e suite does not. Catches wiring / API-mismatch bugs the unit tiers can't. | `tests/e2e/test_pipeline_smoke.py` |
| **6 — cluster submit smoke**<br>*`cluster` marker — opt-in* | The real job submitted to a real standalone cluster through `bin/submit_spark_job.sh`: asserts it completes, writes artifacts to shared storage, and ran on the GPU. The only tier that exercises `--py-files`/venv packaging, executor↔GPU resource negotiation, and the cluster's own `spark-defaults.conf`. Skipped unless `SPARK_MASTER_URL` is set. | `tests/e2e/test_cluster_submit.py` |

Exhaustive per-relationship assertions are intentionally **not** written — the targeted deep tests capture the high-risk logic without the brittleness of pinning every output.

### Local test report

[`bin/generate_report.sh`](bin/generate_report.sh) runs the suites and writes a combined report to [`reports/tests/report.html`](reports/tests/report.html) + [`reports/tests/report.json`](reports/tests/report.json) (same data, two formats), overwriting the previous pair each run. (Reports are namespaced by kind under `reports/` — e.g. `reports/tests/` — to leave room for other report types.)

```bash
bin/generate_report.sh            # fast + e2e (CPU) [+ e2e (GPU) if a GPU + RAPIDS jar are present]
bin/generate_report.sh --no-gpu   # never attempt the GPU run

# Full picture: also submit the real job to a standalone cluster (adds the cluster suite).
SPARK_MASTER_URL=spark://<master>:7077 \
CLUSTER_SMOKE_OUTPUT_PATH=s3a://<bucket>/<prefix> \
  bin/generate_report.sh
```

This is **local-only and not committed** — `reports/tests/` is gitignored. The e2e suite is too heavy for the GitHub Actions runners, so the report can't be produced in CI; instead each developer runs `bin/generate_report.sh` on their own branch to verify locally (the **tests** badge covers the fast suite in CI). The report always includes the fast unit suite and the CPU e2e run; it adds the GPU (RAPIDS) e2e run when the hardware is available, and the [cluster smoke test](#cluster-smoke-test-a-real-cluster-not-local) when `SPARK_MASTER_URL` and `CLUSTER_SMOKE_OUTPUT_PATH` are set (source data auto-stages, so `CLUSTER_SMOKE_SOURCE_PATH` is optional). [`bin/generate_test_report.py`](bin/generate_test_report.py) is the underlying renderer (reads pytest JUnit XML).

---

## Lint

`lint.yml` runs on every push and pull request, and [`.githooks/pre-commit`](.githooks/pre-commit) runs the same checks locally so the two cannot disagree.

| Workflow | What it checks |
|---|---|
| [`lint.yml`](.github/workflows/lint.yml) | `ruff check .` — rules in [`.ruff.toml`](.ruff.toml) |

The rule set is deliberately **correctness-only** (`E4`, `E7`, `E9`, `F`, `W6`): syntax errors, undefined names, unused imports and variables, bare `except`, invalid escape sequences. No formatting, import-ordering, or type-annotation rules are enabled, so a red build always means a real defect rather than a style preference. Notebooks are excluded — several carry pasted tabular output inside code cells and are not parseable Python.

The `ruff` version is pinned in the workflow so a new upstream release cannot redden an untouched branch; bump it deliberately.

### Running it by hand

```bash
pip install ruff==0.16.1
ruff check .
```

### Enabling the hook

```bash
git config core.hooksPath .githooks
```

The hook checks only **staged** files, and skips with a notice (rather than failing) when the linter is not installed — a fresh clone stays committable, and CI remains the enforcement. Bypass with `git commit --no-verify`.
