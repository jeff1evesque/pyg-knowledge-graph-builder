# PyTorch Geometric Knowledge Graph Builder

> GPU-accelerated Apache Spark pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

--8<-- "README.md:overview"

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Raw Data Sources (local filesystem or S3 via s3a://)       │
│                                                            │
│ Turtle Parquet format (a column of Turtle blobs):          │
│ ├── BLS Economic Data (10 categories, ~100 mappers)        │
│ ├── Market Data (1 mapper, intraday snapshots)             │
│ ├── NOAA Weather Alerts (1 mapper)                         │
│ └── SEC Data (4 categories, 4 mappers)                     │
│                                                            │
│ N-Triples (.nt) is still supported by the loader; no       │
│ source currently ships it.                                 │
│                                                            │
│ Total: 100+ mappers and ontologies                         │
│ Volume: one day of intraday market snapshots loads         │
│ 322.7M triples, 421.4M after enrichment. Market is         │
│ 99.5% of that; BLS 1.3M, SEC 198K, NOAA 143K.              │
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
