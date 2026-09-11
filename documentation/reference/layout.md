# Project Structure

```
pyg-knowledge-graph-builder/
├── bin/
│   ├── profiles/
│   │   ├── large-run.env                   # opt-in sizing for a full-day run
│   │   └── pyg-assembly.env                # sizing for the assembly leg (--mode pyg_only)
│   ├── census_sec_terms.py                 # per-form census of the SEC filings vocabulary
│   ├── check_vocabulary_drift.py           # terms we key on that upstream no longer emits
│   ├── generate_bls_e2e_fixtures.py        # rebuild the BLS e2e fixtures from the archive
│   ├── generate_market_e2e_fixtures.py     # rebuild the market e2e fixtures from snapshots
│   ├── generate_report.sh                  # run every suite -> reports/tests/report.{html,json}
│   ├── generate_sec_e2e_fixtures.py        # rebuild the SEC e2e fixtures from the archive
│   ├── generate_test_report.py             # the report renderer (reads pytest JUnit XML)
│   ├── package_venv.sh                     # package the venv so executors can run our Python
│   ├── publish_run.py                      # copy a finished run to the published prefix
│   ├── record_run_outcome.sh               # summarise a finished (or abandoned) run
│   ├── run_e2e_tests.sh                    # the e2e smoke suite, local SparkSession (CPU/GPU)
│   ├── run_tests.sh                        # the fast suite, parallel (sibling of run_e2e_tests.sh)
│   ├── selfloops.py                        # count nodes that are their own object
│   ├── stage_sources.sh                    # mirror source prefixes onto each worker's disk
│   ├── stall_watchdog.py                   # catch a stalled stage, dump the executors
│   └── submit_spark_job.sh                 # spark-submit launcher (RAPIDS/GPU)
├── conf/
│   └── spark-rapids.conf.template          # reference RAPIDS spark-defaults
├── spark_jobs/
│   ├── build_graph.py                      # Main Spark job entry point
│   ├── graph/                              # The job, minus its orchestration
│   │   ├── __init__.py
│   │   ├── config.py                       # JobConfig + the CLI; rejects a
│   │   │                                   # configuration that cannot work
│   │   │                                   # before Spark starts
│   │   ├── loading.py                      # The three loaders, the dispatcher,
│   │   │                                   # and the per-source stamp/counts
│   │   ├── persistence.py                  # Interim Parquet + descriptor, the
│   │   │                                   # final .pt/metadata/node index, and
│   │   │                                   # the job manifest
│   │   └── turtle.py                       # One Turtle blob → the four triple
│   │                                       # columns, plus the bounded batching
│   │                                       # that carries them. The one module
│   │                                       # executors import for themselves,
│   │                                       # so its imports are pinned to the
│   │                                       # executor venv by a test
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
│   │   ├── edge_encoders.py                # The three edge-vector segments, one
│   │   │                                   # function each; called by
│   │   │                                   # edge_feature_extractor.py
│   │   ├── vector_layout.py                # VectorLayout — every node-vector segment
│   │   │                                   # boundary, computed from vector_dim
│   │   ├── edge_vector_layout.py           # EdgeVectorLayout — the same for the edge
│   │   │                                   # vector, from edge_vector_dim
│   │   ├── collision_report.py             # Hash collision stats over the node slot
│   │   │                                   # assignments, and the class_identity
│   │   │                                   # capacity check that reads them
│   │   ├── sparse_scatter.py               # Writes sparse (key, dim, value) rows into
│   │   │                                   # a pre-allocated dense tensor; shared by
│   │   │                                   # both feature extractors
│   │   └── metadata_writer.py              # MetadataCollector (accumulates artifacts
│   │                                       # during construction steps);
│   │                                       # write_metadata_to_local() / _to_s3()
│   │                                       # (serialize the six JSON files, and
│   │                                       # report each one's digest);
│   │                                       # write_checksums_to_local() / _to_s3()
│   │                                       # (checksums.json, written last);
│   │                                       # derive_metadata_prefix() (naming convention)
│   └── utils/
│       ├── __init__.py
│       └── rdf_utils.py                    # Namespace constants, URI helpers, canonical
│                                           # namespace registry (NAMESPACE_PREFIXES,
│                                           # ONTOLOGY_NAMESPACE_INDICES)
├── tests/                                  # Unit and integration tests
├── documentation/                          # These pages (MkDocs reads from here,
│                                           # not docs/, which is ignored)
├── .gitignore
├── mkdocs.yml                              # Site config: nav, theme, strict build
├── README.md                               # Landing page; the site has the rest
├── requirements-docs.txt                   # MkDocs Material, pinned
└── requirements.txt
```

## Module Roles

| Module | Role | Uses PySpark? |
|--------|------|--------------|
| `rdf_utils.py` | Namespace constants, URI string helpers, canonical `NAMESPACE_PREFIXES` and `ONTOLOGY_NAMESPACE_INDICES` registries (single source of truth for all PyG builder modules) | No (pure Python) |
| `patterns.py` / `correlations.py` / `measurements.py` | Configuration dictionaries (sector keywords, correlation definitions) | No (pure Python) |
| `pipeline.py` | Orchestrates enrichment steps, manages triples DataFrame | Yes |
| `temporal_unifier.py` | Produces unified month/year/quarter triples | Yes |
| `bls_linker.py`, `sec_linker.py`, `market_linker.py`, `noaa_linker.py` | Produce intra-source enrichment triples | Yes |
| `cross_source_linker.py` | Produces cross-source enrichment triples | Yes |
| `ontology_mapper.py` | Produces equivalence mapping triples | Yes |
| `build_graph.py` | The entry point and the orchestration only: `main()`, the four execution modes, the enrichment and PyG-construction phases, the SparkSession, the work-dir preflight and the final banner. Everything it reads, writes or is configured by now lives in `graph/` | Yes (orchestration) |
| `graph/config.py` | `JobConfig` — the job's whole contract with its caller: resolves every path the run reads and writes, and REJECTS a configuration that cannot work (a mode without its inputs, a staged mirror that is not there, an SEC prefix naming an unhandled feed) before Spark starts. Also `parse_args()`, `staged_local_path()`, `period_partition()`, and the probe that answers whether the PyG builder is importable | No (pure Python) |
| `graph/loading.py` | The three ways triples get in — `load_ntriples_to_dataframe()` for `.nt`, `load_turtle_parquet_to_dataframe()` for Turtle blobs, and `load_source_triples()` which dispatches per source path, stamps each row with `source_label()` and unions the result. The stamp is what makes the `s3` and `local` input modes report identical per-source counts | Yes (heavy, pure Spark expressions) |
| `graph/persistence.py` | Everything written down and read back: the interim enriched Parquet and its `dataset.json` descriptor (how a `pyg_only` run learns what the `enrichment_only` run read), the final `.pt` / metadata / node index written locally and mirrored to S3, and the job manifest. Also the digests: `_HashingWriter` hashes the `.pt` during the write that already happens, and `_artifact_records()` names each digested file relative to its period directory for `checksums.json` | Yes (writes are distributed; the `.pt` is driver-side) |
| `graph/turtle.py` | One Turtle blob → the four triple columns (`turtle_to_rows()`, its skip policy, the blank-node labels that make a parse reproducible), and `turtle_batches_to_arrow()` — the bounded batching that replaced the array-returning UDF of #380. **The only module executors import for themselves**: `build_graph.py` is submitted by path, so its own functions ship by value and are never imported, while these are pickled by reference and really are imported inside the executor venv. Its imports are therefore pinned to what `requirements-executor.txt` carries, by `tests/test_executor_imports.py` | Yes (the parse itself runs on executors) |
| `constructor.py` | Orchestrates PyG HeteroData construction from triples DataFrame (5 steps: node IDs, edge indices, node features, edge features, assembly); initializes `MetadataCollector`; calls `register_*` methods after each step; returns `(HeteroData, MetadataCollector)` | Yes (orchestration) |
| `node_mapper.py` | Discovers node types, assigns per-type integer IDs via Window functions. `get_type_uri_mapping()` provides a small collect for metadata. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `edge_mapper.py` | Double-joins triples with node IDs, collects edge index tensors. Returns cached resolved edges DataFrame for reuse by `edge_feature_extractor.py`. `get_predicate_uri_mapping()` provides a small collect for metadata. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `feature_extractor.py` | Builds ontology-aware node feature vectors via `VectorLayout` (proportionally scaled segments): extracts class hierarchy, property schema, and literal values on executors; collects sparse entries (chunked for large types); scatters into dense tensors on driver. During `build_features()`, collects normalization stats, ontology schema snapshot, and slot mapping into small Python objects via `_collect_*` methods. `get_metadata_artifacts()` returns these for `MetadataCollector`. Imports `ONTOLOGY_NAMESPACE_INDICES` from `rdf_utils.py`, and the collision report and class-identity guard from `collision_report.py` | Yes (heavy, pure Spark expressions) |
| `edge_feature_extractor.py` | Builds derived edge feature vectors via `EdgeVectorLayout` (proportionally scaled segments): classifies edge types by category, extracts endpoint properties, calls the three segment encoders in `edge_encoders.py` on executors; collects sparse entries per edge type; scatters into dense tensors on driver. Reuses cached resolved edges from `edge_mapper.py` — no double-join replay. `get_encoding_config()` and `get_edge_classification()` provide metadata for `MetadataCollector`. Imports `NAMESPACE_PREFIXES` from `rdf_utils.py` | Yes (heavy, pure Spark expressions) |
| `edge_encoders.py` | The three edge-vector segments, one function each: `encode_temporal_signals()`, `encode_numeric_contrast()` (with the cross-property fallback behind it) and `encode_relational_context()`. Each takes the edges of one type and returns lazy `(edge_idx, dim, value)` entries; every dim index comes from the `EdgeVectorLayout` handed in. `PropertyRows` and the row widths beside it describe the endpoint property frames `edge_feature_extractor.py` builds for them, which is why the dependency runs extractor → encoders and never back | Yes (pure Spark expressions, nothing collected) |
| `vector_layout.py` | `VectorLayout` — turns `vector_dim` into the start index and width of every node-vector segment and sub-segment, and holds the proportions those widths come from. The encoders ask it for a slot and `metadata_writer.py` publishes the same layout, so a vector and the metadata describing it cannot disagree | No (pure Python) |
| `edge_vector_layout.py` | `EdgeVectorLayout` — the same for the edge vector, from `edge_vector_dim`, over the temporal / numeric-contrast / relational-context segments | No (pure Python) |
| `collision_report.py` | `compute_collision_report()` — what collided in the node vector's slot assignments, published as the `collision_report` block of `slot_mapping.json` — and `check_class_identity_capacity()`, which reads that report and raises `ClassIdentityCapacityError` when the class_identity segment can no longer separate the build's classes. Runs once per build on the driver, over a few hundred Python dicts | No (pure Python) |
| `sparse_scatter.py` | `scatter_sparse_entries()` — writes one frame of sparse `(key, dim, value)` rows into a pre-allocated dense tensor. The five collect paths in `feature_extractor.py` and `edge_feature_extractor.py` all end in this write; how they collect (persist level, chunking) stays with them | No (pure Python) |
| `metadata_writer.py` | `MetadataCollector` accumulates metadata artifacts deposited by `constructor.py` during each step; `to_metadata_files()` produces six JSON-serializable dicts; `write_metadata_to_local()` writes them to the local metadata directory and `write_metadata_to_s3()` mirrors them to S3, both returning the size and SHA-256 of what they wrote; `write_checksums_to_local()` / `write_checksums_to_s3()` then write `checksums.json` from those digests; `derive_metadata_prefix()` computes the metadata directory from the `.pt` filename/key and `metadata_dir_name()` gives just its last segment, which is the prefix the record's paths carry | No (pure Python) |

## Scalability

The pipeline is designed to handle:
- **100+ ontologies** with different schemas and vocabularies
- **322.7M triples in a single day** with intraday market snapshots
- **Heterogeneous data types** (prices, rates, levels, changes, categorical)
- **Multiple temporal granularities** (intraday, daily, weekly, monthly, quarterly)
- **Dynamic schema evolution** as new data sources are added
- **Horizontal scaling** by adding Spark workers (and GPUs) — enrichment and PyG construction work distributes automatically
- **Bounded driver memory** — PyG construction collects only compact integer/float tensors, not URI strings. Chunked collection for large node types bounds peak Pandas memory. Chunked collection for large edge types follows the same pattern. `gc.collect()` between types reclaims fragmented memory (safe because it runs on the driver process only, after all Spark executor work is complete).
- **Configurable node vector dimension** — reducing `vector_dim` from 1024 to 512 halves driver memory for node feature tensors while preserving the same three-segment structure via proportional `VectorLayout` scaling
- **Configurable edge vector dimension** — reducing `edge_vector_dim` from 32 to 16 halves driver memory for edge feature tensors while preserving the same three-segment structure via proportional `EdgeVectorLayout` scaling
- **Selective edge featurization** — only high-value edge types receive feature vectors, avoiding wasted memory on constant vectors for structural edges
- **No double-join for edge features** — the expensive double-join (triples × node_id_df) runs exactly once in EdgeMapper; EdgeFeatureExtractor reuses the cached result
- **No Python UDFs in the hot path** — URI-to-name conversions, hash-based encoding, and numeric parsing use pure Spark expressions (JVM-native), avoiding Python serialization overhead on 322.7M rows
- **Controlled Parquet output** — configurable partition count prevents thousands of tiny files or few huge files
- **One parse per source** — the datatype markers are derived from the cached parse rather than from a second, uncached read of the same frame, so the rdflib UDF runs once per source instead of twice (measured: load phase 1,346.3s → 727.7s on identical input)
- **Loaded partitions track the data, not the shuffle default** — the datatype markers are coalesced to one partition and unioned once after that cached parse, not once per source, so a vocabulary-sized frame no longer adds 200 partitions per source to the cached triples that every enrichment stage reads (see [Sizing a large run](../operations/running-a-job.md#sizing-a-large-run))
- **Efficient literal isolation** — anti-join against node_id_df filters out edge triples before numeric parsing, avoiding wasted computation on URI-valued objects
- **Canonical namespace registry** — `NAMESPACE_PREFIXES` and `ONTOLOGY_NAMESPACE_INDICES` in `rdf_utils.py` are the single source of truth, imported by `node_mapper.py`, `edge_mapper.py`, `feature_extractor.py`, and `edge_feature_extractor.py` to eliminate duplication
- **Negligible metadata overhead** — all metadata collect calls target small aggregated DataFrames (<5000 rows each); total metadata memory is under 1 MB; seven JSON files are written after the `.pt` file with no impact on tensor collection or HeteroData assembly

