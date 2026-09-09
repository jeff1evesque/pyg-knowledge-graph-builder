# PyTorch Geometric Knowledge Graph Builder

> GPU-accelerated Apache Spark pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

[![tests](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/tests.yml)
[![unicode](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/unicode.yml/badge.svg)](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/actions/workflows/unicode.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Apache Spark](https://img.shields.io/badge/Apache-Spark-orange.svg)](https://spark.apache.org/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-Accelerator-green.svg)](https://docs.nvidia.com/spark-rapids/)

**[Documentation](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/)** — the design manual, the output reference, and the
operator runbook, split into pages. This file is the short version.

<!-- --8<-- [start:overview] -->
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
<!-- --8<-- [end:overview] -->

## Architecture

Raw RDF arrives as a column of Turtle blobs, one partitioned dataset per source.
A single Spark job parses it into one triples DataFrame, enriches that DataFrame
in place, and builds a PyTorch Geometric `HeteroData` object from the result.
Parsing, enrichment and feature extraction all run on executors with the RAPIDS
Accelerator; only compact tensors cross to the driver, where the graph is
assembled and saved as a `.pt` file with six metadata JSON files beside it.

The job runs in three modes — the full pipeline, enrichment only (which stops at
the reusable enriched Parquet), and PyG only (which starts from it). The second
and third exist so that graph-structure experiments do not pay for enrichment
again.

The diagrams, the triples-DataFrame schema, and why this is PySpark rather than
rdflib are in [Architecture](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/#architecture) on the documentation home page.

## Documentation

**Design** — how the graph is shaped

- [Node feature vectors](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/design/node-features/) — the universal 1024-d ontology-aware vector
- [Edge feature vectors](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/design/edge-features/) — the selective 32-d derived vector
- [PyG construction](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/design/construction/) — node IDs, edge resolution, assembly
- [Enrichment](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/design/enrichment/) — temporal unification, intra- and cross-source linking

**Reference** — what a build produces

- [Metadata files](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/reference/outputs/) — the six JSON files and `node_index/`
- [Data sources](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/reference/sources/) — what is ingested
- [Project layout](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/reference/layout/) — the module map

**Operations** — running it

- [Memory and scaling](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/operations/memory-and-scaling/) — driver budget, chunked collection
- [Running a job](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/operations/running-a-job/) — parameters, local disk, cluster prerequisites, sizing
- [Testing](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/operations/testing/) — the three run groups and the local report
- [Lint](https://jeff1evesque.github.io/pyg-knowledge-graph-builder/operations/lint/) — ruff and the pre-commit hook

## Editing the documentation

The pages are Markdown under `documentation/`, built with MkDocs Material and
published by `.github/workflows/docs.yml` on every push to `master`. To preview
them:

```bash
pip install -r requirements-docs.txt
mkdocs serve          # http://127.0.0.1:8000
```

Run `mkdocs build` from the repository root — Overview and Key Features above are
pulled into the site's home page by path, and the build is strict, so a broken
internal link fails it.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
