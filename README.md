# PyTorch Geometric Knowledge Graph Builder

> Serverless pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![AWS Glue](https://img.shields.io/badge/AWS-Glue-orange.svg)](https://aws.amazon.com/glue/)

## Overview

PyTorch Geometric Knowledge Graph Builder is a serverless pipeline that transforms raw RDF data from multiple heterogeneous sources into enriched knowledge graphs and constructs PyTorch Geometric `HeteroData` objects ready for Graph Neural Network (GNN) training.

The pipeline processes data from **100+ domain-specific ontologies** spanning economic indicators, financial filings, market data, and environmental alerts. All enrichment logic runs as **distributed PySpark DataFrame operations** on AWS Glue, enabling horizontal scaling across the cluster rather than bottlenecking on a single-threaded in-memory graph.

PyG construction also leverages Spark executors for all heavy computation (node ID assignment, edge resolution, feature extraction). Only compact integer and float tensors cross the Spark → driver boundary for final `HeteroData` assembly.

The pipeline supports three execution modes:

- **Full Pipeline**: End-to-end RDF enrichment and PyG graph construction
- **Enrichment Only**: Create reusable enriched RDF artifacts
- **PyG Construction Only**: Rapidly experiment with different PyG graph structures from existing enriched RDF

### Key Features

- **Large-Scale Integration**: Processes 100+ ontologies with tens of millions of triples per time period
- **Distributed Enrichment**: All enrichment runs as PySpark DataFrame operations across Spark executors
- **Distributed PyG Construction**: Node ID assignment, edge resolution, and feature extraction run on Spark executors — only compact tensors are collected to the driver
- **Temporal Unification**: Unified temporal entities across all data sources
- **Intra-Source Linking**: Automatic relationship discovery within data source families
- **Cross-Source Linking**: Automatic relationship discovery across heterogeneous datasets
- **PyTorch Geometric Output**: Native `HeteroData` objects with configurable node/edge types
- **Per-Type Feature Isolation**: Each node type carries only its ontology-relevant features, not a single wide matrix across all 100+ ontologies
- **Flexible Graph Construction**: Experiment with different graph structures without re-enrichment
- **Serverless Architecture**: Fully managed AWS Glue, no infrastructure to maintain
- **Experiment-Friendly**: Rapid iteration on PyG graph structures (5-10 min per experiment)
- **Reusable Artifacts**: Enriched triples (Parquet) can generate multiple PyG graphs

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Raw Data Sources (S3)                                      │
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
│  │ Read RDF     │──▶│ Enrichment    │──▶│ Build PyG    │   │
│  │ (N-Triples   │   │ (PySpark      │   │ (PySpark     │   │
│  │  → triples   │   │  DataFrames   │   │  executors   │   │
│  │  DataFrame)  │   │  on executors)│   │  → driver    │   │
│  └──────────────┘   └───────────────┘   │  tensors)    │   │
│                                         └──────────────┘   │
│                                                            │
│ Mode 1: Full Pipeline                                      │
│   Raw RDF → triples_df → Enrich → Build PyG HeteroData     │
│                                                            │
│ Mode 2: Enrichment Only                                    │
│   Raw RDF → triples_df → Enrich → Save Parquet to S3       │
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

All RDF data is loaded into a single **triples DataFrame** that serves as the universal graph representation throughout the pipeline:

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

Enrichment steps read from this DataFrame, produce new triples DataFrames, and union them back. PyG construction reads the final enriched DataFrame, assigns integer node IDs, resolves edges, and extracts features — all on Spark executors. Only compact tensors cross to the driver for final `HeteroData` assembly.

### Why PySpark Instead of rdflib/SPARQL

| Aspect | rdflib + SPARQL | PySpark DataFrames |
|--------|----------------|-------------------|
| Execution | Single Python process on Glue driver | Distributed across all Spark executors |
| Memory | Entire graph must fit in driver RAM | Partitioned across cluster |
| Query optimization | None (sequential iteration) | Catalyst optimizer, predicate pushdown, broadcast joins |
| Parallelism | None | Automatic partitioning |
| Glue DPU utilization | Pays for cluster, uses 1 core | Uses all allocated DPUs |
| Join pattern | Python dict lookups or nested SPARQL | Distributed hash/sort-merge joins |

rdflib Namespace objects are used as **URI string constants** in the enrichment modules for readability — they produce plain strings and don't hold or query graph data. The PyG builder modules use plain string constants directly.

## PyG Construction Pipeline

The PyG builder converts the enriched triples DataFrame into a PyTorch Geometric `HeteroData` object through four steps, with all heavy computation on Spark executors:

```
triples_df (enriched, on executors)
    │
    ├── Step 1: NodeMapper (on executors)
    │   ├── Discover node types from rdf:type triples
    │   ├── Assign canonical type per entity (most specific wins)
    │   ├── Assign per-type 0-indexed integer IDs via Window functions
    │   └── Output: node_id_df (uri, node_id, node_type) — cached on executors
    │              node_counts Dict[str, int] — small collect to driver
    │
    ├── Step 2: EdgeMapper (on executors → driver tensors)
    │   ├── Double-join triples with node_id_df (subject → src_id, object → dst_id)
    │   ├── Inner join on object naturally filters out literal properties
    │   ├── Derive relation names from predicate URIs
    │   ├── Collect per-edge-type [2, num_edges] int64 arrays via toPandas()
    │   └── Output: Dict[(src_type, relation, dst_type) → LongTensor]
    │
    ├── Step 3: FeatureExtractor (on executors → driver tensors)
    │   ├── Extract numeric properties (auto-discovered or config whitelist)
    │   ├── Extract categorical properties (config whitelist, integer-encoded)
    │   ├── Pivot long → wide per node type on executors
    │   ├── Optional z-score normalization on executors
    │   ├── Collect per-type [num_nodes, num_features] float32 arrays
    │   └── Output: Dict[node_type → FloatTensor]
    │
    └── Step 4: Assemble HeteroData (on driver)
        ├── Only compact tensors on driver — no URI strings
        ├── Attach feature tensors and node counts per type
        ├── Attach edge_index tensors per (src, rel, dst) type
        └── Output: HeteroData ready for torch.save() and GNN training
```

### Scaling Characteristics

The PyG builder is designed to handle the full enriched graph without collecting URI strings to the driver:

| Component | Where it runs | Memory model |
|-----------|--------------|--------------|
| Node ID assignment | Spark executors | URI → int mapping stays on executors via Window functions |
| Edge resolution | Spark executors | Double-join resolves URIs to ints on executors |
| Feature extraction | Spark executors | Pivot and normalization on executors |
| Edge index collection | Driver | Per-edge-type [2, N] int64 — ~16 bytes/edge |
| Feature collection | Driver | Per-node-type [N, F] float32 — ~4 bytes/cell |
| HeteroData assembly | Driver | Only compact tensors, no strings |

**Per-type feature isolation**: HeteroData stores separate feature tensors per node type. A CPI Index node carries ~5-10 CPI-specific features (`indexValue`, `relativeImportance`, etc.), not all 200+ properties from every ontology. This keeps per-type tensors compact:

| Node type example | Typical nodes | Typical features | Memory |
|-------------------|--------------|-----------------|--------|
| cpi_Index | ~50K | 5-10 | ~2 MB |
| market_PriceObservation | ~500K-1M | 10-15 | ~40-60 MB |
| market_options_OptionQuote | ~1-2M | 12-18 | ~80-140 MB |
| jolts_JobOpeningsLevel | ~10K | 3-5 | ~200 KB |
| sec_filings_Form4 | ~50K | 5-8 | ~2 MB |

**Total driver memory** for PyG object: typically 2-8 GB for 30-50M triples. Fits comfortably on Glue G.2X (32 GB) or G.4X (64 GB) workers.

### PyG Configuration

The PyG builder accepts an optional configuration dict:

```json
{
    "node_types": ["cpi_Index", "ppi_MonthlyChange", "market_PriceObservation"],
    "edge_types": ["bls_enrichment_precedes", "bls_enrichment_correlatesWith"],
    "feature_config": {
        "numeric_properties": ["indexValue", "changeValue", "observedPrice"],
        "categorical_properties": ["hasCategory", "hasIndustry"],
        "normalize": true
    },
    "include_temporal_nodes": true,
    "include_sector_nodes": true
}
```

| Config key | Default | Description |
|-----------|---------|-------------|
| `node_types` | All rdf:type classes | Whitelist of PyG node type names to include |
| `edge_types` | All entity-to-entity predicates | Whitelist of relation names to include |
| `feature_config.numeric_properties` | Auto-discovered (all parseable floats) | Whitelist of numeric property local names |
| `feature_config.categorical_properties` | None (opt-in only) | Whitelist of categorical property local names |
| `feature_config.normalize` | `false` | Z-score normalize numeric features |
| `include_temporal_nodes` | `true` | Include Month/Year/Quarter node types |
| `include_sector_nodes` | `true` | Include EconomicSector node types |

When config is empty, sensible defaults are inferred from the data.

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
triples_df (enriched) → Parquet + PyG HeteroData
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

This enriched structure enables GNNs to learn:
- **Temporal Patterns**: How indicators evolve and correlate over time across 100+ sources
- **Cross-Domain Relationships**: How economic, financial, employment, and environmental factors interact
- **Sector Dynamics**: How sector-wide shocks propagate across different data types
- **Lead-Lag Relationships**: Which indicators predict changes in others
- **Geographic Effects**: How regional factors affect economic and market outcomes
- **Company-Specific Patterns**: How company fundamentals relate to market performance
- **Intraday Dynamics**: How market prices and options evolve within trading sessions

## Data Sources

The pipeline ingests RDF data from multiple heterogeneous sources:

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
│   │   └── feature_extractor.py            # Extracts numeric/categorical features per node type
│   └── utils/
│       ├── __init__.py
│       └── rdf_utils.py                    # Namespace constants & URI helpers
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
| `rdf_utils.py` | Namespace constants, URI string helpers | No (pure Python) |
| `patterns.py` / `correlations.py` / `measurements.py` | Configuration dictionaries (sector keywords, correlation definitions) | No (pure Python) |
| `pipeline.py` | Orchestrates enrichment steps, manages triples DataFrame | Yes |
| `temporal_unifier.py` | Produces unified month/year/quarter triples | Yes |
| `bls_linker.py`, `sec_linker.py`, `market_linker.py`, `noaa_linker.py` | Produce intra-source enrichment triples | Yes |
| `cross_source_linker.py` | Produces cross-source enrichment triples | Yes |
| `ontology_mapper.py` | Produces equivalence mapping triples | Yes |
| `constructor.py` | Orchestrates PyG HeteroData construction from triples DataFrame | Yes (orchestration) |
| `node_mapper.py` | Discovers node types, assigns per-type integer IDs via Spark Window functions | Yes (heavy) |
| `edge_mapper.py` | Double-joins triples with node IDs on executors, collects edge index tensors | Yes (heavy) |
| `feature_extractor.py` | Extracts and pivots numeric/categorical features per node type on executors | Yes (heavy) |

### Scalability

The pipeline is designed to handle:
- **100+ ontologies** with different schemas and vocabularies
- **30-50M triples per month** with intraday market snapshots
- **Heterogeneous data types** (prices, rates, levels, changes, categorical)
- **Multiple temporal granularities** (intraday, daily, weekly, monthly, quarterly)
- **Dynamic schema evolution** as new data sources are added
- **Horizontal scaling** by adding Glue DPUs — enrichment and PyG construction work distributes automatically
- **Bounded driver memory** — PyG construction collects only compact integer/float tensors, not URI strings. Per-type feature isolation keeps tensors compact even with 100+ ontologies.