# Knowledge Graph Enrichment

The enrichment pipeline creates a unified knowledge graph by establishing relationships at two levels across **100+ data sources and ontologies**. Each enrichment step is a PySpark transformation that reads the triples DataFrame, computes new relationship triples, and unions them back.

## Enrichment Pipeline Flow

Each enricher below reads `triples_df` and only adds to it — nothing is
rewritten and nothing is removed:

- **BLS Intra-Source Enricher**
    - Temporal sequences (precedes links)
    - Sector classification (belongsToSector)
    - Cross-dataset correlations (correlatesWith)
    - Hierarchical enrichment (hasParent chains)
- **SEC Intra-Source Enricher**
    - Company unification (owl:sameAs by CIK)
    - Person unification (owl:sameAs by CIK)
    - Filing sequences (precedes by date)
    - Transaction sequences (precedes by transaction date, within a reporting owner and instrument class)
    - Sector classification (belongsToSector)
    - Violation type linking (hasViolationType)
- **Market Intra-Source Enricher**
    - Snapshot temporal sequences (precedes by captureTime)
    - Option-to-underlying equity linking (hasUnderlyingEquity)
    - Option strategy detection (straddleWith, spreadWith, strangleWith)
    - Sector classification (belongsToSector via symbol; loads GICS sectors from S3 tickers CSV with hardcoded fallback)
    - Moneyness computation (hasMoneyness: ATM/ITM/OTM)
- **NOAA Intra-Source Enricher**
    - Alert temporal sequences (precedes by sent time)
    - Geographic linking (affectsSameRegion via SAME codes)
    - Event type linking (sameEventType)
    - Severity escalation detection (escalatesTo)
- **Temporal Unifier** (cross-source)
    - Unified months/years/quarters (owl:sameAs)
- **Cross-Source Linker**
    - Sector-based linking across sources
    - Company/ticker linking (SEC ↔ Market)
    - Geographic linking (BLS ↔ NOAA)
    - Causal relationships (BLS → Market, NOAA → Market)
    - Measurement type alignment
- **Ontology Mapper (optional; --enable_ontology_mapping, default true)**
    - owl:equivalentProperty / owl:equivalentClass (one-to-one pairs only)
    - predicate folding to the unified vocabulary
    - skos:prefLabel normalization
    - rdfs:subClassOf ← curated CLASS_MAPPINGS + class naming
    - rdfs:subPropertyOf ← curated PROPERTY_MAPPINGS shared targets
    - rdfs:domain/range ← observed usage + declared XSD datatypes
    - prov:derivedBy ← how each of the above was arrived at

The result is `triples_df` (enriched), written as Parquet locally, alongside
the PyG `HeteroData` `.pt` and the six metadata JSON files (local, and
mirrored to S3 when an archive is configured).

## Intra-Source Linking

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

## Enrichment as PySpark Operations

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

## Cross-Source Linking

Discovers and creates relationships across different data source families:

> **Set `--market_sector_definitions_bucket` / `--market_sector_definitions_key` for real runs.**
> They point at the S&P 500 constituents CSV, and three links read it:
>
> | link | with the CSV | without |
> |---|---|---|
> | quote → company (ticker resolved to the SEC's company ID) | all 503 constituents | **none** |
> | snapshot → GICS sector | 503 tickers, 11 sectors | ~130 tickers from a built-in list |
> | company ↔ company, same sub-industry | 127 groups | **none** |
>
> Both default to empty, so a run that does not set them builds a graph missing
> those links with nothing logged as wrong. There is deliberately **no** built-in
> fallback for the company-ID map: guessing a regulator's ID would silently merge
> two unrelated companies into one node, which is worse than an absent link.
>
> A second limit is upstream and not fixable here: a filing reaches an economic
> sector through `filings:hasSic`, and the SEC leaves that code blank on most
> filings — 15 of 40 on the committed fixtures. So roughly four companies in ten
> get an industry, and the ceiling is EDGAR's classification coverage rather than
> this join's.

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

## Example Intra-Source Patterns

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

## Enrichment Statistics (measured, one four-source day)

| Measure | Triples | Notes |
|---------|---------|-------|
| Loaded | 322.7M | Market 99.5%; BLS 1.3M, SEC 198K, NOAA 143K |
| Enriched (final) | 421.4M | after every phase and its `dropDuplicates` |
| **Net change** | **+98.7M** | `total_enrichment` in the job manifest |

Market is 99.5% of what loads, so nearly all of the net change comes from the
market intra-source enricher — snapshot `precedes` chains per symbol,
`hasUnderlyingEquity`, strategy detection, moneyness, and sector. The other three
sources together are the remaining 0.5%.

There is no per-phase breakdown here because none has been measured at this
scale. `EnrichmentPipeline` computes one on every run — `enrichment_added` sums
what each phase reported — so the numbers reach the manifest for any given job;
they have simply never been recorded for a run this size.

The final count is not loaded + added: each enrichment phase runs
`dropDuplicates` over the whole frame, so duplicates the source already contained
are collapsed at the same time. Real sources carry them — one day of SEC filings
measured ~8.7% duplicate triples — so a run can legitimately finish with *fewer*
triples than it started with while enrichment added millions. The manifest
reports `enrichment_added` and `duplicates_removed` separately for that reason;
`total_enrichment` is their sum (the net change), not a measure of enrichment on
its own.

## Benefits for GNN Training

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

