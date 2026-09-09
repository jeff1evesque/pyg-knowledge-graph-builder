# Derived Edge Feature Vectors

## The Problem With Featureless Edges

In a basic heterogeneous graph, edges carry only their type label (e.g., `("cpi_Index", "bls_enrichment_precedes", "cpi_Index")`). The GNN learns a single weight matrix per edge type, applied identically to all edges of that type. This has limitations:

- **No per-instance variation**: A `precedes` edge spanning 1 month is treated identically to one spanning 12 months
- **No endpoint contrast**: An option-stock edge where the option is deep in-the-money looks the same as one that is far out-of-the-money
- **No cross-source signal**: A `correlatesWith` edge between two CPI categories is indistinguishable from one linking CPI to PPI
- **Wasted information**: Endpoint node properties that could inform message passing are ignored until the GNN aggregates them — edge features let the GNN modulate messages *before* aggregation

## The Solution: Selective Derived Edge Feature Vectors

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

### Which Edge Types Get Features

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

### Segment 1: Temporal Signals (37.5% of edge_vector_dim)

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

### Segment 2: Numeric Contrast (37.5% of edge_vector_dim)

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

### Segment 3: Relational Context (25% of edge_vector_dim)

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

## Proportional Dimension Scaling via EdgeVectorLayout

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

## Why Edge Features Require Zero Enrichment Changes

Edge features are derived entirely from properties already present in the enriched triples:

1. **EdgeMapper** resolves each edge's subject and object URIs to integer node IDs via a double-join against the node ID table. This produces a cached resolved edges DataFrame with `(src_type, src_id, relation, dst_type, dst_id)`.

2. **EdgeFeatureExtractor** receives this cached DataFrame directly from the constructor — **no double-join replay**. It joins endpoint node IDs against the literal triples to access numeric properties and labels, all on Spark executors.

3. Temporal signals are derived from month/year properties that nodes already have (e.g., `cpi:hasMonth`, `cpi:hasYear`). Numeric contrast is derived from literal properties (e.g., `cpi:indexValue`, `market:strikePrice`). Label similarity uses existing `rdfs:label` values.

No new triples, no new predicates, no changes to any enrichment module.

## All Edge Encoding Runs on Spark Executors

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

## Why This Is Better for GNNs

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

