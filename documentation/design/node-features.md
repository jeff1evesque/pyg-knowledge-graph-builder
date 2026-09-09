# Ontology-Aware Node Feature Vectors

## The Problem With Flat Literal Vectors

A naive approach encodes each node as a flat bag of its literal property values — `[indexValue, percentChange, relativeImportance, ...]`. This has fundamental limitations:

- **No structural context**: The vector for a `cpi:Index` node knows `indexValue = 295.8` but encodes nothing about what that node *is* in the ontology
- **Semantic collisions**: Two nodes from different ontologies sharing a property name (e.g., `hasValue`) get the same feature column despite completely different meanings
- **Ambiguous zeros**: A zero in a column is indistinguishable between "missing value" and "property doesn't apply to this type"
- **Per-type isolation**: Each node type has a different feature width, requiring type-specific linear layers in the GNN and preventing cross-type weight sharing

## The Solution: Ontology-Structure + Literal Hybrid Vector

Every node gets a fixed-width vector (default 1024-d) with three segments encoding progressively more specific information. All segment boundaries are computed proportionally by `VectorLayout`, so the structure scales to any `vector_dim`:

<div>
<svg viewBox="0 0 760 84" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Node feature vector, 1024 dims"
     style="width:100%;height:auto;display:block;margin:1rem auto">
  <style>
    .bt  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); letter-spacing: .06em; }
    .bn  { font: 600 13px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color, #1a1a1a); }
    .bd  { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); }
    .bc  { fill: var(--md-code-bg-color, #f4f4f5);
           stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.4; }
  </style>
  <text class="bt" x="0" y="14">NODE FEATURE VECTOR, 1024 DIMS</text>
  <rect class="bc" x="3.0" y="26" width="184.0" height="54" rx="4"/>
  <text class="bn" x="15.0" y="49">Ontology structure</text>
  <text class="bd" x="15.0" y="66">256 dims &#183; 25%</text>
  <rect class="bc" x="193.0" y="26" width="279.0" height="54" rx="4"/>
  <text class="bn" x="205.0" y="49">Property schema</text>
  <text class="bd" x="205.0" y="66">384 dims &#183; 37.5%</text>
  <rect class="bc" x="478.0" y="26" width="279.0" height="54" rx="4"/>
  <text class="bn" x="490.0" y="49">Literal values</text>
  <text class="bd" x="490.0" y="66">384 dims &#183; 37.5%</text>
</svg>
</div>

- **Ontology structure** — class hierarchy and type identity
- **Property schema** — which properties are defined and present
- **Literal values** — numeric, and categorical after encoding

### Segment 1: Ontology Structure (25% of vector_dim)

Encodes **what the node is** in the ontology hierarchy — its class, its superclasses, and its ontology membership. Gives the GNN a structural fingerprint consistent across all nodes of the same type.

<div>
<svg viewBox="0 0 760 84" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Segment 1 &#183; ontology structure, 256 dims"
     style="width:100%;height:auto;display:block;margin:1rem auto">
  <style>
    .bt  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); letter-spacing: .06em; }
    .bn  { font: 600 13px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color, #1a1a1a); }
    .bd  { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); }
    .bc  { fill: var(--md-code-bg-color, #f4f4f5);
           stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.4; }
  </style>
  <text class="bt" x="0" y="14">SEGMENT 1 &#183; ONTOLOGY STRUCTURE, 256 DIMS</text>
  <rect class="bc" x="3.0" y="26" width="469.0" height="54" rx="4"/>
  <text class="bn" x="15.0" y="49">Class identity</text>
  <text class="bd" x="15.0" y="66">160 dims &#183; 62.5%</text>
  <rect class="bc" x="478.0" y="26" width="136.5" height="54" rx="4"/>
  <text class="bn" x="490.0" y="49">Class hierarchy</text>
  <text class="bd" x="490.0" y="66">48 dims &#183; 18.8%</text>
  <rect class="bc" x="620.5" y="26" width="136.5" height="54" rx="4"/>
  <text class="bn" x="632.5" y="49">Ontology / source</text>
  <text class="bd" x="632.5" y="66">48 dims &#183; 18.8%</text>
</svg>
</div>

- **Class identity** — multi-hot hash of the node's `rdf:type` URIs
- **Class hierarchy** — the `rdfs:subClassOf` chain, depth-weighted
- **Ontology / source** — which namespace the node comes from, multi-hot

- **Class Identity**: Each `rdf:type` URI is hashed into 4 deterministic slots. Nodes of the same type share identical bits.
- **Class Hierarchy**: `rdfs:subClassOf` chains are traversed (transitive closure up to depth 10). Superclass URIs are hashed with depth-weighted values (direct superclass = 1.0, grandparent = 0.5, etc.). Nodes sharing a superclass share bits in this segment.
- **Ontology/Source Membership**: Multi-hot encoding of which ontology namespace(s) the node belongs to, derived from both the type URI and the node URI itself. Uses the canonical namespace registry from `rdf_utils.py`.

### Segment 2: Property Schema (37.5% of vector_dim)

Encodes **which ontology-defined properties are present** for this node, regardless of their values. This tells the GNN about schema conformance and distinguishes "missing because not observed" from "missing because inapplicable."

<div>
<svg viewBox="0 0 760 84" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Segment 2 &#183; property schema, 384 dims"
     style="width:100%;height:auto;display:block;margin:1rem auto">
  <style>
    .bt  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); letter-spacing: .06em; }
    .bn  { font: 600 13px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color, #1a1a1a); }
    .bd  { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); }
    .bc  { fill: var(--md-code-bg-color, #f4f4f5);
           stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.4; }
  </style>
  <text class="bt" x="0" y="14">SEGMENT 2 &#183; PROPERTY SCHEMA, 384 DIMS</text>
  <rect class="bc" x="3.0" y="26" width="374.0" height="54" rx="4"/>
  <text class="bn" x="15.0" y="49">Property presence</text>
  <text class="bd" x="15.0" y="66">192 dims &#183; 50%</text>
  <rect class="bc" x="383.0" y="26" width="213.7" height="54" rx="4"/>
  <text class="bn" x="395.0" y="49">Domain / range</text>
  <text class="bd" x="395.0" y="66">111 dims &#183; 28.9%</text>
  <rect class="bc" x="602.7" y="26" width="154.3" height="54" rx="4"/>
  <text class="bn" x="614.7" y="49">Property hierarchy</text>
  <text class="bd" x="614.7" y="66">81 dims &#183; 21.1%</text>
</svg>
</div>

- **Property presence** — which properties this node has, multi-hot hashed
- **Domain / range** — the `rdfs:domain` and `rdfs:range` of those properties
- **Property hierarchy** — `rdfs:subPropertyOf` chains

- **Property Presence**: Each predicate URI the node has is hashed into 3 slots. A CPI Index node with `indexValue`, `percentChange`, `hasMonth` gets different bits than a JOLTS node with `jobOpeningsLevel`, `hasIndustry`.
- **Domain/Range Signals**: For each property this node has, its `rdfs:domain` and `rdfs:range` are hashed. This tells the GNN what types of relationships this node can participate in. No source declares either, so both are **derived** — the domain from the observed `rdf:type` of the property's subjects, the range from its objects' types and from the XSD datatype the source declared on its literals. A property used on more than one class gets neither, because `rdfs:domain` is an intersection; see [`ontology_schema.json`](../reference/outputs.md#ontology_schemajson) for the provenance and coverage this publishes.
- **Property Hierarchy**: `rdfs:subPropertyOf` relationships are hashed, connecting specific properties to their abstract parents. Also derived — from the `PROPERTY_MAPPINGS` entries whose target several properties share (nine point at `unified:measurementValue`, and a rate is not a price, so they are sub-properties rather than equivalents).

### Segment 3: Literal Values (37.5% of vector_dim)

Carries the actual numeric and categorical values in a fixed-width format with proper encoding.

<div>
<svg viewBox="0 0 760 84" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="Segment 3 &#183; literal values, 384 dims"
     style="width:100%;height:auto;display:block;margin:1rem auto">
  <style>
    .bt  { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); letter-spacing: .06em; }
    .bn  { font: 600 13px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color, #1a1a1a); }
    .bd  { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
           fill: var(--md-default-fg-color--light, #5a5a5a); }
    .bc  { fill: var(--md-code-bg-color, #f4f4f5);
           stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.4; }
  </style>
  <text class="bt" x="0" y="14">SEGMENT 3 &#183; LITERAL VALUES, 384 DIMS</text>
  <rect class="bc" x="3.0" y="26" width="502.6" height="54" rx="4"/>
  <text class="bn" x="15.0" y="49">Numeric values</text>
  <text class="bd" x="15.0" y="66">257 dims &#183; 66.9%</text>
  <rect class="bc" x="511.6" y="26" width="245.4" height="54" rx="4"/>
  <text class="bn" x="523.6" y="49">Categorical values</text>
  <text class="bd" x="523.6" y="66">127 dims &#183; 33.1%</text>
</svg>
</div>

- **Numeric values** — z-score normalised into hashed slots
- **Categorical values** — multi-hot hash encoding

- **Numeric Values**: Each numeric property's predicate URI hashes to a fixed slot. The value is z-score normalized (per-predicate stats computed in a single pass on executors) and placed at that slot. Hash collisions sum — rare with 256 dims and ~10 properties per type.
- **Categorical Values**: Multi-hot hash encoding instead of `dense_rank`. Each `(predicate, value)` pair hashes to 4 slots. No ordinal assumption.

### Numeric vs. categorical is decided per property, not per value

**A property is numeric or categorical, never both.** A predicate is numeric only when *more than* `feature_config.numeric_predicate_min_share` (default `0.5`) of its literal values parse as a number; otherwise every one of its values — including any that happen to parse — is encoded as a category label.

Classifying each *value* independently splits one property across both sub-segments whenever its labels are not uniformly shaped. SEC `hasDocumentType` is the motivating case: 315 of its 2,372 values (13.3%) are bare-digit form types — Form `4`, `144`, `3`, `425`, `497`, `487`, `25` — while the rest are hyphenated (`10-K`, `8-K`, `S-1`). Those 315 were z-scored into the numeric sub-segment as if a form number were a magnitude (mean 62.24, std 128.64), inventing a continuous ordering over labels, while the other 2,057 were correctly multi-hot encoded.

A simple majority is deliberate: it is the least presumptuous rule that fixes the above, and it lets a genuinely numeric measurement carry a minority of unparseable sentinels (`"N/A"`, `"unknown"`) without demoting the whole property out of the numeric sub-segment — those sentinels are then dropped as missing data rather than re-encoded as labels, exactly as an absent property is. The threshold is recorded in `encoding_config.json` under `numeric_values.predicate_min_numeric_share`, since it determines which sub-segment a property is encoded into.

## Proportional Dimension Scaling via VectorLayout

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

> **Changing `class_identity_dim` or `vector_dim` invalidates trained models.** Slots are `hash % dim`, so a different width re-maps every class. This is why the width is a published tuning constant in `encoding_config.json` rather than a figure recomputed on every build — one that moved whenever a class appeared would re-map every existing class for no benefit.
>
> Deriving the width per build was tried and removed. Widening `class_identity` means taking dims from `class_hierarchy` and `ontology_source`, and neither has a requirement to size against — `ontology_source` indexes a fixed 26-entry namespace table and is *expected* to collide, so "what it needs" is not a measurable quantity there. Any automatic split is therefore a guess about budgets nobody has established. The build fails instead, and the failure names the `vector_dim` that would fit along with what it costs in driver memory, so the arithmetic is not left to the reader.

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

## Why This Is Better for GNNs

!!! failure "Flat literal vectors, the old approach"

    | node type | vector | width |
    |---|---|---|
    | `cpi_Index` | `[295.8, 0.3, 2.1, 0.05, 3.0, 1.0]` | 6 dims, literals only |
    | `ppi_Index` | `[187.2, 0.1, 1.5, 0.03, 2.0, 1.0]` | 6 dims, literals only |

    - Separate tensors per type, each a different width.
    - The GNN needs type-specific linear layers.
    - No cross-type weight sharing is possible.
    - A zero is ambiguous — missing, or inapplicable? The GNN cannot tell.

!!! success "Ontology-aware vectors, the current approach"

    | node type | vector | width |
    |---|---|---|
    | `cpi_Index` | `[ontology 25%, schema 37.5%, literals 37.5%]` | 1024-d, universal |
    | `ppi_Index` | `[ontology 25%, schema 37.5%, literals 37.5%]` | 1024-d, universal |

    - The same tensor width for every node type.
    - Ontology bits are shared wherever two types share ancestry.
    - The GNN can use shared layers across all types.
    - Cross-type message passing works naturally.
    - Property presence distinguishes missing from not applicable.
    - `vector_dim` is an override, trading memory against fidelity.

## All Node Encoding Runs on Spark Executors

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

