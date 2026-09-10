# PyTorch Geometric Knowledge Graph Builder

> GPU-accelerated Apache Spark pipeline for constructing PyTorch Geometric heterogeneous graphs from enriched RDF knowledge graphs

<div>
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-labelledby="archTitle archDesc"
     style="width:100%;height:auto;display:block;margin:1.2rem auto">
  <title id="archTitle">Pipeline architecture</title>
  <desc id="archDesc">Turtle from four sources is parsed into one triples
    DataFrame, enriched, and built into a PyTorch Geometric HeteroData graph.
    All three stages run on Spark executors with the RAPIDS Accelerator. The
    outputs are enriched Parquet, a .pt graph, and six metadata JSON
    files.</desc>

  <style>
    .lbl   { font: 600 13px var(--md-text-font-family, system-ui, sans-serif);
             fill: var(--md-default-fg-color, #1a1a1a); }
    .sub   { font: 400 10.5px var(--md-text-font-family, system-ui, sans-serif);
             fill: var(--md-default-fg-color--light, #5a5a5a); }
    .cap   { font: 600 11px var(--md-text-font-family, system-ui, sans-serif);
             fill: var(--md-default-fg-color--light, #5a5a5a);
             letter-spacing: .07em; }
    .box   { fill: var(--md-code-bg-color, #f4f4f5);
             stroke: var(--md-default-fg-color--lightest, #d8d8dc); }
    .stage { fill: var(--md-code-bg-color, #f4f4f5);
             stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.5; }
    .band  { fill: none; stroke: var(--md-default-fg-color--lightest, #d8d8dc);
             stroke-dasharray: 4 4; }
    .flow  { stroke: var(--md-primary-fg-color, #3f51b5); stroke-width: 1.6;
             fill: none; }
    .tip   { fill: var(--md-primary-fg-color, #3f51b5); }
  </style>

  <!-- sources -->
  <text class="cap" x="8" y="26">SOURCES</text>
  <rect class="box" x="8" y="38" width="140" height="42" rx="5"/>
  <text class="lbl" x="20" y="58">BLS</text>
  <text class="sub" x="20" y="72">10 categories</text>
  <rect class="box" x="8" y="90" width="140" height="42" rx="5"/>
  <text class="lbl" x="20" y="110">Market</text>
  <text class="sub" x="20" y="124">intraday snapshots</text>
  <rect class="box" x="8" y="142" width="140" height="42" rx="5"/>
  <text class="lbl" x="20" y="162">NOAA</text>
  <text class="sub" x="20" y="176">weather alerts</text>
  <rect class="box" x="8" y="194" width="140" height="42" rx="5"/>
  <text class="lbl" x="20" y="214">SEC</text>
  <text class="sub" x="20" y="228">filings</text>
  <text class="sub" x="8" y="258">Turtle in Parquet</text>
  <text class="sub" x="8" y="272">100+ ontologies</text>

  <!-- sources into the cluster -->
  <path class="flow" d="M148 137 H176"/>
  <path class="tip" d="M176 132 l10 5 -10 5 z"/>

  <!-- the cluster band -->
  <rect class="band" x="190" y="18" width="378" height="264" rx="8"/>
  <text class="cap" x="204" y="40">SPARK EXECUTORS &#183; RAPIDS (GPU)</text>

  <rect class="stage" x="204" y="56" width="350" height="52" rx="5"/>
  <text class="lbl" x="220" y="78">Parse</text>
  <text class="sub" x="220" y="94">Turtle &#8594; one triples DataFrame</text>

  <path class="flow" d="M379 108 V126"/>
  <path class="tip" d="M374 126 l5 10 5 -10 z"/>

  <rect class="stage" x="204" y="138" width="350" height="52" rx="5"/>
  <text class="lbl" x="220" y="160">Enrich</text>
  <text class="sub" x="220" y="176">temporal, intra-source, cross-source</text>

  <path class="flow" d="M379 190 V208"/>
  <path class="tip" d="M374 208 l5 10 5 -10 z"/>

  <rect class="stage" x="204" y="220" width="350" height="52" rx="5"/>
  <text class="lbl" x="220" y="242">Build PyG</text>
  <text class="sub" x="220" y="258">node IDs, edges, feature vectors</text>

  <!-- cluster into outputs -->
  <path class="flow" d="M568 150 H596"/>
  <path class="tip" d="M596 145 l10 5 -10 5 z"/>

  <!-- outputs -->
  <text class="cap" x="612" y="26">OUTPUTS</text>
  <rect class="box" x="612" y="76" width="140" height="42" rx="5"/>
  <text class="lbl" x="624" y="96">Enriched Parquet</text>
  <text class="sub" x="624" y="110">reusable, skips enrichment</text>
  <rect class="box" x="612" y="128" width="140" height="42" rx="5"/>
  <text class="lbl" x="624" y="148">HeteroData .pt</text>
  <text class="sub" x="624" y="162">GNN ready</text>
  <rect class="box" x="612" y="180" width="140" height="42" rx="5"/>
  <text class="lbl" x="624" y="200">Metadata JSON</text>
  <text class="sub" x="624" y="214">six files per build</text>
</svg>
</div>

One day of intraday market snapshots loads **322.7M triples** and enriches to
**421.4M**. Market is 99.5% of that; BLS 1.3M, SEC 198K, NOAA 143K. The job runs
in three modes — the full pipeline, enrichment only (which stops at the reusable
Parquet), and PyG only (which starts from it) — so experimenting with graph
structure never pays for enrichment twice.

<div class="grid cards" markdown>

-   :material-vector-triangle:{ .lg .middle } **Design**

    ---

    How the graph is shaped: the universal node vector, the selective edge
    vector, how construction and enrichment work.

    [:octicons-arrow-right-24: Node feature vectors](design/node-features.md)

-   :material-file-document-outline:{ .lg .middle } **Reference**

    ---

    What a build produces: the six metadata files, the integrity record beside
    them, the node index, the data sources, and the module map.

    [:octicons-arrow-right-24: Metadata files](reference/outputs.md)

-   :material-console:{ .lg .middle } **Operations**

    ---

    Running it: job parameters, cluster prerequisites, sizing a large run,
    driver memory, and the test suites.

    [:octicons-arrow-right-24: Running a job](operations/running-a-job.md)

</div>

--8<-- "README.md:overview"

## Architecture

The diagram above is the whole job in one picture. The rest of this section is
what the pieces are.

### Core Representation

All RDF data is parsed from N-Triples files into a single **triples DataFrame** that serves as the universal graph representation throughout the pipeline:

Schema: `(subject: string, predicate: string, object: string)`.

| subject | predicate | object |
|---|---|---|
| `cpi:Food_Nov2024_Index` | `rdf:type` | `cpi:Index` |
| `cpi:Food_Nov2024_Index` | `cpi:indexValue` | `295.8` |
| `cpi:Food_Nov2024_Index` | `cpi:hasMonth` | `cpi:November` |
| `cpi:Food_Nov2024_Index` | `cpi:hasCategory` | `cpi:Food_Entity` |
| `market:AAPL_20241115T143000Z` | `rdf:type` | `market:EquitySnap` |
| `market:AAPL_20241115T143000Z` | `market:lastPrice` | `191.45` |
| `market:AAPL_20241115T143000Z` | `market:symbol` | `AAPL` |
| `market:AAPL_20241115T143000Z` | `market:captureTime` | `2024-11-15T14:30Z` |

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

**Terms this project invents** all live under one base we control — `jefflevesque.com`, the author's own domain — sub-pathed by concern:

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
