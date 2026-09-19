# Query Tables

The query tables are the part of a run meant for looking things up: six Parquet
tables and a triple store for each day, holding the enriched data's nodes,
edges and values under their names and URIs. SQL, SPARQL and an LLM's retrieval
step can all read them. [Questions the tables answer](questions.md) shows what
they answer, and [Using the tables with an LLM](llm.md) shows retrieval.

They are not the `.pt`. A run's `hetero_data_<variant>.pt` is a PyTorch
Geometric graph for training a graph neural network, built over every source
except NOAA weather: numbers without names, in a 42 GB pickle that is deleted
with its run after 21 days. The tables cover every source, weather included.
[The tables and the `.pt`](#the-tables-and-the-pt) lists every difference.

They are written by the enriching leg of a run
([`spark_jobs/graph/tables.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/spark_jobs/graph/tables.py)),
on by default, and switched off with `--enable_query_tables false`.

| Artifact | Shape | Measured size/day |
|---|---|---|
| `snapshots/` | market quotes, pivoted wide — one row per snapshot | 1.31 GB |
| `edges/` | `(src_type, src_id, relation, dst_type, dst_id)` | 0.67 GB |
| `graph/` | the non-snapshot subgraph as a triple store | 0.19 GB |
| `nodes/` | `(node_type, node_id, uri)` | 0.13 GB |
| `facts/` | every literal a snapshot does not carry, long format | 0.04 GB |
| `entities/` | one row per text-bearing node | ~0.01 GB |
| `edge_types/` | one row per edge type | <0.01 GB |

## A day at a time, kept for a year

Every table is split into days. A run writes one `day=YYYY-MM-DD` partition of
each table, for the day its data was cut from, and each day is kept for 365
days. A [scheduled run](../operations/running-a-job.md#scheduled-runs) adds a
day every day, so the tables build up to a year of days, and a query can read
one day or all of them.

Three rules on this page mention a day. None of them limits a query to one:

- **Ids join within a day.** `node_id` is renumbered every day, so an `edges/`
  row joins to `nodes/` on the same `day`. Across days, follow an entity by its
  `uri`, which does not change. See [Day-scoped node ids](#day-scoped-node-ids).
- **A query across days sees a node once per day.** Every day is written whole,
  so take each URI's newest day. See
  [Written every day, deduplicated on read](#written-every-day-deduplicated-on-read).
- **`graph/` is one store per day.** A SPARQL query reads one day's store. A
  question across days reads the Parquet tables, or asks each day's store in
  turn. See [`graph/`](#graph).

## Where they are published

Not inside the run folder. Runs expire at 21 days and each day of tables is kept
a year, and S3 applies the **shortest** expiration where two prefix rules
overlap — so a table under the run's prefix would be deleted with the run
whatever a second rule said.

```
<PYG_TABLES_ROOT>/<dataset>/
├── _days/
│   └── 2026-09-10.json        the day's completion marker, written last
├── nodes/day=2026-09-10/
├── edges/day=2026-09-10/
├── edge_types/day=2026-09-10/
├── facts/day=2026-09-10/
├── entities/day=2026-09-10/
├── snapshots/day=2026-09-10/
└── graph/day=2026-09-10/
```

`bin/publish_run.py` sends them there, with its own sync and its own guard;
see [Publishing a finished run](../operations/testing.md#publishing-a-finished-run).
A day that already has its marker is refused: the destination takes writes and
listings but not deletes, so a rewritten day whose part-file count went down
would leave the old parts behind and a reader would see their rows twice. A day
whose publish was interrupted has no marker, and running the same command again
finishes it, including when the run folder itself had already gone up.

The run's own `index.json` names the tables and their root, so a consumer
holding a run can find them.

## Reading them

Any engine that reads Hive-partitioned Parquet can query them: DuckDB, Spark,
Athena or pyarrow. Each `day=` directory becomes a `day` column. The SQL on this
page runs as written once each table is a view over all its days, as in DuckDB:

```sql
CREATE VIEW nodes AS FROM read_parquet('<root>/nodes/*/*.parquet', hive_partitioning = true);
CREATE VIEW edges AS FROM read_parquet('<root>/edges/*/*.parquet', hive_partitioning = true);
CREATE VIEW facts AS FROM read_parquet('<root>/facts/*/*.parquet', hive_partitioning = true);
```

`<root>` is `<PYG_TABLES_ROOT>/<dataset>`, on S3 or copied to local disk. On S3,
DuckDB also needs its `httpfs` extension and credentials, which
[Using the tables with an LLM](llm.md#a-worked-example) sets up.

- **Read only days that have a marker.** A day without `_days/<day>.json` is
  still being published, and its parts may be incomplete.
- **Reading needs its own access.** The identity that publishes the tables
  cannot read them back.
- **`graph/` is not Parquet.** It is a store directory: copy one day of it to
  local disk and open it with pyoxigraph, as [`graph/`](#graph) shows.

## Day-scoped node ids

**Edges from day D may only be joined to `nodes/` from day D.** That limits a
join, not a query: a query over many days joins each day's `edges/` to that
day's `nodes/`, as the `day` conditions below do, and follows an entity from one
day to the next by its `uri`.

`node_id` is `row_number()` over a uri-ordered window within a node type, so a
URI's id changes whenever the node set changes — which is every day. An edge
row carries ids and not URIs, because URIs would multiply the table several
times over, and that is the price.

```sql
SELECT e.relation, n.uri AS source, m.uri AS target
FROM   edges e
JOIN   nodes n ON n.day = e.day AND n.node_type = e.src_type AND n.node_id = e.src_id
JOIN   nodes m ON m.day = e.day AND m.node_type = e.dst_type AND m.node_id = e.dst_id
WHERE  e.day = DATE '2026-09-10'
```

Without the `WHERE` line, the same query reads every day.

## Written every day, deduplicated on read

There is no upsert. Merging into a published table means reading it back, and
the builder identity has no read on the published prefix. Every day is written
whole, and a consumer takes each URI's rows from the newest day that has it:

```sql
SELECT * FROM (
  SELECT *, RANK() OVER (PARTITION BY uri ORDER BY day DESC) AS rn
  FROM   facts
) WHERE rn = 1
```

`RANK`, not `ROW_NUMBER`: a node has a row per value in `facts/`, and
`ROW_NUMBER` would keep one of them and drop the rest.

This is one rule rather than a "stable versus daily" split because no split is
right for every source. Measured churn between two days: BLS repeats 99.9% of
its nodes, SEC 5.4%, NOAA none at all. BLS costs 21 copies of ~97K rows a month,
which is nothing.

## The tables

### `nodes/`

`(node_type, node_id, uri)`, sorted by `(node_type, node_id)` so a reader after
one type skips the rest. It covers every node the sources carried, NOAA weather
included.

### `edges/`

`(src_type, src_id, relation, dst_type, dst_id)`, sorted by edge type. Every
triple whose subject and object are both nodes. A triple pointing at a literal
is a fact; a triple pointing at a URI nothing typed is neither, and appears in
no table.

### `edge_types/`

`(src_type, relation, dst_type, count, predicate_uri, origin, relation_group)`,
one row per edge type — 838 on 2026-09-09.

This is what keeps `edges/` readable once its run has expired. A relation name
alone says neither which predicate it came from nor whether the link was
observed in a source or inferred by this pipeline, and `graph_schema.json`,
which does, is deleted with its run after 21 days.

`origin` is keyed by the **full** edge type rather than by the relation name,
because it depends on the endpoints as well as the predicate: the same relation
is `raw` from a source-typed subject and `enrichment` from one this pipeline
minted. Its three values are `raw`, `enrichment` and `unification`.

### `facts/`

`(node_type, uri, predicate, predicate_name, value, is_numeric)` — every literal
a node other than a market snapshot carries, one row per value.

Long, not wide, because wide would mean about 150 tables: the median type here
holds two literal predicates, the widest (`filings_SECFiling`) holds 21,
and 119 of 155 types hold under a thousand nodes. A new source appears in this
table with no code change.

`predicate` is the full URI, which joins to `ontology_schema.json`;
`predicate_name` is the short name a query is written against. `is_numeric` is
per value: whether that value parses as a finite number.

### `entities/`

`(node_type, uri, text)` — one row per node that carries text, its text-bearing
values joined in predicate order.

What comes out is the text that exists. On 2026-09-09 that is 12,607 nodes
across 49 node types, mostly names and labels; of the 990 nodes with 300 or more
characters, 901 are weather alert descriptions. Rendering a node and its
neighbourhood into a real sentence is what would make a vector index over this
useful, and it is separate work. This repository writes the table and embeds
nothing; [Using the tables with an LLM](llm.md#embedding-entities) covers
embedding it.

### `snapshots/`

Market quotes, pivoted wide: `(node_type, uri, <one column per property>)`,
sorted by the underlying ticker.

Wide earns its keep here and nowhere else — two node types, 10.2M rows a day, 45
stable columns, 77.8% populated (measured 2026-09-17). Numeric properties are
`double` columns and the rest are strings, decided per property: a property is
numeric when more than half of its values parse as numbers. A snapshot missing a
property gets a null, not a dropped row.

**Quotes only, not everything in a market namespace.** Market enrichment mints
a handful of hub nodes — the GICS sector nodes, which carry a
`relationConfidence`, and the moneyness classes. They are neither big nor a
time series: eight sector nodes against 10,174,755 quotes on 2026-09-17. They
go where every other node goes, so their values are in `facts/` and their
triples in `graph/`. Pivoting them in here put a
`market_enrichment_relationConfidence` column on the table that was null on
every quote row.

The sort is what makes a ticker query cheap. There are 525 underlying tickers
and 19 snapshots a day, a median 13,224 rows per ticker, so a single-ticker
filter wants 0.14% of a day. Parquet prunes row groups on min/max statistics:
sorted, a 30-day ticker query reads a few hundred MB; unsorted, every ticker
appears in every row group and the same query reads all 39 GB.

### `graph/`

The non-snapshot subgraph as a [pyoxigraph](https://pyoxigraph.readthedocs.io/)
store, one per day — a directory, not a Parquet file.

`edges/` and `nodes/` answer a one-hop question well and a longer one
awkwardly. Every extra hop is one more join of `edges/` to `nodes/` on the same
`day`, and a path whose length is not known in advance needs a recursive query.
SPARQL states the whole path as one pattern instead. Speed is not the problem. On
2026-09-09 the two-hop traversal from weather alerts to the regions they affect,
and on to the BLS measurements for those regions, returned its 38,352 pairs in
0.05 s from a local copy of `edges/`, and in 10 ms from the store, which held
1,413,546 triples in 185 MB. The store also keeps what no table can: a triple
whose object is a URI nothing typed, such as a weather alert's severity, is in
`graph/` and nowhere else.

```python
import pyoxigraph

store = pyoxigraph.Store.read_only("graph/day=2026-09-10")
for row in store.query("""
    SELECT ?measurement WHERE {
      ?alert       <https://jefflevesque.com/ontology/bls/affectsRegion> ?region .
      ?measurement <https://jefflevesque.com/ontology/bls/hasRegion>     ?region
    }
"""):
    print(row["measurement"].value)
```

**The store is one day; the Parquet tables are every day.** A day's store is
about 185 MB, so 30 days of stores would be about 5.6 GB and a year about
68 GB. The store therefore supplements the tables rather than replacing them:
open one day's store for a multi-hop question about that day, and read `edges/`
and `nodes/` for anything spanning more.

**No snapshot enters it.** At the store's rate, about 131 bytes a triple, the
quotes' ~415M triples a day would be about 54 GB, and they are a time series of
numbers carrying one edge per snapshot — not a shape a triple store earns
anything on.

Market's *hub* nodes do, and that is the point of them: `EquitySector`
`relatedToEconomicSector` `EconomicSector` is how market reaches the economic
sectors, and a multi-hop question about that route is the kind of question the
store exists for. A handful of hubs costs the store nothing.

Market *terms* appear as well, which is different again. The statements about the
vocabulary — the derived `rdfs:subClassOf` hierarchy, the observed domains and
ranges, the provenance markers — have predicate and class URIs as their subjects
rather than entities, so they describe no source's data and belong to none of
them. They are the store's schema, a few hundred triples, and most of what makes
a SPARQL query over it worth writing.

The store holds the *enriched* frame's view of the data: literals are the plain
strings the loader converted them to, with their source datatypes already folded
in. It is the same view every other artifact in this pipeline is built from, and
not a re-serialization of the original RDF.

## The tables and the `.pt`

Both are built from the same enriched triples. Everything about how they differ
is here, so the sections above describe the tables alone.

- **Why the tables exist.** The rest of a published run is no easier to query.
  With its node index and 200 Parquet parts of triples a run comes to about
  95 GB, none of it filterable by ticker, by date or by meaning, and the graph's
  70.8 million edges are recoverable only by unpickling the `.pt`. The tables
  are the same data in shapes something can query, about 2.4 GB a day against
  94.9 GB for the run.
- **Coverage.** A run's `.pt`, and its `node_index/`, hold only the node types
  the run builds the `.pt` over, and runs leave NOAA weather out. The tables
  ignore that setting and cover every source. NOAA is 0.05% of the nodes on
  2026-09-09, shares nothing between days and has no edge to market, so it earns
  little in a graph neural network and still answers state-by-month questions
  in a table.
- **`nodes/` against `node_index/`.** The same three columns, over every source
  rather than over the node types the `.pt` holds.
- **What counts as a number.** `is_numeric` in `facts/` uses the same test the
  feature extractor types the `.pt`'s numeric segment by, and `snapshots/` types
  its columns by the same majority rule, so the tables and the `.pt` agree.
- **Feature vectors stay in the `.pt`.** The dense node feature matrix is
  38.16 GB for the dominant type, of which 183 of 1024 dimensions vary per row.
- **Retention.** A run, with its `.pt`, is kept 21 days; each day of tables is
  kept a year.

## What is deliberately not here

- **A snapshot's literals in `facts/`, and snapshot subjects in `graph/`.**
  They are in `snapshots/` and in `edges/`. Market's sector and moneyness hub
  nodes are not snapshots and are in `facts/` and `graph/` like anything else.
- **A triple whose object is a URI nothing typed.** It is no edge, and putting a
  dangling pointer in a `value` column would have consumers reading it as a name.
- **Anything that answers a question the data cannot.** See
  [Questions the tables answer](questions.md).
