# Query Tables

A published run is a pickle, a node index and 200 Parquet parts of triples —
about 95 GB, none of it filterable by ticker, by date or by meaning, and the
graph's 70.8 million edges recoverable only by unpickling a 42 GB blob.

The query tables are the same data in shapes something can query: six Parquet
tables and a triple store, about 2.4 GB a day against 94.9 GB for the run.
They are written by the enriching leg of a run
([`spark_jobs/graph/tables.py`](https://github.com/jeff1evesque/pyg-knowledge-graph-builder/blob/master/spark_jobs/graph/tables.py)),
on by default, and switched off with `--enable_query_tables false`.

| Artifact | Shape | Measured size/day |
|---|---|---|
| `snapshots/` | market, pivoted wide — one row per snapshot | 1.31 GB |
| `edges/` | `(src_type, src_id, relation, dst_type, dst_id)` | 0.67 GB |
| `graph/` | the non-market subgraph as a triple store | 0.19 GB |
| `nodes/` | `(node_type, node_id, uri)` | 0.13 GB |
| `facts/` | every non-market literal, long format | 0.04 GB |
| `entities/` | one row per text-bearing node | ~0.01 GB |
| `edge_types/` | one row per edge type | <0.01 GB |

## Where they are published

Not inside the run folder. Runs expire at 21 days and the tables keep a year,
and S3 applies the **shortest** expiration where two prefix rules overlap — so a
table under the run's prefix would be deleted with the run whatever a second
rule said.

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
whose publish was interrupted has no marker and resumes normally.

The run's own `index.json` names the tables and their root, so a consumer
holding a run can find them.

## Day-scoped node ids

**Edges from day D may only be joined to `nodes/` from day D.**

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

## Written every day, deduplicated on read

There is no upsert. Merging into a published table means reading it back, and
the builder identity has no read on the published prefix. Every day is written
whole, and a consumer takes the newest row per URI:

```sql
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY uri ORDER BY day DESC) AS rn
  FROM   facts
) WHERE rn = 1
```

This is one rule rather than a "stable versus daily" split because no split is
right for every source. Measured churn between two days: BLS repeats 99.9% of
its nodes, SEC 5.4%, NOAA none at all. BLS costs 21 copies of ~97K rows a month,
which is nothing.

## The tables

### `nodes/`

`(node_type, node_id, uri)`, the same three columns as the run's own
`node_index/`, sorted by `(node_type, node_id)` so a reader after one type skips
the rest.

It is not a copy of that file. The run's index covers whatever node types the
run built its `.pt` over; this covers everything the sources carried. A run can
legitimately exclude a source from its model and still have to publish it — NOAA
is 0.076% of the nodes, shares nothing between days and reaches market through
no edge at all, so it earns little in a graph neural network and still answers
state-by-month questions in a table.

### `edges/`

`(src_type, src_id, relation, dst_type, dst_id)`, sorted by edge type. Every
triple whose subject and object are both nodes. A triple pointing at a literal
is a fact; a triple pointing at a URI nothing typed is neither, and appears in
no table.

### `edge_types/`

`(src_type, relation, dst_type, count, predicate_uri, origin, relation_group)`,
one row per edge type — 837 on a production day.

This is what keeps `edges/` readable once its run has expired. A relation name
alone says neither which predicate it came from nor whether the link was
observed in a source or inferred by this pipeline, and `graph_schema.json`,
which does, is run-scoped.

`origin` is keyed by the **full** edge type rather than by the relation name,
because it depends on the endpoints as well as the predicate: the same relation
is `raw` from a source-typed subject and `enrichment` from one this pipeline
minted. Its three values are `raw`, `enrichment` and `unification`.

### `facts/`

`(node_type, uri, predicate, predicate_name, value, is_numeric)` — every literal
a non-market node carries, one row per value.

Long, not wide, because wide would mean about 150 tables: the median non-market
type holds two literal predicates, the widest (`filings_SECFiling`) holds 21,
and 119 of 155 types hold under a thousand nodes. A new source appears in this
table with no code change.

`predicate` is the full URI, which joins to `ontology_schema.json`;
`predicate_name` is the short name a query is written against.
`is_numeric` is per value and uses the same test the feature extractor types
its numeric segment by, so the table and the model agree on what a number is.

### `entities/`

`(node_type, uri, text)` — one row per node that carries text, its text-bearing
values joined in predicate order.

What comes out is the text that exists: 22,096 nodes and about 60 characters
each, mostly bare names like `INTUIT` and `FORM 4`, because only 5 of 155 node
types carry any text at all. Rendering a node and its neighbourhood into a real
sentence is what would make a vector index over this useful, and it is separate
work. This repository emits the table; the serving side owns the model.

### `snapshots/`

Market, pivoted wide: `(node_type, uri, <one column per property>)`, sorted by
the underlying ticker.

Wide earns its keep here and nowhere else — one type, 9.3M rows a day, 56 stable
columns, 77.7% populated. Numeric properties are `double` columns and the rest
are strings, decided per predicate by the same majority rule the feature
extractor uses. A snapshot missing a property gets a null, not a dropped row.

The sort is what makes a ticker query cheap. There are 525 underlying tickers
and 19 snapshots a day, a median 13,224 rows per ticker, so a single-ticker
filter wants 0.14% of a day. Parquet prunes row groups on min/max statistics:
sorted, a 30-day ticker query reads a few hundred MB; unsorted, every ticker
appears in every row group and the same query reads all 39 GB.

### `graph/`

The non-market subgraph as a [pyoxigraph](https://pyoxigraph.readthedocs.io/)
store, one per day — a directory, not a Parquet file.

`edges/` and `nodes/` answer a one-hop question well and a two-hop question
badly: a self-join carrying a mandatory same-day guard, with hand-rolled
recursion for anything deeper. Measured on one day: 1,395,049 triples in a
194 MB store, counting `affectsRegion` edges in 1 ms and the two-hop
`weather → region ← measurement` traversal in 11 ms.

```python
import pyoxigraph

store = pyoxigraph.Store.read_only("graph/day=2026-09-10")
for row in store.query("""
    SELECT ?measurement WHERE {
      ?alert       <https://jefflevesque.com/ontology/noaa/affectsRegion> ?region .
      ?measurement <https://jefflevesque.com/ontology/bls/hasRegion>      ?region
    }
"""):
    print(row["measurement"].value)
```

**One day fits in memory; a window does not.** At 139 bytes a triple a 30-day
window is 5.8 GB and a year is 71 GB, against 194 MB for a day. So the store
supplements the tables rather than replacing them: load one day for traversal,
and fall back to `edges/` for anything spanning more.

**Market never enters it.** At the same byte rate its ~415M triples a day would
be roughly 58 GB, and market is a time series of numbers carrying one edge per
snapshot — not a shape a triple store earns anything on.

The store holds the *enriched* frame's view of the data: literals are the plain
strings the loader converted them to, with their source datatypes already folded
in. It is the same view every other artifact in this pipeline is built from, and
not a re-serialization of the original RDF.

## What is deliberately not here

- **Market literals in `facts/`, and market subjects in `graph/`.** They are in
  `snapshots/` and in `edges/`.
- **A triple whose object is a URI nothing typed.** It is no edge, and putting a
  dangling pointer in a `value` column would have consumers reading it as a name.
- **Node feature vectors.** The dense matrix is 38.16 GB for the dominant type,
  of which 183 of 1024 dimensions vary per row.
- **Anything that answers a question the data cannot.** See
  [What the graph answers](questions.md).
