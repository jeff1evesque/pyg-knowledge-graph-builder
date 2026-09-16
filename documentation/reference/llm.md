# Using the Tables with an LLM

The [query tables](tables.md) can be the retrieval side of retrieval-augmented
generation (RAG) with any LLM. This repository publishes the tables and stops
there: it calls no LLM, builds no embeddings, and runs no service that queries
the tables for you. The query, the prompt and the LLM are yours.

The `.pt` has no part in this. It holds numbers without names; see
[The tables, not the `.pt`](questions.md#the-tables-not-the-pt).

## What each table gives an LLM

| Table | Retrieve it for | Watch for |
|---|---|---|
| `nodes/` and `edges/` | what an entity is linked to | ids join only within a day; across days, join on `uri` |
| `edge_types/` | the relations that exist, and whether each link was stated by a source (`raw`) or inferred by this pipeline (`enrichment`, `unification`) | 838 rows on 2026-09-09 |
| `facts/` | names, dates, amounts and codes on every non-market node | several rows per node |
| `snapshots/` | option and equity quotes | 9.3 million rows a day: summarize before prompting |
| `entities/` | finding a node by name, or embedding | mostly short names and labels |
| `graph/` | multi-hop questions over one day, in SPARQL | one day at a time, and no market data |

## The loop

1. **Pick the days.** A query reads one `day=` partition or many. Node ids join
   only within a day, so follow an entity across days by its URI.
2. **Find the starting node:** by a value in `facts/`, such as a ticker; by text
   in `entities/`; or by a column in `snapshots/`.
3. **Walk its edges.** Join `edges/` to the same day's `nodes/`.
   [Questions the tables answer](questions.md) gives the path for each kind of
   question.
4. **Collect the values:** `facts/` for every node the walk reached, and a
   summary of `snapshots/` for market data.
5. **Write the rows as lines of text**, one edge or value per line with its URI,
   so an answer can point at the line it used.
6. **Send the lines and the question to your LLM**, and ask it to answer from
   those lines only.

## A worked example

One company on one day, in Python with DuckDB, which reads the published
Parquet directly. Any engine that reads Hive-partitioned Parquet can run the
same queries.

```python
import duckdb

ROOT = "s3://BUCKET/PREFIX/all-sources"  # <PYG_TABLES_ROOT>/<dataset>, or a local copy
DAY = "2026-09-09"
TICKER = "NVDA"

con = duckdb.connect()
if ROOT.startswith("s3://"):
    con.sql("INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain)")


def day(table):
    return f"read_parquet('{ROOT}/{table}/day={DAY}/*.parquet')"


def linked(uris):
    """Every edge between `uris` and a non-market node, in its stored direction."""
    return con.execute(f"""
        WITH n AS (FROM {day('nodes')}),
             seed AS (SELECT node_type, node_id FROM n WHERE list_contains($uris, uri)),
             e AS (
                 SELECT e.* FROM {day('edges')} e
                 JOIN seed ON e.src_type = seed.node_type AND e.src_id = seed.node_id
                 WHERE e.dst_type NOT LIKE 'market%'
                 UNION
                 SELECT e.* FROM {day('edges')} e
                 JOIN seed ON e.dst_type = seed.node_type AND e.dst_id = seed.node_id
                 WHERE e.src_type NOT LIKE 'market%'
             )
        SELECT s.uri, e.relation, d.uri, t.origin
        FROM e
        JOIN n s ON s.node_type = e.src_type AND s.node_id = e.src_id
        JOIN n d ON d.node_type = e.dst_type AND d.node_id = e.dst_id
        JOIN {day('edge_types')} t USING (src_type, relation, dst_type)
        ORDER BY ALL
    """, {"uris": uris}).fetchall()


# 1. The company the ticker names.
company = con.execute(f"""
    SELECT uri FROM {day('facts')}
    WHERE predicate_name = 'bls_enrichment_ticker' AND value = ?
""", [TICKER]).fetchone()[0]

# 2. One hop out: its issuers, sector and peers. Two hops: the issuers' filings.
edges = linked([company])
issuers = [s for s, rel, d, _ in edges if rel == "bls_enrichment_refersToCompany"]
edges += [row for row in linked(issuers) if row[1] == "filings_hasIssuer"]

# 3. The values on every node those edges reach.
uris = sorted({company} | {s for s, *_ in edges} | {d for _, _, d, _ in edges})
values = con.execute(f"""
    SELECT uri, predicate_name, value FROM {day('facts')}
    WHERE list_contains($uris, uri) ORDER BY ALL
""", {"uris": uris}).fetchall()

# 4. The option chain, summarized per snapshot rather than row by row.
chain = con.execute(f"""
    SELECT market_quotes_captureTime, count(*), round(avg(market_quotes_volatility), 3),
           sum(market_quotes_openInterest)
    FROM {day('snapshots')}
    WHERE market_quotes_underlyingSymbol = ?
    GROUP BY ALL ORDER BY ALL
""", [TICKER]).fetchall()

# 5. One line per row, then the prompt.
lines = [f"{s} {rel} {d} ({origin})" for s, rel, d, origin in edges]
lines += [f"{uri} {name} {value!r}" for uri, name, value in values]
lines += [f"{TICKER} options at {at}: {n} contracts, mean volatility {vol}, open interest {oi:,.0f}"
          for at, n, vol, oi in chain]

question = "What did NVIDIA file on 2026-09-09, and how did its options trade that day?"
prompt = (
    f"Facts from the query tables for {DAY}, one per line:\n\n"
    + "\n".join(lines)
    + f"\n\nAnswer using only these facts, and name the line each claim comes from. "
      f"If they do not answer it, say so.\n\nQuestion: {question}"
)
# Send `prompt` to the LLM of your choice.
```

What it relies on:

- **The ticker is a fact on the company.** A `sec_enrichment_UnifiedCompany`
  node carries its ticker as `bls_enrichment_ticker` in `facts/`.
- **Market is summarized, not listed.** A company has a `refersToCompany` edge
  from every one of its option snapshots, a median 13,224 a day per ticker, so
  `linked` skips market nodes, and step 4 reads `snapshots/` by
  `market_quotes_underlyingSymbol` instead.
- **Each edge line carries its `origin`**, so the LLM can tell a link a source
  stated from one this pipeline inferred.

On S3, each call to `linked` reads the whole day of `edges/`, about 0.67 GB. For
more than a few questions, copy the day's tables to local disk and point `ROOT`
at the copy.

## Letting the LLM write the query

Instead of fixed queries, the LLM can write the SQL. Give it:

- the columns of each table, from [Query tables](tables.md#the-tables);
- the day's `edge_types/` rows for the node types involved, so it writes the
  real names rather than guessing them;
- the rules: join `edges/` to `nodes/` on the same `day`; across days, take each
  URI's rows from its newest day
  ([Written every day, deduplicated on read](tables.md#written-every-day-deduplicated-on-read));
  and look at both ends of `sharesSubIndustryWith`, which is stored once per
  pair.

Run what it writes with read-only access, check how many rows come back before
sending them, and have it answer from the rows.

## Embedding `entities/`

`entities/` holds `(node_type, uri, text)`, one row per node that carries text,
where `text` is the node's text values as `predicate_name: value` pairs joined
with `; `. Embed `text` with an embedding model of your choice and keep `uri`
beside each vector: the URI is what every other table joins on.

It finds a node by name. It does not answer a question by similarity: on
2026-09-09 it held 12,607 nodes across 49 node types, mostly names and labels,
and 901 of the 990 texts with 300 or more characters were weather alert
descriptions. A company carries no text of its own, so search for its issuer's
name and follow `refersToCompany` from there.

## Access

The tables are wherever `PYG_TABLES_ROOT` points, an S3 prefix. Reading them
needs read access to that prefix, which the identity that publishes them does
not have. Read only days whose `_days/<day>.json` marker exists. `graph/` is a
store directory rather than Parquet: copy one day to local disk and open it with
pyoxigraph, as
[Weather against regional economics](questions.md#weather-against-regional-economics)
shows.
