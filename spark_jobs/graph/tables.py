"""
The query tables: what a downstream service actually reads.

A published run is a pickle, a node index and 200 Parquet parts of triples --
about 95 GB, none of it filterable by ticker, by date or by meaning, and the
graph's structure only recoverable by unpickling a 42 GB blob. These tables are
the same data in shapes something can query, at roughly 2.5% of the size.

Six tables and a triple store:

    nodes/        (node_type, node_id, uri)
    edges/        (src_type, src_id, relation, dst_type, dst_id)
    edge_types/   one row per edge type: origin, predicate_uri, count
    facts/        every literal a snapshot does not carry, long format
    entities/     the text each node carries, assembled
    snapshots/    market quotes, pivoted wide -- one row per snapshot
    graph/        the non-snapshot subgraph as a triple store (graph_store.py)

Three things about them are deliberate and easy to get wrong:

**They are written over ALL the data, whatever the run is building.** A run may
build its ``.pt`` over a subset -- a node_types allowlist, temporal or sector
nodes switched off -- and the tables must not inherit that. Weather is the
standing example: NOAA is 0.076% of the nodes, shares nothing between days and
reaches market through no edge at all, so it earns little in a model, but it
answers real questions in a table. That is why the node table here is built
fresh with an empty config rather than reusing the run's.

**Node ids are day-scoped.** ``node_id`` is ``row_number()`` over a uri-ordered
window within a type, so a URI's id changes whenever the node set does, which is
every day. Edges from day D may only be joined to nodes from day D. The column
order puts ``node_type`` beside its id to make the pairing visible.

**Every table is written every day, and deduplicated on read.** Measured churn
between two days: BLS repeats 99.9% of its nodes, NOAA repeats none. No
stable-versus-daily split is right for both, so there is no split -- a consumer
takes each URI's rows from the newest day that has it, with
``RANK() OVER (PARTITION BY uri ORDER BY day DESC)``.

``RANK``, not ``ROW_NUMBER``: a node has a row per value in ``facts/``, and
``ROW_NUMBER`` keeps one of them and drops the rest.

A market SNAPSHOT is in ``edges/`` and in ``snapshots/`` and nowhere else. It is
99.5% of the graph, a time series of numbers carrying one edge per snapshot --
not a shape a long table or a triple store serves well, and the links it has to
everything else are the part that matters.

The line is drawn at snapshots rather than at market, because the two are not
the same set. Market enrichment mints a handful of hub nodes -- the GICS sector
nodes and the moneyness classes -- and they are neither big nor a time series:
fourteen of them against 10.2 million quotes on 2026-09-17, eight carrying a
value. They go where every other node goes, so a sector's
``relationConfidence`` is a fact and its route to the economic sectors is in
the store.
"""
import logging
from typing import Dict, List

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_jobs.graph.config import JobConfig
from spark_jobs.graph.graph_store import write_graph_store
from spark_jobs.pyg_builder.naming import (
    ALLOW_UNREGISTERED_NAMESPACES,
    EXCLUDED_EDGE_PREDICATES,
    RDF_TYPE,
    check_registered_names,
    prefixed_local_name,
    prefixed_local_name_expr,
    relation_to_predicate_uri,
)
from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.utils.rdf_utils import (
    MARKET_QUOTES,
    SNAPSHOT_NODE_TYPE_PREFIX,
    classify_edge_origin,
)
from spark_jobs.utils.spark_rdf_utils import (
    NUMERIC_PREDICATE_MIN_SHARE,
    collect_sorted,
    numeric_literal_expr,
)

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# zstd rather than the snappy default. These are published artifacts read far
# more often than they are written, and the measurements the issue sizes them
# from (an edge at 9.41 bytes, a fact at 26.7) are zstd numbers.
COMPRESSION = "zstd"

# What a snapshot row is sorted by, in order, when the run carries them: the
# underlying ticker first, then the contract's own symbol. Named from the
# vocabulary so a re-homed namespace moves the column and this together.
_SNAPSHOT_SORT_TERMS = (
    prefixed_local_name(str(MARKET_QUOTES.underlyingSymbol)),
    prefixed_local_name(str(MARKET_QUOTES.symbol)),
)

# Local names whose values are prose rather than a code, a date or a
# measurement. Matched against the local name, case-insensitively, so
# ``filings_hasIssuerName`` and ``rdfs_label`` both qualify.
#
# A declared list, not a derived one: nothing in the data distinguishes a name
# from an identifier -- both are short non-numeric literals -- so the judgement
# has to be written down somewhere. Kept deliberately small. What comes out is
# the text that exists, mostly bare names like INTUIT and FORM 4; turning a node
# and its neighbours into a real sentence is a separate job.
_TEXT_PREDICATE_TERMS = (
    "label",
    "comment",
    "name",
    "title",
    "description",
    "summary",
    "headline",
    "text",
)


def table_path(root: str, table: str, day: str) -> str:
    """Where one day of one table lives.

    ``day=`` is a directory rather than a column, which is what lets Spark's
    partition discovery hand a consumer reading the parent a real ``day``
    column to filter, prune and deduplicate on.
    """
    return f"{root}/{table}/day={day}"


# The RAPIDS Parquet writer crashes the executor writing these tables as zstd --
# an illegal memory access, again on every retry -- where the CPU writer wrote the
# same 9,447,814 nodes. So only the encoding runs on the CPU; every step that
# builds the frame stays on the GPU. Snappy on the GPU writer also works, but came
# out five times larger.
_GPU_PARQUET_WRITE = "spark.rapids.sql.format.parquet.write.enabled"


def _write(df: DataFrame, root: str, table: str, day: str) -> str:
    path = table_path(root, table, day)
    conf = df.sparkSession.conf
    previous = conf.get(_GPU_PARQUET_WRITE, None)
    conf.set(_GPU_PARQUET_WRITE, "false")
    try:
        (
            df
            .write
            .mode("overwrite")
            .option("compression", COMPRESSION)
            .parquet(path)
        )
    finally:
        if previous is None:
            conf.unset(_GPU_PARQUET_WRITE)
        else:
            conf.set(_GPU_PARQUET_WRITE, previous)
    return path


def _quoted(name: str) -> str:
    """A column named from the data, referenced safely.

    A predicate's local name can carry a dot, and ``F.col("a.b")`` reads that as
    a field inside a struct called ``a`` rather than as a column called ``a.b``.
    Every column in the wide market table is named from a predicate, so every
    reference to one goes through here.
    """
    return f"`{name}`"


def _is_snapshot(column: str = "node_type") -> F.Column:
    """Whether a node type is a market snapshot.

    Snapshots, not market: the market-enrichment hub nodes are in the same
    source and belong on the other side of this line. See the module docstring.
    """
    return F.col(column).startswith(SNAPSHOT_NODE_TYPE_PREFIX)


# ============================================
# nodes/
# ============================================
def write_nodes(node_id_df: DataFrame, root: str, day: str) -> str:
    """The identity map: which entity each graph row is.

    The same three columns as the run's own ``node_index/`` (see
    ``constructor._node_index``), republished on the tables' schedule rather
    than the run's -- runs expire at 21 days and these do not, and an ``edges/``
    row is unreadable without the nodes it names.

    Ordered rather than merely written: sorted output gives Parquet row groups
    tight min/max statistics, so a reader after one node type skips the rest.
    """
    return _write(
        node_id_df
        .select("node_type", "node_id", "uri")
        .orderBy("node_type", "node_id"),
        root, "nodes", day,
    )


# ============================================
# edges/
# ============================================
def resolve_edges(triples_df: DataFrame, node_id_df: DataFrame) -> DataFrame:
    """Triples whose subject and object are both nodes, as edges.

    The double join is what makes an edge an edge: a triple pointing at a
    literal is a fact, and a triple pointing at a URI nothing typed is neither.

    ``EdgeMapper.build_edge_indices`` resolves edges the same way for the ``.pt``
    and cannot be called here -- it imports torch, which the enrichment leg does
    not require, and it goes on to collect tensors this has no use for. The two
    share the excluded predicates and the naming rule, and the e2e check that
    ``edges/`` sums to ``graph_schema.json``'s edge counts is what holds them to
    the same answer.
    """
    src = node_id_df.select(
        F.col("uri").alias("_src_uri"),
        F.col("node_id").alias("src_id"),
        F.col("node_type").alias("src_type"),
    )
    dst = node_id_df.select(
        F.col("uri").alias("_dst_uri"),
        F.col("node_id").alias("dst_id"),
        F.col("node_type").alias("dst_type"),
    )

    candidates = triples_df.filter(
        ~F.col("predicate").isin(list(EXCLUDED_EDGE_PREDICATES))
    )

    return (
        candidates
        .join(src, candidates["subject"] == src["_src_uri"], "inner")
        .drop("_src_uri")
        .join(dst, F.col("object") == dst["_dst_uri"], "inner")
        .drop("_dst_uri")
        .withColumn("relation", prefixed_local_name_expr("predicate"))
        .select(
            "src_type",
            F.col("src_id").cast("long"),
            "relation",
            "dst_type",
            F.col("dst_id").cast("long"),
        )
        .dropDuplicates(
            ["src_type", "src_id", "relation", "dst_type", "dst_id"]
        )
    )


def _predicates_named(triples_df: DataFrame, relations: List[str]) -> List[str]:
    """The predicate URIs the naming rule turns into these relation names, for
    an error message."""
    rows = (
        triples_df
        .filter(prefixed_local_name_expr("predicate").isin(relations))
        .select("predicate")
        .distinct()
        .limit(50)
        .collect()
    )
    return [row["predicate"] for row in rows]


def write_edges(edges_df: DataFrame, root: str, day: str) -> str:
    """The graph's structure, which nothing published has ever carried.

    Ids rather than URIs, because URIs would multiply the table several times
    over for nothing a join cannot recover -- at the cost that the ids only mean
    anything alongside the same day's ``nodes/``.

    Sorted by edge type so a consumer asking about one relation reads the row
    groups holding it and skips the rest.
    """
    return _write(
        edges_df.orderBy("src_type", "relation", "dst_type", "src_id"),
        root, "edges", day,
    )


def count_edge_types(edges_df: DataFrame) -> List:
    """One row per edge type and its edge count, in a stable order.

    Small -- one row per edge type, 837 on a production day -- so it is
    collected. write_query_tables counts before it writes any edge, so a
    relation from an unregistered namespace stops the tables before edges/ is
    written, and edge_types/ is built from the same rows.
    """
    return collect_sorted(
        edges_df
        .groupBy("src_type", "relation", "dst_type")
        .agg(F.count("*").alias("count"))
    )


def write_edge_types(
    spark: SparkSession, rows: List, root: str, day: str
) -> str:
    """What each edge type MEANS, so an edge stays readable with no run present.

    ``edges/`` carries a relation name and nothing else. The predicate it came
    from, and whether this pipeline observed the link or inferred it, live in
    ``graph_schema.json`` -- which belongs to a run, and runs expire at 21 days
    while these tables keep a year. Without this table the last 344 days of
    edges are uninterpretable.

    ``origin`` is keyed by the FULL edge type rather than by the relation name,
    because it depends on the endpoints as well as the predicate: the same
    relation is raw from a source-typed subject and enrichment from one this
    pipeline minted. Keyed by name those collapse to whichever was written last,
    which reports an inferred link as an observed fact.

    ``rows`` are ``count_edge_types``' rows; the table is built from them on
    the driver.
    """
    described = []
    for row in rows:
        predicate_uri = relation_to_predicate_uri(row["relation"])
        described.append((
            row["src_type"],
            row["relation"],
            row["dst_type"],
            row["count"],
            predicate_uri,
            classify_edge_origin(
                predicate_uri, row["src_type"], row["dst_type"]
            ),
            # Same value as `relation`, under the name graph_schema.json uses
            # for it: the group a consumer may share weights over. Separate
            # because it answers a different question, and could widen later
            # without repurposing a column that means something else.
            row["relation"],
        ))

    schema = (
        "src_type string, relation string, dst_type string, count long, "
        "predicate_uri string, origin string, relation_group string"
    )
    # An empty list is fine against a DDL schema, so a run with no edges at all
    # publishes an empty table rather than no table -- a consumer reading it
    # gets zero rows instead of a missing path.
    frame = spark.createDataFrame(described, schema)

    return _write(frame.coalesce(1), root, "edge_types", day)


# ============================================
# facts/ and entities/
# ============================================
def literals(triples_df: DataFrame, node_id_df: DataFrame) -> DataFrame:
    """Every literal every node carries, one row per value.

    The frame both long and wide tables come from -- ``facts/`` takes the
    non-market rows as they are, ``snapshots/`` pivots the market ones -- so
    the split between the two is one filter and not two code paths.

    The anti-join is the same test the feature extractor uses to separate
    literals from edges -- a triple whose object is a known node URI is an edge,
    and what remains is literal-valued. ``rdf:type`` is dropped because it says
    what a node IS, which ``nodes/`` already answers.

    One thing is dropped that the feature extractor keeps: an object that still
    LOOKS like a URI after the anti-join. Nothing typed it, so it became no
    node and no edge, and it is not a literal value either -- it is a pointer
    at something absent. In a ``value`` column a consumer would read it as a
    name. Rare in practice (88.2% of URI-object triples resolve to edges and
    almost all the rest are rdf:type targets, which are already gone), and
    documented as a known omission rather than counted anywhere.

    Both the predicate URI and its short name are carried. The URI is the honest
    key and joins to ``ontology_schema.json``; the name is what a query is
    written against, and putting it here saves every consumer from
    reimplementing the naming rule. At roughly 1.4M rows a day the second column
    costs nothing.
    """
    subjects = node_id_df.select(
        F.col("uri").alias("_subject_uri"),
        F.col("node_type"),
        F.col("uri"),
    )
    objects = node_id_df.select(F.col("uri").alias("_object_uri"))

    candidates = triples_df.filter(F.col("predicate") != RDF_TYPE)
    valued = candidates.join(
        objects,
        candidates["object"] == objects["_object_uri"],
        "left_anti",
    )

    return (
        valued
        .join(
            subjects,
            valued["subject"] == subjects["_subject_uri"],
            "inner",
        )
        .drop("_subject_uri")
        .filter(~F.col("object").rlike(r"^(https?://|_:)"))
        .select(
            "node_type",
            "uri",
            F.col("predicate"),
            prefixed_local_name_expr("predicate").alias("predicate_name"),
            F.col("object").alias("value"),
            numeric_literal_expr("object").isNotNull().alias("is_numeric"),
        )
    )


def write_facts(facts_df: DataFrame, root: str, day: str) -> str:
    """Every non-market literal, sorted so one node type reads cheaply."""
    return _write(
        facts_df.orderBy("node_type", "uri", "predicate"),
        root, "facts", day,
    )


def write_entities(facts_df: DataFrame, root: str, day: str) -> str:
    """One row per node that carries text, for the side that embeds it.

    ``text`` is the node's text-bearing values in predicate order, joined with
    "; ". Deterministic rather than clever: the assembly is stable run to run,
    which is what lets a vector index be rebuilt and compared.

    This emits the text that exists and nothing more. On 2026-09-09 that was
    12,607 nodes across 49 node types, mostly names and labels -- see
    _TEXT_PREDICATE_TERMS for which predicates count as text, and why that
    is a written-down list rather than something derived.
    """
    is_text = F.lit(False)
    for term in _TEXT_PREDICATE_TERMS:
        is_text = is_text | F.lower(F.col("predicate_name")).contains(term)

    return _write(
        facts_df
        .filter(~F.col("is_numeric") & is_text & (F.length("value") > 0))
        .groupBy("node_type", "uri")
        .agg(
            F.concat_ws(
                "; ",
                F.sort_array(F.collect_set(
                    F.concat_ws(": ", F.col("predicate_name"), F.col("value"))
                )),
            ).alias("text")
        )
        .orderBy("node_type", "uri"),
        root, "entities", day,
    )


# ============================================
# graph/
# ============================================
def non_snapshot_triples(
    triples_df: DataFrame, node_id_df: DataFrame
) -> DataFrame:
    """Every triple describing something other than a market snapshot.

    By SUBJECT: a triple belongs to the node it describes. So an edge FROM a
    snapshot into a company is a snapshot row and stays out, while the
    company's own triples stay in -- the same line ``facts/`` and
    ``snapshots/`` draw, applied to whole triples rather than to literals.

    The market hub nodes are on the other side of that line: an
    ``EquitySector`` and its route to the economic sectors are in the store,
    because a handful of hubs linking market to the rest of the graph is what
    the store is for. Only the 10.2 million quotes a day stay out.

    A triple whose subject is no node at all stays in, whatever vocabulary it
    names. Those are the statements ABOUT the terms -- the derived
    ``rdfs:subClassOf`` hierarchy, the observed domains and ranges, the
    provenance markers -- whose subjects are predicate and class URIs rather
    than entities. They are the store's schema, which is most of what makes a
    SPARQL query over it worth writing, and a few hundred triples carry them.
    So the store holds the market VOCABULARY while holding no quote.
    """
    snapshot_uris = (
        node_id_df
        .filter(_is_snapshot("node_type"))
        .select(F.col("uri").alias("_snapshot_uri"))
    )

    return (
        triples_df
        .join(
            snapshot_uris,
            triples_df["subject"] == snapshot_uris["_snapshot_uri"],
            "left_anti",
        )
        .select("subject", "predicate", "object")
    )


# ============================================
# snapshots/
# ============================================
def _numeric_columns(snapshot_df: DataFrame) -> set:
    """Which pivoted columns should hold numbers rather than strings.

    The same majority rule the feature extractor classifies literal predicates
    by, for the same reason: a predicate whose values mostly parse as numbers
    is a measurement, and one where only a minority do is a set of labels that
    happen to look numeric. A column typed off a per-value test would make
    ``strikePrice > 100`` answerable for some rows and not others.

    One aggregation over the snapshot literals, collected at one row per
    predicate -- 45 on a production day.
    """
    shares = (
        snapshot_df
        .groupBy("predicate_name")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.col("is_numeric").cast("int")).alias("numeric"),
        )
        .collect()
    )

    return {
        row["predicate_name"]
        for row in shares
        if row["total"]
        and row["numeric"] / row["total"] > NUMERIC_PREDICATE_MIN_SHARE
    }


def write_snapshots(snapshot_df: DataFrame, root: str, day: str) -> str:
    """Market quotes, pivoted wide: one row per snapshot, one column per property.

    Wide earns its keep here and nowhere else. Quotes are two node types with
    10.2M rows a day and 45 stable columns, 77.8% of them populated (measured
    2026-09-17) -- a shape a long table serves badly and a pivot serves well.
    Every other source has a median of two literal predicates per type, which
    is why they stay long.

    Quotes only. The market hub nodes -- eight GICS sector nodes carrying a
    relationConfidence, and the moneyness classes -- are not snapshots, and
    pivoting them in put a column on 10.2M quote rows that was null on every
    one of them.

    Sorted by the underlying ticker. Measured: 525 underlying tickers, 19
    snapshots a day, a median 13,224 rows per ticker -- so a single-ticker
    filter wants 0.14% of a day. Parquet prunes row groups on min/max
    statistics, so sorted, a 30-day ticker query reads a few hundred MB;
    unsorted, every ticker appears in every row group, nothing prunes, and the
    same query reads all 39 GB.

    A snapshot missing a property gets a null, not a dropped row.

    Returns "" when the run carried no quotes at all, which is a legitimate
    configuration (a no-market dataset) rather than a failure.
    """
    columns = sorted(
        row["predicate_name"]
        for row in snapshot_df.select("predicate_name").distinct().collect()
    )
    if not columns:
        logger.info("  No snapshot literals in this run -- no snapshots table")
        return ""

    numeric = _numeric_columns(snapshot_df)

    # max() rather than first(): a subject carrying two values for one
    # predicate has to resolve the same way on every run, and first() over a
    # shuffled frame does not.
    pivoted = (
        snapshot_df
        .groupBy("node_type", "uri")
        .pivot("predicate_name", columns)
        .agg(F.max("value"))
    )

    typed = pivoted.select(
        "node_type",
        "uri",
        *[
            numeric_literal_expr(_quoted(name)).alias(name) if name in numeric
            else F.col(_quoted(name))
            for name in columns
        ],
    )

    # The sort keys are named after the vocabulary rather than spelled out, and
    # only used when the run actually carries them -- an equity-only day has no
    # underlying symbol, and sorting by a column that is not there would fail
    # the write rather than the query.
    sort_keys = [
        _quoted(name) for name in _SNAPSHOT_SORT_TERMS if name in columns
    ] + ["uri"]

    return _write(typed.orderBy(*sort_keys), root, "snapshots", day)


# ============================================
# The whole set
# ============================================
def write_query_tables(
    spark: SparkSession, triples_df: DataFrame, config: JobConfig
) -> Dict[str, str]:
    """Write every query table for this run's day. Returns table -> path.

    Returns an empty dict, and writes nothing, when the run asked for no tables
    or cannot say which day its data describes. Neither is a failure: the graph
    the run builds is unaffected either way.
    """
    if not config.enable_query_tables:
        logger.info("Query tables disabled (--enable_query_tables false)")
        return {}

    if not config.source_data_day:
        logger.warning(
            "No query tables: source_paths name no single day, and no "
            "--source_data_day was given, so there is no day to write them "
            "under"
        )
        return {}

    day = config.source_data_day
    root = config.query_tables_path

    logger.info("=" * 80)
    logger.info(f"PHASE: WRITING QUERY TABLES (day={day})")
    logger.info("=" * 80)

    # No node_types allowlist and no temporal or sector switch, on purpose:
    # whatever subset this run's .pt is built over, the tables cover everything
    # the sources carried. The namespace allowance is the one setting carried
    # over, so a run that allows unregistered namespaces is not stopped here.
    allow_unregistered = bool(
        config.pyg_config.get(ALLOW_UNREGISTERED_NAMESPACES, False)
    )
    node_id_df, node_counts = NodeMapper(
        spark, {ALLOW_UNREGISTERED_NAMESPACES: allow_unregistered}
    ).build_node_id_table(triples_df)

    written: Dict[str, str] = {}
    try:
        written["nodes"] = write_nodes(node_id_df, root, day)

        edges_df = resolve_edges(triples_df, node_id_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            edge_types = count_edge_types(edges_df)
            check_registered_names(
                "Relations",
                (row["relation"] for row in edge_types),
                allow_unregistered,
                lambda names: _predicates_named(triples_df, names),
            )
            written["edges"] = write_edges(edges_df, root, day)
            written["edge_types"] = write_edge_types(
                spark, edge_types, root, day
            )
        finally:
            edges_df.unpersist()

        # One frame, two shapes: the market rows pivot wide and the rest stay
        # long. Cached because both readings scan it.
        literals_df = literals(triples_df, node_id_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            facts_df = literals_df.filter(~_is_snapshot("node_type"))
            written["facts"] = write_facts(facts_df, root, day)
            written["entities"] = write_entities(facts_df, root, day)

            snapshots = write_snapshots(
                literals_df.filter(_is_snapshot("node_type")), root, day
            )
            if snapshots:
                written["snapshots"] = snapshots
        finally:
            literals_df.unpersist()

        store = write_graph_store(
            non_snapshot_triples(triples_df, node_id_df),
            table_path(root, "graph", day),
        )
        if store:
            written["graph"] = store
    finally:
        node_id_df.unpersist()

    logger.info(
        f"Wrote {len(written)} query tables for {day} "
        f"({sum(node_counts.values()):,} nodes across "
        f"{len(node_counts)} types)"
    )
    for table, path in written.items():
        logger.info(f"    {table}: {path}")

    return written
