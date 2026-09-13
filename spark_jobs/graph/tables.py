"""
The query tables: what a downstream service actually reads.

A published run is a pickle, a node index and 200 Parquet parts of triples --
about 95 GB, none of it filterable by ticker, by date or by meaning, and the
graph's structure only recoverable by unpickling a 42 GB blob. These tables are
the same data in shapes something can query, at roughly 2.5% of the size.

Five tables here:

    nodes/        (node_type, node_id, uri)
    edges/        (src_type, src_id, relation, dst_type, dst_id)
    edge_types/   one row per edge type: origin, predicate_uri, count
    facts/        every non-market literal, long format
    entities/     the text each node carries, assembled

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
takes the newest row per URI with
``ROW_NUMBER() OVER (PARTITION BY uri ORDER BY day DESC)``.

Market is not here. It is 99.5% of the graph, a time series of numbers carrying
one edge per snapshot, and it gets its own wide table.
"""
import logging
from typing import Dict

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_jobs.graph.config import JobConfig
from spark_jobs.pyg_builder.naming import (
    EXCLUDED_EDGE_PREDICATES,
    RDF_TYPE,
    prefixed_local_name_expr,
    relation_to_predicate_uri,
)
from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.utils.rdf_utils import (
    MARKET_NODE_TYPE_PREFIXES,
    classify_edge_origin,
)
from spark_jobs.utils.spark_rdf_utils import collect_sorted, numeric_literal_expr

# The job's logger, not this module's -- see the note in graph/config.py.
logger = logging.getLogger("build_graph")

# zstd rather than the snappy default. These are published artifacts read far
# more often than they are written, and the measurements the issue sizes them
# from (an edge at 9.41 bytes, a fact at 26.7) are zstd numbers.
COMPRESSION = "zstd"

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


def _write(df: DataFrame, root: str, table: str, day: str) -> str:
    path = table_path(root, table, day)
    (
        df
        .write
        .mode("overwrite")
        .option("compression", COMPRESSION)
        .parquet(path)
    )
    return path


def _is_market(column: str = "node_type") -> F.Column:
    """Whether a node type belongs to market data."""
    expr = F.lit(False)
    for prefix in MARKET_NODE_TYPE_PREFIXES:
        expr = expr | F.col(column).startswith(prefix)
    return expr


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


def write_edge_types(
    spark: SparkSession, edges_df: DataFrame, root: str, day: str
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

    Small -- one row per edge type, 837 on a production day -- so it is counted
    and built on the driver.
    """
    rows = collect_sorted(
        edges_df
        .groupBy("src_type", "relation", "dst_type")
        .agg(F.count("*").alias("count"))
    )

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
    frame = spark.createDataFrame(described, schema) if described else (
        spark.createDataFrame([], schema)
    )

    return _write(frame.coalesce(1), root, "edge_types", day)


# ============================================
# facts/ and entities/
# ============================================
def non_market_facts(
    triples_df: DataFrame, node_id_df: DataFrame
) -> DataFrame:
    """Every literal a non-market node carries, one row per value.

    Long format, one shape for every source, because wide would mean about 150
    tables: the median non-market type holds two literal predicates, the widest
    holds 21, and 119 of 155 types hold under a thousand nodes. A new source
    appears in this table with no code change.

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
    literals = candidates.join(
        objects,
        candidates["object"] == objects["_object_uri"],
        "left_anti",
    )

    return (
        literals
        .join(
            subjects,
            literals["subject"] == subjects["_subject_uri"],
            "inner",
        )
        .drop("_subject_uri")
        .filter(~_is_market("node_type"))
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

    This emits the text that exists and nothing more. Five of 155 node types
    carry any at all, so a search over it reaches those five -- see the module
    docstring on _TEXT_PREDICATE_TERMS.
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

    # An empty config on purpose: no node_types allowlist, no temporal or
    # sector switch. Whatever subset this run's .pt is built over, the tables
    # cover everything the sources carried.
    node_id_df, node_counts = NodeMapper(spark, {}).build_node_id_table(
        triples_df
    )

    written: Dict[str, str] = {}
    try:
        written["nodes"] = write_nodes(node_id_df, root, day)

        edges_df = resolve_edges(triples_df, node_id_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            written["edges"] = write_edges(edges_df, root, day)
            written["edge_types"] = write_edge_types(
                spark, edges_df, root, day
            )
        finally:
            edges_df.unpersist()

        facts_df = non_market_facts(triples_df, node_id_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            written["facts"] = write_facts(facts_df, root, day)
            written["entities"] = write_entities(facts_df, root, day)
        finally:
            facts_df.unpersist()
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
