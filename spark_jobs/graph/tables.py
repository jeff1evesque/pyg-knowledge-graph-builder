"""
The query tables: what a downstream service actually reads.

A published run is a pickle, a node index and 200 Parquet parts of triples --
about 95 GB, none of it filterable by ticker, by date or by meaning, and the
graph's structure only recoverable by unpickling a 42 GB blob. These tables are
the same data in shapes something can query, at roughly 2.5% of the size.

Nine tables and a triple store:

    nodes/        (node_type, node_id, uri)
    edges/        (src_type, src_id, relation, dst_type, dst_id)
    edge_types/   one row per edge type: origin, predicate_uri, count
    facts/        every literal a snapshot does not carry, long format
    entities/     the text each node carries, assembled
    snapshots/    market quotes, pivoted wide -- one row per snapshot
    splits/       the day's stock splits, read from the split feed (splits.py)
    companyfacts/ the SEC numbers filed that day, read from the company facts
                  history (companyfacts.py)
    minute_quotes/ the constituents' quotes, one row per stock per minute,
                  read from the minute feed (minute_quotes.py)
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

The companyfacts snapshot's facts are in no table. The snapshot restates every
company's latest numbers each day, so they would repeat in every day's tables
until a company files again; the 2026-10-05 snapshot matched 2026-10-04's on all
15,618 rows. ``companyfacts/`` holds the numbers instead, each on the day it was
filed. The issuers the snapshot names stay, as they are the filings feed's own.
The period days only its facts were dated to go with them, as nothing else
points at them (days_only_left_out_nodes_dated).
"""
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from spark_jobs.graph.companyfacts import read_companyfacts
from spark_jobs.graph.config import JobConfig
from spark_jobs.graph.graph_store import write_graph_store
from spark_jobs.graph.minute_quotes import read_minute_quotes
from spark_jobs.graph.splits import read_splits
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
    BLS_ENRICHMENT,
    MARKET_QUOTES,
    SEC_COMPANYFACTS,
    SNAPSHOT_NODE_TYPE_PREFIX,
    SOURCE_TEMPORAL,
    classify_edge_origin,
)
from spark_jobs.utils.spark_rdf_utils import (
    CODE_PREDICATES,
    IDENTIFIER_PREDICATES,
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

# Node types no table holds, with every edge, value and store triple of theirs.
# See the module docstring.
LEFT_OUT_NODE_TYPES = (
    prefixed_local_name(str(SEC_COMPANYFACTS.CompanyFact)),
)

# The day rung of the period spine (enrichment/temporal_unifier.py): the source
# day a dated node points at, and the unified day that is sameAs it.
SOURCE_DAY_TYPE = prefixed_local_name(str(SOURCE_TEMPORAL.SourceDay))
UNIFIED_DAY_TYPE = prefixed_local_name(str(BLS_ENRICHMENT.UnifiedDay))

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


def _write(
    df: DataFrame,
    root: str,
    table: str,
    day: str,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Write one day of one table. ``options`` are added to the Parquet
    writer's."""
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
            .options(**(options or {}))
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


def days_only_left_out_nodes_dated(edges_df: DataFrame) -> Dict[str, List[int]]:
    """The period days that only nodes of LEFT_OUT_NODE_TYPES were dated to, as
    {node_type: [node_id]}.

    A source day goes when every node pointing at it is left out, and a unified
    day when every source day it is sameAs goes. Kept, nothing would point at
    them: on 2026-10-05, 213 of the 277 source days were dated only by a
    snapshot fact. Read off ``edges_df`` over every node, left-out ones too, and
    small: a few hundred days.
    """
    # One row per source day: whether every node pointing at it is left out
    # (null when only its unified day does), and the unified days sameAs it.
    is_unified = F.col("src_type") == UNIFIED_DAY_TYPE
    days = (
        edges_df
        .filter(F.col("dst_type") == SOURCE_DAY_TYPE)
        .groupBy("dst_id")
        .agg(
            F.min(
                F.when(
                    ~is_unified,
                    F.col("src_type").isin(list(LEFT_OUT_NODE_TYPES)).cast("int"),
                )
            ).alias("_only_left_out"),
            F.collect_set(F.when(is_unified, F.col("src_id"))).alias("_unified"),
        )
        .collect()
    )
    source_days = {row["dst_id"] for row in days if row["_only_left_out"] == 1}
    if not source_days:
        return {}

    stands_for: Dict[int, set] = {}
    for row in days:
        for unified in row["_unified"]:
            stands_for.setdefault(unified, set()).add(row["dst_id"])
    unified_days = {
        unified for unified, sources in stands_for.items()
        if sources <= source_days
    }

    left_out = {SOURCE_DAY_TYPE: sorted(source_days)}
    if unified_days:
        left_out[UNIFIED_DAY_TYPE] = sorted(unified_days)
    return left_out


def kept_node(left_out_days: Dict[str, List[int]]) -> Callable[[str, str], F.Column]:
    """``kept(type_column, id_column)``: whether a node is in the tables, by its
    type and id. Not one of LEFT_OUT_NODE_TYPES, and not one of the days
    ``days_only_left_out_nodes_dated`` found."""
    def kept(type_column: str, id_column: str) -> F.Column:
        keep = ~F.col(type_column).isin(list(LEFT_OUT_NODE_TYPES))
        for node_type, ids in left_out_days.items():
            keep = keep & ~(
                (F.col(type_column) == node_type) & F.col(id_column).isin(ids)
            )
        return keep
    return kept


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
            # An ID or a code is never a number, however it parses, so a CUSIP
            # stays a string in snapshots/ instead of a double that drops its
            # letters and leading zeros (#456), and a CIK or a form type is
            # not a number in facts/ (#460).
            (
                numeric_literal_expr("object").isNotNull()
                & ~F.col("predicate").isin(
                    sorted(IDENTIFIER_PREDICATES | CODE_PREDICATES)
                )
            ).alias("is_numeric"),
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
def store_triples(
    triples_df: DataFrame,
    node_id_df: DataFrame,
    left_out_day_uris: Sequence[str] = (),
) -> DataFrame:
    """Every triple describing something other than a market snapshot or a
    node of the LEFT_OUT_NODE_TYPES.

    By SUBJECT: a triple belongs to the node it describes. So an edge FROM a
    snapshot into a company is a snapshot row and stays out, while the
    company's own triples stay in -- the same line ``facts/`` and
    ``snapshots/`` draw, applied to whole triples rather than to literals.
    ``node_id_df`` still holds the left-out nodes, so their triples are found
    the same way. ``left_out_day_uris`` are the days only left-out nodes
    were dated to, and a triple pointing at one goes too: the month's
    coversDay link to it would otherwise point at nothing.

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
    kept_out = (
        node_id_df
        .filter(
            _is_snapshot("node_type")
            | F.col("node_type").isin(list(LEFT_OUT_NODE_TYPES))
        )
        .select(F.col("uri").alias("_kept_out_uri"))
    )

    stored = (
        triples_df
        .join(
            kept_out,
            triples_df["subject"] == kept_out["_kept_out_uri"],
            "left_anti",
        )
        .select("subject", "predicate", "object")
    )
    if left_out_day_uris:
        days = list(left_out_day_uris)
        stored = stored.filter(
            ~F.col("subject").isin(days) & ~F.col("object").isin(days)
        )
    return stored


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
# Market coverage
# ============================================
_STOCK_SNAPSHOT = str(MARKET_QUOTES.EquitySnapshot)
_OPTION_SNAPSHOT = str(MARKET_QUOTES.OptionSnapshot)
_CAPTURE_TIME = str(MARKET_QUOTES.captureTime)
_UNDERLYING_SYMBOL = str(MARKET_QUOTES.underlyingSymbol)
_SYMBOL = str(MARKET_QUOTES.symbol)
_REFERS_TO_COMPANY = str(BLS_ENRICHMENT.refersToCompany)


def market_coverage(
    triples_df: DataFrame, constituents: Optional[int] = None
) -> Dict[str, Any]:
    """How complete the day's market data was. #452

    A market day passes the nightly's source check with one snapshot under its
    folder, so a partial day builds and publishes like any other. On 2026-10-16
    upstream fetched the option chains of 138 of the 503 stocks, and in that
    day's graph the other 365 look like stocks that have no options. These
    counts are where the difference shows. What is too thin is a reader's call,
    so there is no threshold here.

    A snapshot is one capture time, which the stock and option rows taken in it
    share. For each: ``stocks``, the stock rows; ``stocks_with_options``, the
    distinct underlying symbols its options name, so an option naming none
    counts in ``option_rows`` only; and ``option_rows``. The day's three are
    each the lowest across its snapshots, which is what the day marker carries.
    ``constituents`` is how many stocks the day's constituents CSV lists, to
    read the others against, or None when the CSV was not read.

    ``stocks_without_company`` names, sorted, the stocks none of whose
    snapshots links to a company: a ticker the company map could not resolve,
    which nothing else in a run shows. BRK/B and BF/B were two on every day
    checked, 2026-10-05 and 2026-10-16, because the CSV's Symbol spells them
    BRK.B and BF.B. #461

    One pass over five predicates, collected at one row per snapshot: 20 on a
    full day.
    """
    snapshots = (
        triples_df
        .filter(
            (
                (F.col("predicate") == RDF_TYPE)
                & F.col("object").isin(_STOCK_SNAPSHOT, _OPTION_SNAPSHOT)
            )
            | F.col("predicate").isin(
                _CAPTURE_TIME, _UNDERLYING_SYMBOL, _SYMBOL, _REFERS_TO_COMPANY
            )
        )
        .groupBy("subject")
        .agg(*[
            F.max(F.when(F.col("predicate") == predicate, F.col("object")))
            .alias(name)
            for name, predicate in (
                ("kind", RDF_TYPE),
                ("capture_time", _CAPTURE_TIME),
                ("underlying", _UNDERLYING_SYMBOL),
                ("symbol", _SYMBOL),
                ("company", _REFERS_TO_COMPANY),
            )
        ])
        .filter(F.col("kind").isNotNull() & F.col("capture_time").isNotNull())
    )

    option = F.col("kind") == _OPTION_SNAPSHOT
    stock = F.col("kind") == _STOCK_SNAPSHOT
    rows = collect_sorted(
        snapshots
        .groupBy("capture_time")
        .agg(
            F.sum(stock.cast("long")).alias("stocks"),
            F.countDistinct(F.when(option, F.col("underlying")))
            .alias("stocks_with_options"),
            F.sum(option.cast("long")).alias("option_rows"),
            F.collect_set(F.when(stock, F.col("symbol"))).alias("named"),
            F.collect_set(
                F.when(stock & F.col("company").isNotNull(), F.col("symbol"))
            ).alias("linked"),
        )
    )
    by_snapshot = [
        {
            "capture_time": row["capture_time"],
            "stocks": row["stocks"],
            "stocks_with_options": row["stocks_with_options"],
            "option_rows": row["option_rows"],
        }
        for row in rows
    ]
    named = {symbol for row in rows for symbol in row["named"]}
    linked = {symbol for row in rows for symbol in row["linked"]}

    def lowest(name):
        return min((snapshot[name] for snapshot in by_snapshot), default=0)

    return {
        "constituents": constituents,
        "snapshots": len(by_snapshot),
        "stocks": lowest("stocks"),
        "stocks_with_options": lowest("stocks_with_options"),
        "option_rows": lowest("option_rows"),
        "stocks_without_company": sorted(named - linked),
        "by_snapshot": by_snapshot,
    }


SPLITS_SCHEMA = (
    "ticker string, ratio string, shares_after double, shares_before double, "
    "split_date date"
)


def write_splits(
    spark: SparkSession, splits: List[Dict[str, Any]], root: str, day: str
) -> str:
    """The day's stock splits, as ``read_splits`` returned them.

    Written even when there are none. An empty partition is how a reader tells a
    day with no splits from a day the feed was not read for, which has no
    partition at all.
    """
    rows = [
        (
            split["ticker"], split["ratio"], split["shares_after"],
            split["shares_before"], split["split_date"],
        )
        for split in splits
    ]
    # One file: a few rows a day, already sorted by read_splits.
    frame = spark.createDataFrame(rows, SPLITS_SCHEMA).coalesce(1)
    return _write(frame, root, "splits", day)


# In companyfacts.COLUMNS order, which read_companyfacts' rows follow.
COMPANYFACTS_SCHEMA = (
    "cik string, entity_name string, taxonomy string, concept string, "
    "unit string, period_start date, period_end date, value double, fy int, "
    "fp string, form string, accn string, filed date, status string, "
    "first_filed date, prior_value double, property string, length string"
)


def write_companyfacts(
    spark: SparkSession, rows: List[Tuple], root: str, day: str
) -> str:
    """The SEC numbers filed on the day, as ``read_companyfacts`` returned them.

    Written even when there are none, as ``splits/`` is: an empty partition is a
    day nothing was filed, and no partition is a day the history was not read.
    """
    # One file: 11,919 rows on the busiest day of 2026, already sorted.
    frame = spark.createDataFrame(rows, COMPANYFACTS_SCHEMA).coalesce(1)
    return _write(frame, root, "companyfacts", day)


# In minute_quotes.COLUMNS order, which read_minute_quotes' rows follow.
MINUTE_QUOTES_SCHEMA = (
    "ticker string, capture_time string, quote_time string, "
    "trade_time string, bid double, ask double, bid_size long, ask_size long, "
    "last double, mark double, open double, high double, low double, "
    "prev_close double, volume long, net_change double, "
    "net_percent_change double"
)

# The Parquet writer's row group size for minute_quotes/. A full day is one
# file, and in one row group a reader after one ticker reads all of it. At 1 MB
# a synthetic full day had 37 row groups of about 14 stocks each, and took
# 8.4 MB against 7.1 MB in one.
MINUTE_QUOTES_ROW_GROUP_BYTES = 1024 * 1024


def write_minute_quotes(
    spark: SparkSession, rows: List[Tuple], root: str, day: str
) -> str:
    """The constituents' one-minute quotes, as ``read_minute_quotes`` returned
    them: one row per stock per minute, sorted by ticker, then minute.

    Written even when there are none, as ``splits/`` is. A day with no
    partition is a day the feed was not read.
    """
    # One file, already sorted: about 196,000 rows on a full day.
    frame = spark.createDataFrame(rows, MINUTE_QUOTES_SCHEMA).coalesce(1)
    return _write(
        frame, root, "minute_quotes", day,
        {"parquet.block.size": str(MINUTE_QUOTES_ROW_GROUP_BYTES)},
    )


# ============================================
# The whole set
# ============================================
def _utc_stamp() -> str:
    """Now, as YYYY-MM-DDTHH:MM:SSZ: the way the day marker's `published` is."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WrittenTables(dict):
    """What write_query_tables wrote: table -> path, and when it wrote them.

    A dict, so every reader of the table -> path map reads it as before.
    ``started`` and ``finished`` are when the run began writing the day's tables
    and when it had written the last, or None when it wrote none. The job
    manifest carries them to the day's marker, beside a ``published`` that
    belongs to the upload, not to the tables. #471
    """

    def __init__(
        self,
        paths: Optional[Dict[str, str]] = None,
        started: Optional[str] = None,
        finished: Optional[str] = None,
    ):
        super().__init__(paths or {})
        self.started = started
        self.finished = finished


def write_query_tables(
    spark: SparkSession,
    triples_df: DataFrame,
    config: JobConfig,
    s3_client=None,
) -> WrittenTables:
    """Write every query table for this run's day. Returns table -> path.

    Returns an empty map with no times, and writes nothing, when the run asked
    for no tables or cannot say which day its data describes. Neither is a
    failure: the graph the run builds is unaffected either way. Otherwise the
    map's ``started`` and ``finished`` bracket the writes.

    ``s3_client`` is for the reads of the split feed, the company facts
    history and the minute feed, and tests pass a stub.
    """
    if not config.enable_query_tables:
        logger.info("Query tables disabled (--enable_query_tables false)")
        return WrittenTables()

    if not config.source_data_day:
        logger.warning(
            "No query tables: source_paths name no single day, and no "
            "--source_data_day was given, so there is no day to write them "
            "under"
        )
        return WrittenTables()

    # Past the guards, so a run that writes no tables records no times.
    started = _utc_stamp()
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
        # Over every node, left-out ones too: the days only they were dated to
        # are read off these edges. Ids are numbered within a type, so leaving
        # nodes out moves no other node's id.
        edges_df = resolve_edges(triples_df, node_id_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            left_out_days = days_only_left_out_nodes_dated(edges_df)
            for node_type in LEFT_OUT_NODE_TYPES:
                count = node_counts.pop(node_type, 0)
                if count:
                    logger.info(
                        f"  Leaving {count:,} {node_type} nodes out of the tables"
                    )
            for node_type, ids in left_out_days.items():
                node_counts[node_type] -= len(ids)
                logger.info(
                    f"  Leaving {len(ids):,} {node_type} nodes out of the "
                    f"tables: only left-out nodes were dated to them"
                )
            kept = kept_node(left_out_days)

            # The tables' nodes. edges/ and facts/ keep only rows whose ends or
            # subject are among them, so a left-out node's edges and values go
            # with it.
            table_nodes = node_id_df.filter(kept("node_type", "node_id"))
            written["nodes"] = write_nodes(table_nodes, root, day)

            table_edges = edges_df.filter(
                kept("src_type", "src_id") & kept("dst_type", "dst_id")
            )
            edge_types = count_edge_types(table_edges)
            check_registered_names(
                "Relations",
                (row["relation"] for row in edge_types),
                allow_unregistered,
                lambda names: _predicates_named(triples_df, names),
            )
            written["edges"] = write_edges(table_edges, root, day)
            written["edge_types"] = write_edge_types(
                spark, edge_types, root, day
            )
        finally:
            edges_df.unpersist()

        left_out_day_uris = [
            row["uri"]
            for row in node_id_df
            .filter(~kept("node_type", "node_id"))
            .filter(~F.col("node_type").isin(list(LEFT_OUT_NODE_TYPES)))
            .select("uri")
            .collect()
        ] if left_out_days else []

        # One frame, two shapes: the market rows pivot wide and the rest stay
        # long. Cached because both readings scan it.
        literals_df = literals(triples_df, table_nodes).persist(
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
            store_triples(triples_df, node_id_df, left_out_day_uris),
            table_path(root, "graph", day),
        )
        if store:
            written["graph"] = store
    finally:
        node_id_df.unpersist()

    # Not from the triples: the split feed is read on its own. None means it was
    # not read for this day, which read_splits has already logged.
    splits = read_splits(
        config.stock_splits_bucket, config.stock_splits_prefix, day, s3_client
    )
    if splits is not None:
        written["splits"] = write_splits(spark, splits, root, day)

    # Read on its own too, and None in the same cases.
    facts = read_companyfacts(
        config.companyfacts_history_bucket,
        config.companyfacts_history_prefix,
        day,
        s3_client,
    )
    if facts is not None:
        written["companyfacts"] = write_companyfacts(spark, facts, root, day)

    # The minute feed too. None means the day was not read, which
    # read_minute_quotes has already logged.
    minutes = read_minute_quotes(
        config.minute_quotes_bucket, config.minute_quotes_prefix, day, s3_client
    )
    if minutes is not None:
        written["minute_quotes"] = write_minute_quotes(spark, minutes, root, day)
    finished = _utc_stamp()

    logger.info(
        f"Wrote {len(written)} query tables for {day} "
        f"({sum(node_counts.values()):,} nodes across "
        f"{len(node_counts)} types), {started} to {finished}"
    )
    for table, path in written.items():
        logger.info(f"    {table}: {path}")

    return WrittenTables(written, started, finished)
