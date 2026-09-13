"""Unit tests for the query tables (spark_jobs.graph.tables).

These are the artifacts a downstream service reads, so what they pin is mostly
about what must NOT reach them:

  * a run building its ``.pt`` over a subset of node types still writes every
    source's rows to the tables. That is the whole reason the tables build
    their own node table instead of reusing the run's
  * market stays out of ``facts/`` and ``entities/`` -- it is 99.5% of the
    graph and belongs in a wide table of its own -- while staying IN ``edges/``,
    because the links between market and everything else are the point
  * a triple pointing at a literal is a fact and a triple pointing at an
    untyped URI is neither, so neither becomes an edge

Tier 4: real Spark, small frames, no cluster.
"""
import pytest

from spark_jobs.graph.config import JobConfig
from spark_jobs.graph.tables import table_path, write_query_tables
from spark_jobs.pyg_builder.naming import RDF_TYPE

ONT = "https://jefflevesque.com/ontology/"

# Types and predicates taken from the namespace table's own vocabularies, so a
# re-homed namespace shows up here as a failure rather than as a silently
# renamed column value.
CPI_INDEX = f"{ONT}cpi/Index"
OPTION_SNAPSHOT = f"{ONT}market-quotes/OptionSnapshot"
WEATHER_ALERT = f"{ONT}weather/WeatherAlert"

PRECEDES = f"{ONT}bls/precedes"
UNDERLYING_SYMBOL = f"{ONT}market-quotes/underlyingSymbol"
CONTRACT_SYMBOL = f"{ONT}market-quotes/symbol"
REFERS_TO_COMPANY = f"{ONT}market/refersToCompany"
CPI_VALUE = f"{ONT}cpi/hasValue"
STRIKE_PRICE = f"{ONT}market-quotes/strikePrice"
ALERT_HEADLINE = f"{ONT}weather/hasHeadline"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

INDEX_A = "https://ex/index/a"
INDEX_B = "https://ex/index/b"
QUOTE = "https://ex/quote/1"
QUOTE_2 = "https://ex/quote/2"
ALERT = "https://ex/alert/1"
UNTYPED = "https://ex/untyped/1"

TRIPLES = [
    (INDEX_A, RDF_TYPE, CPI_INDEX),
    (INDEX_B, RDF_TYPE, CPI_INDEX),
    (QUOTE, RDF_TYPE, OPTION_SNAPSHOT),
    (ALERT, RDF_TYPE, WEATHER_ALERT),

    # Edges: one within a source, one from market into it.
    (INDEX_A, PRECEDES, INDEX_B),
    (QUOTE, REFERS_TO_COMPANY, INDEX_A),

    # A URI object nothing ever typed. Neither an edge nor a fact.
    (INDEX_A, PRECEDES, UNTYPED),

    # Literals: one numeric, one text, one on a market node.
    (INDEX_A, CPI_VALUE, "3.75"),
    (INDEX_A, RDFS_LABEL, "All items"),
    (ALERT, ALERT_HEADLINE, "Coastal flood warning"),
    (QUOTE, STRIKE_PRICE, "100.5"),
    (QUOTE, UNDERLYING_SYMBOL, "MSFT"),
    (QUOTE, CONTRACT_SYMBOL, "MSFT260918C00100500"),
    (QUOTE_2, RDF_TYPE, OPTION_SNAPSHOT),
    (QUOTE_2, UNDERLYING_SYMBOL, "AAPL"),
    (QUOTE_2, CONTRACT_SYMBOL, "AAPL260918C00200000"),
]

DAY = "2026-09-12"


def _config(tmp_path, **overrides):
    args = {
        "mode": "enrichment_only",
        "source_paths": "s3a://b/raw/year=2026/month=09/day=12/",
        "source_format": "turtle_parquet",
        "local_work_dir": str(tmp_path),
        "time_period": "2026-09",
    }
    args.update(overrides)
    return JobConfig(args)


@pytest.fixture(scope="module")
def written(spark, tmp_path_factory):
    """One write of the whole set, read back as {table: [row dicts]}."""
    tmp_path = tmp_path_factory.mktemp("tables")
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    paths = write_query_tables(spark, triples, _config(tmp_path))
    return {
        table: [row.asDict() for row in spark.read.parquet(path).collect()]
        for table, path in paths.items()
    }


# ======================================================================
# nodes/
# ======================================================================

def test_nodes_carries_every_typed_entity_with_its_id(written):
    rows = {row["uri"]: row for row in written["nodes"]}
    assert set(rows) == {INDEX_A, INDEX_B, QUOTE, QUOTE_2, ALERT}
    assert rows[INDEX_A]["node_type"] == "cpi_Index"
    assert rows[QUOTE]["node_type"] == "market_quotes_OptionSnapshot"


def test_node_ids_are_zero_based_within_a_type(written):
    """Ids come from row_number() over a uri-ordered window per type, which is
    what makes them day-scoped: the same URI gets a different id as soon as the
    node set changes, so an edge is only ever joinable to its own day."""
    cpi = sorted(
        row["node_id"] for row in written["nodes"]
        if row["node_type"] == "cpi_Index"
    )
    assert cpi == [0, 1]


# ======================================================================
# edges/
# ======================================================================

def test_edges_resolve_both_endpoints_through_the_node_table(written):
    ids = {
        row["uri"]: (row["node_type"], row["node_id"])
        for row in written["nodes"]
    }
    edges = {
        (row["src_type"], row["src_id"], row["relation"],
         row["dst_type"], row["dst_id"])
        for row in written["edges"]
    }

    assert (
        *ids[INDEX_A], "bls_enrichment_precedes", *ids[INDEX_B]
    ) in edges
    assert (
        *ids[QUOTE], "market_enrichment_refersToCompany", *ids[INDEX_A]
    ) in edges
    assert len(edges) == 2


def test_a_triple_pointing_at_an_untyped_uri_is_neither_edge_nor_fact(written):
    """Nothing typed it, so it is no node and the triple is no edge. It is not
    a literal value either -- it is a pointer at something absent, and in a
    `value` column a consumer would read it as a name. It drops out of both
    tables, which is a deliberate omission and worth pinning."""
    assert len([
        row for row in written["edges"]
        if row["relation"] == "bls_enrichment_precedes"
    ]) == 1
    assert not [
        row for row in written["facts"] if row["value"] == UNTYPED
    ]


def test_market_edges_are_kept(written):
    """Market is excluded from the long tables, NOT from the graph structure.
    The link from a quote to the company it refers to is the bridge the whole
    pipeline exists to build."""
    assert [
        row for row in written["edges"]
        if row["relation"] == "market_enrichment_refersToCompany"
    ]


# ======================================================================
# edge_types/
# ======================================================================

def test_edge_types_describes_every_relation_in_edges(written):
    in_edges = {
        (row["src_type"], row["relation"], row["dst_type"])
        for row in written["edges"]
    }
    described = {
        (row["src_type"], row["relation"], row["dst_type"])
        for row in written["edge_types"]
    }
    assert described == in_edges


def test_edge_types_recovers_the_predicate_and_its_origin(written):
    """An edge is only interpretable through this table once its run has
    expired -- the relation name alone says neither what predicate it came from
    nor whether the link was observed or inferred."""
    row = next(
        row for row in written["edge_types"]
        if row["relation"] == "bls_enrichment_precedes"
    )
    assert row["predicate_uri"] == PRECEDES
    assert row["origin"] == "enrichment"
    assert row["count"] == 1
    assert row["relation_group"] == "bls_enrichment_precedes"


# ======================================================================
# facts/
# ======================================================================

def test_facts_carries_non_market_literals_and_marks_the_numbers(written):
    facts = {
        (row["uri"], row["predicate"]): row for row in written["facts"]
    }
    assert facts[(INDEX_A, CPI_VALUE)]["value"] == "3.75"
    assert facts[(INDEX_A, CPI_VALUE)]["is_numeric"] is True
    assert facts[(INDEX_A, RDFS_LABEL)]["is_numeric"] is False
    assert facts[(INDEX_A, CPI_VALUE)]["predicate_name"] == "cpi_hasValue"


def test_facts_excludes_market(written):
    """9.3M market rows a day in a long table would swamp the 1.4M rows every
    other source contributes, and market is better served by a wide one."""
    assert not [
        row for row in written["facts"]
        if row["node_type"].startswith("market_")
    ]
    assert not [
        row for row in written["facts"] if row["predicate"] == STRIKE_PRICE
    ]


def test_facts_covers_every_non_market_source(written):
    """Long format is what lets one shape hold every source. Weather earns
    little in a model -- 0.076% of nodes, no edge into market -- and still has
    to come out in a table."""
    assert {row["node_type"] for row in written["facts"]} == {
        "cpi_Index", "weather_WeatherAlert"
    }


def test_facts_does_not_restate_the_type(written):
    """rdf:type is what nodes/ answers, for every node, in three columns."""
    assert not [
        row for row in written["facts"] if row["predicate"] == RDF_TYPE
    ]


# ======================================================================
# entities/
# ======================================================================

def test_entities_assembles_the_text_a_node_carries(written):
    rows = {row["uri"]: row["text"] for row in written["entities"]}
    assert rows[ALERT] == "weather_hasHeadline: Coastal flood warning"
    assert rows[INDEX_A] == "rdfs_label: All items"


def test_entities_leaves_out_nodes_carrying_no_text(written):
    """A number is not something to embed. INDEX_B carries nothing at all and
    the quote carries only a strike price."""
    assert set(row["uri"] for row in written["entities"]) == {INDEX_A, ALERT}


# ======================================================================
# What the run's own filters must not do to the tables
# ======================================================================

def test_node_filters_do_not_reach_the_tables(spark, tmp_path):
    """A run may build its .pt over one node type and still has to publish
    every source. If this ever fails, the tables are being built from the
    filtered frame and a whole source has silently stopped being published.
    """
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    config = _config(
        tmp_path, pyg_config='{"node_types": ["cpi_Index"]}'
    )
    assert config.pyg_config["node_types"] == ["cpi_Index"]

    paths = write_query_tables(spark, triples, config)

    node_types = {
        row["node_type"]
        for row in spark.read.parquet(paths["nodes"]).collect()
    }
    assert "weather_WeatherAlert" in node_types
    assert "market_quotes_OptionSnapshot" in node_types


# ======================================================================
# The day partition
# ======================================================================

def test_tables_are_written_under_the_data_day(spark, tmp_path):
    config = _config(tmp_path)
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    paths = write_query_tables(spark, triples, config)

    assert paths["facts"] == table_path(
        config.query_tables_path, "facts", DAY
    )
    # Read the parent, and the day comes back as a column to filter and
    # deduplicate on. That is the whole reason it is a directory.
    parent = spark.read.parquet(f"{config.query_tables_path}/facts")
    assert "day" in parent.columns
    # Spark's partition-type inference reads day=2026-09-12 as a DATE, which is
    # what a consumer ordering by it wants.
    assert {str(row["day"]) for row in parent.select("day").collect()} == {DAY}


def test_a_uri_written_on_two_days_keeps_both_rows(spark, tmp_path):
    """No upsert, deliberately: merging into a published table means reading it
    back, which the builder identity cannot do. Both rows are kept and a
    consumer takes the newest per URI."""
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    config = _config(tmp_path)
    write_query_tables(spark, triples, config)
    write_query_tables(
        spark, triples,
        _config(
            tmp_path,
            source_paths="s3a://b/raw/year=2026/month=09/day=13/",
        ),
    )

    rows = (
        spark.read.parquet(f"{config.query_tables_path}/nodes")
        .filter("uri = '{}'".format(INDEX_A))
        .collect()
    )
    assert sorted(str(row["day"]) for row in rows) == [
        "2026-09-12", "2026-09-13"
    ]


# ======================================================================
# When nothing is written
# ======================================================================

def test_the_flag_off_writes_nothing(spark, tmp_path):
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    config = _config(tmp_path, enable_query_tables="false")

    assert write_query_tables(spark, triples, config) == {}
    assert not (tmp_path / "tables").exists()


def test_no_day_writes_nothing_rather_than_guessing_one(spark, tmp_path):
    """Every table is day-partitioned. A run whose paths name no day, and that
    was given none, has no partition to write under -- and the day it happens
    to run on is not the day its data describes."""
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    config = _config(tmp_path, source_paths="s3a://b/raw/sec/")
    assert config.source_data_day == ""

    assert write_query_tables(spark, triples, config) == {}
    assert not (tmp_path / "tables").exists()


def test_a_stated_day_supplies_the_partition_the_paths_lack(spark, tmp_path):
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    config = _config(
        tmp_path,
        source_paths="s3a://b/raw/sec/",
        source_data_day="2026-09-12",
    )

    paths = write_query_tables(spark, triples, config)
    assert paths["nodes"].endswith("day=2026-09-12")


# ======================================================================
# snapshots/
# ======================================================================

def test_snapshots_pivots_market_wide(written):
    rows = {row["uri"]: row for row in written["snapshots"]}
    assert set(rows) == {QUOTE, QUOTE_2}
    assert rows[QUOTE]["market_quotes_underlyingSymbol"] == "MSFT"
    assert rows[QUOTE]["market_quotes_strikePrice"] == 100.5


def test_a_snapshot_missing_a_property_gets_a_null_not_a_dropped_row(written):
    """A quote with no strike is still a quote. Dropping the row would lose a
    snapshot; the null says the property was not stated."""
    second = next(
        row for row in written["snapshots"] if row["uri"] == QUOTE_2
    )
    assert second["market_quotes_strikePrice"] is None
    assert second["market_quotes_underlyingSymbol"] == "AAPL"


def test_a_numeric_property_is_a_number_and_a_symbol_is_not(written):
    """Typed off the same majority rule the feature extractor classifies
    predicates by, so `strikePrice > 100` is answerable for every row rather
    than for whichever ones happened to parse."""
    row = next(r for r in written["snapshots"] if r["uri"] == QUOTE)
    assert isinstance(row["market_quotes_strikePrice"], float)
    assert isinstance(row["market_quotes_underlyingSymbol"], str)


def test_snapshots_are_sorted_by_the_underlying_ticker(spark, tmp_path):
    """Sorted partitions are what let Parquet prune row groups: a single-ticker
    filter wants 0.14% of a day, and unsorted every ticker is in every group."""
    triples = spark.createDataFrame(
        TRIPLES, "subject string, predicate string, object string"
    )
    paths = write_query_tables(spark, triples, _config(tmp_path))
    symbols = [
        row["market_quotes_underlyingSymbol"]
        for row in spark.read.parquet(paths["snapshots"]).collect()
    ]
    assert symbols == sorted(symbols)


def test_a_run_with_no_market_writes_no_snapshots(spark, tmp_path):
    """A no-market dataset is a legitimate configuration, not a failure."""
    triples = spark.createDataFrame(
        [t for t in TRIPLES if "quote" not in t[0]],
        "subject string, predicate string, object string",
    )
    paths = write_query_tables(spark, triples, _config(tmp_path))
    assert "snapshots" not in paths
    assert paths["facts"]
