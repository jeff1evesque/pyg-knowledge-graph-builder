"""The query tables, from a real run over the committed fixtures.

The unit tests build the tables from a handful of hand-written triples, which
proves the shapes and not much else. What only a real run can show is that the
tables agree with the graph the same run built: ``nodes/`` holds the nodes
``graph_schema.json`` counted, and ``edges/`` holds the edges it counted. Those
two numbers come from separate code -- the tables resolve edges themselves
because ``EdgeMapper`` imports torch and goes on to build tensors -- so they are
the check that keeps the two implementations saying the same thing.

The fixture paths carry no ``day=`` partition, so these runs state
``--source_data_day``. That is the same path an operator takes when rebuilding a
day from an archive laid out some other way.

Marked ``e2e``: heavy, excluded from the fast suite, run with ``-m e2e``.
"""

import json
from pathlib import Path

import pytest

from spark_jobs.graph.tables import table_path
from spark_jobs.pyg_builder.metadata_writer import derive_metadata_prefix

pytestmark = pytest.mark.e2e

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "e2e"
TURTLE_PARQUET = FIXTURES / "turtle_parquet"

DAY = "2099-01-02"
TABLES = ("nodes", "edges", "edge_types", "facts", "entities", "snapshots")


def _source_paths():
    dirs = sorted({str(p.parent) for p in TURTLE_PARQUET.rglob("*.parquet")})
    assert dirs, f"no turtle-parquet fixtures under {TURTLE_PARQUET}"
    return ",".join(dirs)


def _config(tmp_path, **overrides):
    from spark_jobs.graph.config import JobConfig

    args = {
        "mode": "full",
        "source_format": "turtle_parquet",
        "turtle_column": "triples",
        "time_period": "2099-01",
        "parquet_partitions": "2",
        "local_work_dir": str(tmp_path),
        "source_paths": _source_paths(),
        "source_data_day": DAY,
    }
    args.update({key: str(value) for key, value in overrides.items()})
    return JobConfig(args)


def _schema(config):
    prefix = Path(derive_metadata_prefix(config.pyg_output_path))
    return json.loads((prefix / "graph_schema.json").read_text())


def _rows(spark, config, table):
    return spark.read.parquet(table_path(config.query_tables_path, table, DAY))


@pytest.fixture(scope="module")
def run(spark, tmp_path_factory):
    """One real full-pipeline run, shared by the assertions below."""
    from spark_jobs.build_graph import execute_full_pipeline

    tmp_path = tmp_path_factory.mktemp("query-tables-e2e")
    config = _config(tmp_path)
    result = execute_full_pipeline(config, spark, s3_client=None)
    return config, result


def test_a_default_run_writes_every_table_and_the_store(run):
    config, result = run

    for table in TABLES:
        assert Path(table_path(config.query_tables_path, table, DAY)).is_dir(), (
            f"{table}/ was not written"
        )
    assert Path(table_path(config.query_tables_path, "graph", DAY)).is_dir()

    # And the run reports them, so the manifest and index can name them.
    assert set(result["query_tables"]) >= set(TABLES) | {"graph"}


def test_nodes_holds_the_nodes_the_graph_schema_counted(spark, run):
    config, _ = run
    assert (
        _rows(spark, config, "nodes").count()
        == _schema(config)["summary"]["total_nodes"]
    )


def test_edges_holds_the_edges_the_graph_schema_counted(spark, run):
    """The two resolve edges through separate code. This is what keeps them
    from drifting -- a predicate excluded in one and not the other shows up
    here as a mismatch rather than as a silently thinner published graph."""
    config, _ = run
    schema = _schema(config)
    counted = sum(
        entry["count"] for entry in schema["edge_types"].values()
    )

    assert _rows(spark, config, "edges").count() == counted
    assert counted == schema["summary"]["total_edges"]


def test_edge_types_describes_every_edge_type_the_schema_names(spark, run):
    config, _ = run
    schema = _schema(config)

    published = {
        (row["src_type"], row["relation"], row["dst_type"]): row
        for row in _rows(spark, config, "edge_types").collect()
    }
    described = {
        (entry["src_type"], entry["relation"], entry["dst_type"]): entry
        for entry in schema["edge_types"].values()
    }

    assert set(published) == set(described)
    for key, entry in described.items():
        assert published[key]["count"] == entry["count"], key
        assert published[key]["origin"] == entry["origin"], key
        assert published[key]["predicate_uri"] == entry["predicate_uri"], key


def test_the_store_holds_no_market_data(spark, run):
    """No market NODE, which is what "market never enters the store" means.

    Market terms do appear, and that is not the same thing: the vocabulary
    statements -- the derived subClassOf hierarchy, the observed domains and
    ranges, the provenance markers -- have predicate and class URIs as their
    subjects rather than entities, so they are in no node table and belong to
    no source's data. They are the store's schema, and they stay.
    """
    import pyoxigraph

    config, _ = run
    store = pyoxigraph.Store.read_only(
        table_path(config.query_tables_path, "graph", DAY)
    )

    subjects = {
        quad.subject.value
        for quad in store.quads_for_pattern(None, None, None)
    }
    assert subjects, "the store is empty"

    market_nodes = {
        row["uri"]
        for row in _rows(spark, config, "nodes").collect()
        if row["node_type"].startswith("market_")
    }
    assert market_nodes, "the fixtures carry no market nodes to exclude"
    assert market_nodes.isdisjoint(subjects)


def test_the_flag_off_leaves_the_existing_artifact_set_unchanged(
    spark, tmp_path
):
    """Off, a run produces exactly what it produced before any of this existed
    -- no tables, no store, no failure."""
    from spark_jobs.build_graph import execute_full_pipeline

    config = _config(tmp_path, enable_query_tables="false")
    result = execute_full_pipeline(config, spark, s3_client=None)

    assert result["query_tables"] == {}
    assert not Path(config.query_tables_path).exists()
    assert Path(config.pyg_output_path).exists()
    assert (
        Path(derive_metadata_prefix(config.pyg_output_path))
        / "graph_schema.json"
    ).exists()


def test_a_node_filter_narrows_the_graph_and_not_the_tables(spark, tmp_path):
    """A run may build its .pt over a subset and still has to publish every
    source. Weather is the standing example: it earns little in a model and
    answers real questions in a table."""
    from spark_jobs.build_graph import execute_full_pipeline

    everything = _config(tmp_path / "all")
    execute_full_pipeline(everything, spark, s3_client=None)
    weather_types = {
        row["node_type"]
        for row in _rows(spark, everything, "nodes").collect()
        if row["node_type"].startswith(("weather_", "alert_", "cap_"))
    }
    assert weather_types, "the fixtures carry no weather nodes to exclude"

    kept = sorted(
        node_type
        for node_type in _schema(everything)["node_types"]
        if node_type not in weather_types
    )
    narrowed = _config(
        tmp_path / "narrowed",
        pyg_config=json.dumps({"node_types": kept}),
    )
    execute_full_pipeline(narrowed, spark, s3_client=None)

    assert weather_types.isdisjoint(_schema(narrowed)["node_types"])
    published = {
        row["node_type"]
        for row in _rows(spark, narrowed, "nodes").collect()
    }
    assert weather_types <= published, (
        "a node filter reached the tables; they must cover every source the "
        "run read, whatever the .pt was built over"
    )
