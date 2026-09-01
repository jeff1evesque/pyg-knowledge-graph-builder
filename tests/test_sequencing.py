"""
Unit tests for the shared windowed sequencing helper.

The defect these pin down (#360): every source builds its precedes chains with
a window and lead(), and none of them removed duplicate entities first. A
measurement that appeared twice in one partition sorted next to its own copy,
so lead() returned it to itself and the step emitted "X precedes X".

The invariant every test here defends is simple: no step may make a node its
own object.
"""
import pytest

from pyspark.sql import functions as F

from spark_jobs.enrichment.intra_source.sequencing import (
    sequence_to_triples,
    sequence_within_partitions,
)

PRECEDES = "https://example.org/ontology/test/precedes"


@pytest.fixture
def make_rows(spark):
    """Build a (category, entity, sort_key) DataFrame from 3-tuples."""
    def _make(rows):
        return spark.createDataFrame(
            rows, schema="category STRING, entity STRING, sort_key STRING"
        )

    return _make


def _pairs(df):
    """Collect (entity, next_entity) pairs as a sorted list."""
    return sorted(
        (r["entity"], r["next_entity"]) for r in df.collect()
    )


def _sequence(df):
    return sequence_within_partitions(
        df,
        entity_col="entity",
        partition_cols=["category"],
        order_cols=["sort_key"],
    )


def test_chain_follows_sort_order(make_rows):
    """Three measurements in one category link in date order."""
    df = make_rows([
        ("jobs", "june", "2026-06"),
        ("jobs", "august", "2026-08"),
        ("jobs", "july", "2026-07"),
    ])

    assert _pairs(_sequence(df)) == [("july", "august"), ("june", "july")]


def test_duplicate_entity_does_not_precede_itself(make_rows):
    """The #360 defect: a repeated entity must not link to its own copy."""
    df = make_rows([
        ("jobs", "july", "2026-07"),
        ("jobs", "july", "2026-07"),
        ("jobs", "august", "2026-08"),
    ])

    pairs = _pairs(_sequence(df))

    assert ("july", "july") not in pairs
    assert pairs == [("july", "august")]


def test_duplicate_keeps_first_in_sort_order(make_rows):
    """Which copy survives is fixed, so the chain is the same every run.

    dropDuplicates would keep an arbitrary copy; when the copies disagree on
    the ordering column that would make the output vary between runs.
    """
    df = make_rows([
        ("jobs", "july", "2026-07-b"),
        ("jobs", "july", "2026-07-a"),
        ("jobs", "august", "2026-08"),
    ])

    rows = _sequence(df).collect()

    assert len(rows) == 1
    assert rows[0]["sort_key"] == "2026-07-a"


def test_partitions_do_not_cross(make_rows):
    """A measurement never precedes one from another category."""
    df = make_rows([
        ("jobs", "june", "2026-06"),
        ("jobs", "july", "2026-07"),
        ("prices", "june", "2026-06"),
        ("prices", "july", "2026-07"),
    ])

    result = _sequence(df)

    assert result.count() == 2
    for row in result.collect():
        assert row["entity"] != row["next_entity"]


def test_lone_row_has_no_successor(make_rows):
    """A partition of one produces no edge rather than a self-edge."""
    df = make_rows([("jobs", "july", "2026-07")])

    assert _sequence(df).count() == 0


def test_repeated_entity_alone_in_partition_yields_nothing(make_rows):
    """The same entity three times is one measurement, not a chain."""
    df = make_rows([
        ("jobs", "july", "2026-07"),
        ("jobs", "july", "2026-07"),
        ("jobs", "july", "2026-07"),
    ])

    assert _sequence(df).count() == 0


def test_no_self_pairs_on_heavily_duplicated_input(make_rows):
    """The invariant holds when every entity is duplicated."""
    df = make_rows([
        ("jobs", entity, key)
        for entity, key in [
            ("june", "2026-06"), ("july", "2026-07"), ("august", "2026-08"),
        ]
        for _ in range(3)
    ])

    result = _sequence(df)

    assert result.count() == 2
    for row in result.collect():
        assert row["entity"] != row["next_entity"]


def test_join_fanout_does_not_create_self_edges(spark):
    """Mirrors the real cause: an entity carrying two years fans out on join.

    base_enricher.py resolves the year with an inner join on entity. An entity
    with two year triples becomes two rows, both landing in one partition.
    """
    entities = spark.createDataFrame(
        [("jobs", "july"), ("jobs", "august")],
        schema="category STRING, entity STRING",
    )
    years = spark.createDataFrame(
        [("july", "2026"), ("july", "2025"), ("august", "2026")],
        schema="entity STRING, year_value STRING",
    )

    fanned_out = entities.join(years, "entity", "inner")
    assert fanned_out.count() == 3

    result = sequence_within_partitions(
        fanned_out,
        entity_col="entity",
        partition_cols=["category"],
        order_cols=["year_value"],
    )

    for row in result.collect():
        assert row["entity"] != row["next_entity"]


def test_multiple_partition_columns(make_rows):
    """Partitioning on several columns keeps each chain separate."""
    df = make_rows([
        ("jobs", "june", "2026-06"),
        ("jobs", "july", "2026-07"),
    ]).withColumn("region", F.lit("south"))

    result = sequence_within_partitions(
        df,
        entity_col="entity",
        partition_cols=["category", "region"],
        order_cols=["sort_key"],
    )

    assert _pairs(result) == [("june", "july")]


def test_sequence_to_triples_shape(make_rows):
    """The triples wrapper emits subject/predicate/object and nothing else."""
    df = make_rows([
        ("jobs", "june", "2026-06"),
        ("jobs", "july", "2026-07"),
    ])

    result = sequence_to_triples(
        df,
        entity_col="entity",
        predicate=PRECEDES,
        partition_cols=["category"],
        order_cols=["sort_key"],
    )

    assert result.columns == ["subject", "predicate", "object"]

    rows = result.collect()
    assert len(rows) == 1
    assert rows[0]["subject"] == "june"
    assert rows[0]["predicate"] == PRECEDES
    assert rows[0]["object"] == "july"


def test_sequence_to_triples_drops_self_edges(make_rows):
    """No triple may have the same subject and object."""
    df = make_rows([
        ("jobs", "july", "2026-07"),
        ("jobs", "july", "2026-07"),
    ])

    result = sequence_to_triples(
        df,
        entity_col="entity",
        predicate=PRECEDES,
        partition_cols=["category"],
        order_cols=["sort_key"],
    )

    assert result.count() == 0
