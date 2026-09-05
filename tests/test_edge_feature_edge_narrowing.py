"""Regression tests for the resolved-edge scan in the edge feature phase.

Every small edge type filters the resolved edge frame down to its own
(src_type, relation, dst_type) triple, and each of those filters is a full scan
of it. The scans multiply twice over: by edge type, and again by how many times
a type's three segments reference their edge frame.

Measured on the 2026-09-04 cluster run, where the union carried 60 small types:
78,065,931,890 rows scanned in one stage, about 1,090 passes over a
71,620,121-row frame. 71 minutes, the longest single stage in the pipeline. The
60 types hold 57,811 edges between them, so better than 99.9% of those rows were
read and thrown away.

The fix narrows the edge frame once, to the types that take the batched collect,
and hands that to the encoders. It is the same move #362 made for the endpoint
property frame, on the other frame the encoders scan.

These tests pin what it has to keep: the narrow frame holds the right edges and
all of them, reading it does not go back to the full frame, a large type still
reads the full frame, and narrowing changes no feature value.
"""
import pytest
import torch

from pyspark.sql import functions as F

from spark_jobs.pyg_builder.edge_feature_extractor import EdgeFeatureExtractor

from tests.test_edge_feature_extractor import (
    CPI_INDEX,
    CPI_SERIES,
    HAS_MONTH,
    HAS_YEAR,
    PRECEDES,
    PRECEDES_REL,
    RDF_TYPE,
    _edge_features,
)

# Two more registered CPI classes, so the fixture graph carries edge types that
# differ in their endpoints rather than only in edge count.
CPI_AREA = "https://jefflevesque.com/ontology/cpi/Area"
CPI_ITEM = "https://jefflevesque.com/ontology/cpi/Item"

LARGE_KEY = ("cpi_Index", PRECEDES_REL, "cpi_Series")
SMALL_KEY = ("cpi_Area", PRECEDES_REL, "cpi_Item")

# Six edges of the large type against one of the small type, so the small side
# is well under the half-the-edges bar that decides whether narrowing pays.
LARGE_EDGES = 6

# The budget _max_edges_per_collect returns is min(chunk_edge_threshold,
# whatever maxResultSize allows), so lowering the threshold is what splits this
# fixture into one large type and one small one.
SPLIT_AT_TWO = {"edge_feature_config": {"chunk_edge_threshold": 2}}


def _edges(spark, rows):
    """A stand-in for the resolved edge frame EdgeMapper hands over."""
    return spark.createDataFrame(
        rows,
        schema="src_type STRING, src_id LONG, relation STRING, "
               "dst_type STRING, dst_id LONG, edge_idx LONG",
    )


@pytest.fixture
def edge_rows():
    """Eight edges: six of one type, one each of two others."""
    rows = [("big", i, "rel", "big_dst", i, i) for i in range(LARGE_EDGES)]
    rows.append(("small_a", 0, "rel", "small_a_dst", 0, 0))
    rows.append(("small_b", 0, "rel", "small_b_dst", 0, 0))
    return rows


class _CountingEdges:
    """The edge frame, counting how many rows get scanned.

    The counting UDF sits on the frame itself, so one full pass ticks once per
    row. It is non-deterministic so Catalyst cannot move it above the join that
    narrows the frame, which would hide the very thing this measures.
    """

    def __init__(self, spark, df):
        self.accumulator = spark.sparkContext.accumulator(0)
        acc = self.accumulator

        def _tick(v):
            acc.add(1)
            return v

        tick = F.udf(_tick, "long").asNondeterministic()
        self.df = df.withColumn("src_id", tick(F.col("src_id")))

    @property
    def ticks(self) -> int:
        return self.accumulator.value


def _mixed_graph():
    """Six edges of one temporal type and one of another, as triples."""
    rows = []
    for i in range(LARGE_EDGES):
        src, dst = f"https://ex/src{i}", f"https://ex/dst{i}"
        rows += [
            (src, RDF_TYPE, CPI_INDEX),
            (dst, RDF_TYPE, CPI_SERIES),
            (src, HAS_MONTH, str((i % 12) + 1)),
            (src, HAS_YEAR, "2020"),
            (dst, HAS_MONTH, str(((i + 1) % 12) + 1)),
            (dst, HAS_YEAR, "2020"),
            (src, PRECEDES, dst),
        ]
    rows += [
        ("https://ex/asrc", RDF_TYPE, CPI_AREA),
        ("https://ex/adst", RDF_TYPE, CPI_ITEM),
        ("https://ex/asrc", HAS_MONTH, "3"),
        ("https://ex/asrc", HAS_YEAR, "2020"),
        ("https://ex/adst", HAS_MONTH, "4"),
        ("https://ex/adst", HAS_YEAR, "2020"),
        ("https://ex/asrc", PRECEDES, "https://ex/adst"),
    ]
    return rows


# ======================================================================
# The narrowing itself
# ======================================================================

def test_narrow_frame_holds_only_the_small_types(spark, edge_rows):
    """The frame holds the small types' edges and leaves the large one out.

    Leaving the large type out is the point: it holds nearly every edge on real
    data, so narrowing to include it would be narrowing to the whole frame.
    """
    efe = EdgeFeatureExtractor(spark, {})

    narrow = efe._narrow_edges_for_small_types(
        _edges(spark, edge_rows),
        [("small_a", "rel", "small_a_dst"), ("small_b", "rel", "small_b_dst")],
        small_edges=2,
        total_edges=8,
    )

    assert narrow is not None
    got = {row["src_type"] for row in narrow.select("src_type").collect()}
    assert got == {"small_a", "small_b"}


def test_narrow_frame_keeps_every_edge_of_the_types_it_holds(
    spark, edge_rows
):
    """Narrowing drops types, never edges of a type it kept.

    A dropped edge is a feature row that silently reads as absent, so this is
    the correctness half of the change.
    """
    efe = EdgeFeatureExtractor(spark, {})
    full = _edges(spark, edge_rows)

    narrow = efe._narrow_edges_for_small_types(
        full,
        [("small_a", "rel", "small_a_dst"), ("small_b", "rel", "small_b_dst")],
        small_edges=2,
        total_edges=8,
    )

    expected = sorted(
        tuple(r) for r in full.filter(
            F.col("src_type").isin(["small_a", "small_b"])
        ).collect()
    )
    actual = sorted(tuple(r) for r in narrow.collect())

    assert actual == expected
    assert len(actual) == 2


def test_narrow_frame_keeps_the_edge_frame_schema(spark, edge_rows):
    """The encoders are written against the resolved edge frame's columns.

    A join that leaked the key frame's columns, or dropped edge_idx, would
    break them somewhere far from here.
    """
    efe = EdgeFeatureExtractor(spark, {})
    full = _edges(spark, edge_rows)

    narrow = efe._narrow_edges_for_small_types(
        full,
        [("small_a", "rel", "small_a_dst")],
        small_edges=1,
        total_edges=8,
    )

    assert narrow.columns == full.columns


def test_per_type_filters_do_not_rescan_the_full_edge_frame(
    spark, edge_rows
):
    """The point of the change: the encoders' filters scan the narrow frame.

    Before it, every per-type filter was a pass over the full edge frame, and
    the passes multiplied by edge type and by how often a type's segments
    referenced their edge frame. After it the full frame is read once, to build
    the narrow one, and the per-type filters that follow add nothing.
    """
    efe = EdgeFeatureExtractor(spark, {})
    counting = _CountingEdges(spark, _edges(spark, edge_rows))

    narrow = efe._narrow_edges_for_small_types(
        counting.df,
        [("small_a", "rel", "small_a_dst"), ("small_b", "rel", "small_b_dst")],
        small_edges=2,
        total_edges=8,
    )
    after_narrowing = counting.ticks

    assert after_narrowing > 0, "the narrowing pass never read the frame"

    # Stand in for the filters a type's three segments issue, across types.
    for _ in range(3):
        for src_type in ("small_a", "small_b"):
            narrow.filter(F.col("src_type") == src_type).count()

    assert counting.ticks == after_narrowing, (
        f"the per-type filters read the full frame again "
        f"({counting.ticks - after_narrowing} extra row scans); the settle did "
        f"not truncate the lineage, so the scans still multiply by edge type"
    )


# ======================================================================
# When narrowing is not worth it
# ======================================================================

def test_no_narrowing_when_the_small_types_hold_most_of_the_edges(
    spark, edge_rows
):
    """Narrowing costs a pass and a settle; there has to be something to gain.

    When the small types hold most of the edges the narrow frame is the full
    frame, written twice, and the caller keeps the full frame throughout.
    """
    efe = EdgeFeatureExtractor(spark, {})

    assert efe._narrow_edges_for_small_types(
        _edges(spark, edge_rows),
        [("big", "rel", "big_dst"), ("small_a", "rel", "small_a_dst")],
        small_edges=7,
        total_edges=8,
    ) is None


def test_no_narrowing_when_there_are_no_small_types(spark, edge_rows):
    """Every type is on the chunked path, so nothing reads a narrowed frame."""
    efe = EdgeFeatureExtractor(spark, {})

    assert efe._narrow_edges_for_small_types(
        _edges(spark, edge_rows), [], small_edges=0, total_edges=8
    ) is None


def test_no_narrowing_when_the_edge_count_is_unknown(spark, edge_rows):
    """A zero total cannot be reasoned about, so the safe answer is the full
    frame rather than a division that decides on nothing."""
    efe = EdgeFeatureExtractor(spark, {})

    assert efe._narrow_edges_for_small_types(
        _edges(spark, edge_rows),
        [("small_a", "rel", "small_a_dst")],
        small_edges=0,
        total_edges=0,
    ) is None


# ======================================================================
# End to end: the narrowed frame changes no answer
# ======================================================================

def test_narrowing_changes_no_feature_value(spark, monkeypatch):
    """The whole path, with narrowing and without, must agree exactly.

    This is also the guard against the worst failure mode available here. The
    large type is deliberately absent from the narrow frame, so routing it there
    by mistake would give it no edges to read and an all-zero feature tensor,
    with nothing in the log to say so. That shows up here as a difference.
    """
    rows = _mixed_graph()

    with_narrowing, _ei, layout = _edge_features(
        spark, rows, config=SPLIT_AT_TWO
    )

    monkeypatch.setattr(
        EdgeFeatureExtractor,
        "_narrow_edges_for_small_types",
        lambda *args, **kwargs: None,
    )
    without, _ei2, _layout2 = _edge_features(spark, rows, config=SPLIT_AT_TWO)

    assert set(with_narrowing) == set(without)
    for key in with_narrowing:
        assert torch.equal(with_narrowing[key], without[key]), (
            f"{key} differs with narrowing on"
        )

    # Both halves of the split are present, or the test proves nothing.
    assert with_narrowing[LARGE_KEY].shape == (
        LARGE_EDGES, layout.edge_vector_dim
    )
    assert with_narrowing[SMALL_KEY].shape == (1, layout.edge_vector_dim)
    assert with_narrowing[LARGE_KEY].abs().sum() > 0, (
        "the large type came back all zeros, which is what routing it to the "
        "narrow frame would look like"
    )


def test_narrowing_reports_what_the_filters_now_scan(spark, caplog):
    """The log line is how a cluster run is read back.

    Without it the only evidence the narrowing happened is the wall clock,
    which is what made the pre-#362 version of this phase unreadable.
    """
    import logging

    logger_name = "spark_jobs.pyg_builder.edge_feature_extractor"
    with caplog.at_level(logging.INFO, logger=logger_name):
        _edge_features(spark, _mixed_graph(), config=SPLIT_AT_TWO)

    assert "Edges narrowed to the 1 small type(s)" in caplog.text
