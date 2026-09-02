"""
Regression tests for the endpoint-property scan in the edge feature phase.

The encoders filter the numeric property frame by endpoint node type, and each
of those filters is a full scan of it: segment 1 takes four per temporal edge
type, segment 2 two more. That was affordable while the collect ran one edge
type at a time, because each scan was its own small job.

#362 unioned every small type into a single job, which made the scans stop
being independent: 61 types put several hundred scans of a 305,127,393-row
frame into one plan. The 2026-09-01 cluster run spent 83 minutes in that job
and was killed without finishing it.

The fix narrows the frame once, to the endpoint types whose properties the
encoders were already going to broadcast, and hands that to the encoders. These
tests pin the three properties that has to keep: the narrow frame holds the
right types, reading it does not go back to the full frame, and an edge type
whose endpoint is NOT in it still reads the full frame rather than silently
producing zeros.
"""
import pytest

from pyspark.sql import functions as F

from spark_jobs.pyg_builder.edge_feature_extractor import (
    EdgeFeatureExtractor,
    _NUMERIC_ROW_BYTES,
    _PropertyRows,
)


# 10 rows of a type cost 1,280 bytes and 400 cost 51,200, so this bar puts
# "small_a" and "small_b" under it and "huge" over.
BROADCAST_BAR = 10_000

SMALL_ROWS = 10
HUGE_ROWS = 400


class _CountingProps:
    """The numeric property frame, counting how many rows get scanned.

    The counting UDF sits on the frame itself, so one full pass ticks once per
    row. It is non-deterministic so Catalyst cannot push a node_type filter
    underneath it and turn a full scan into a narrow one -- which would hide
    the very thing these tests measure.
    """

    def __init__(self, spark, df):
        self.accumulator = spark.sparkContext.accumulator(0)
        acc = self.accumulator

        def _tick(v):
            acc.add(1)
            return v

        tick = F.udf(_tick, "double").asNondeterministic()
        self.df = df.withColumn("numeric_value", tick(F.col("numeric_value")))

    @property
    def ticks(self) -> int:
        return self.accumulator.value


@pytest.fixture
def props(spark):
    """A stand-in for the endpoint property frames: two small types, one big."""
    rows = []
    for i in range(SMALL_ROWS):
        rows.append(("small_a", i, f"p{i % 3}", float(i)))
    for i in range(SMALL_ROWS):
        rows.append(("small_b", i, f"p{i % 3}", float(i) + 0.5))
    for i in range(HUGE_ROWS):
        rows.append(("huge", i, f"p{i % 3}", float(i) * 2.0))

    df = spark.createDataFrame(
        rows,
        schema="node_type STRING, node_id LONG, predicate STRING, "
               "numeric_value DOUBLE",
    )
    return _CountingProps(spark, df)


@pytest.fixture
def sizes():
    return {
        "small_a": _PropertyRows(total=SMALL_ROWS),
        "small_b": _PropertyRows(total=SMALL_ROWS),
        "huge": _PropertyRows(total=HUGE_ROWS),
    }


def _extractor(spark):
    efe = EdgeFeatureExtractor(spark, {})
    efe._broadcast_limit = BROADCAST_BAR
    return efe


def test_narrow_frame_holds_only_the_types_under_the_bar(spark, props, sizes):
    """The frame must hold the broadcastable types and leave out the rest.

    The bar is the same measured test _maybe_broadcast applies, so the narrow
    frame ends up holding exactly the types whose joins were meant to be
    broadcasts.
    """
    efe = _extractor(spark)

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )

    assert narrow is not None
    assert kept_types == {"small_a", "small_b"}, (
        f"'huge' is {HUGE_ROWS * _NUMERIC_ROW_BYTES:,} bytes against a bar of "
        f"{BROADCAST_BAR:,} and must not be narrowed into the frame"
    )

    got = {row["node_type"] for row in narrow.select("node_type").collect()}
    assert got == {"small_a", "small_b"}


def test_narrow_frame_keeps_every_row_of_the_types_it_holds(
    spark, props, sizes
):
    """Narrowing drops types, never rows of a type it kept.

    A missing row is a feature that silently reads as absent, so this is the
    correctness half of the change.
    """
    efe = _extractor(spark)

    narrow, _ = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )

    expected = sorted(
        (r["node_type"], r["node_id"], r["predicate"], r["numeric_value"])
        for r in props.df.filter(
            F.col("node_type").isin(["small_a", "small_b"])
        ).collect()
    )
    actual = sorted(
        (r["node_type"], r["node_id"], r["predicate"], r["numeric_value"])
        for r in narrow.collect()
    )

    assert actual == expected
    assert len(actual) == 2 * SMALL_ROWS


def test_per_type_filters_do_not_go_back_to_the_full_frame(
    spark, props, sizes
):
    """The point of the change: the encoders' filters scan the narrow frame.

    Before it, every per-type filter was a pass over the full frame, and the
    passes multiplied by edge type and by segment. After it the full frame is
    read once, to build the narrow one, and the per-type filters that follow
    add nothing to that count.
    """
    efe = _extractor(spark)

    narrow, _ = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )
    after_narrowing = props.ticks

    assert after_narrowing > 0, "the narrowing pass never read the frame"

    # Stand in for the six filters segment 1 and segment 2 issue per edge
    # type, across several types.
    for _ in range(3):
        for node_type in ("small_a", "small_b"):
            narrow.filter(F.col("node_type") == node_type).count()

    assert props.ticks == after_narrowing, (
        f"the per-type filters read the full frame again "
        f"({props.ticks - after_narrowing} extra row scans); the checkpoint "
        f"did not truncate the lineage, so the scans still multiply by edge "
        f"type"
    )


def test_an_edge_type_with_a_big_endpoint_reads_the_full_frame(
    spark, props, sizes
):
    """The guard against silent zeros.

    An edge type whose endpoint is missing from the narrow frame would read as
    having no properties at all and come back with an all-zero feature vector,
    with nothing in the log to say so. It has to fall back to the full frame.
    """
    efe = _extractor(spark)

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )

    both_small = efe._props_for(
        ("small_a", "rel", "small_b"), narrow, kept_types, props.df
    )
    assert both_small is narrow

    for key in [
        ("huge", "rel", "small_a"),
        ("small_a", "rel", "huge"),
        ("huge", "rel", "huge"),
    ]:
        assert efe._props_for(key, narrow, kept_types, props.df) is props.df, (
            f"{key} has an endpoint outside the narrow frame and must read "
            f"the full one"
        )


def test_an_unknown_endpoint_type_reads_the_full_frame(spark, props, sizes):
    """A type nobody sized is not assumed to be small.

    _endpoint_property_rows omits a type that contributes no rows, and such a
    type is genuinely narrow-safe. But a type that never reached sizing at all
    is a different case, and the routing only trusts the set it was handed.
    """
    efe = _extractor(spark)

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )

    assert efe._props_for(
        ("small_a", "rel", "never_sized"), narrow, kept_types, props.df
    ) is props.df


def test_no_narrowing_when_broadcasting_is_disabled(spark, props, sizes):
    """A negative bar means broadcasting is off, so there is nothing to size.

    _maybe_broadcast already honours that by never hinting; narrowing has to
    honour it the same way rather than invent its own threshold.
    """
    efe = _extractor(spark)
    efe._broadcast_limit = -1

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "small_b", "huge"}
    )

    assert narrow is None
    assert kept_types == set()
    assert efe._props_for(
        ("small_a", "rel", "small_b"), narrow, kept_types, props.df
    ) is props.df


def test_no_narrowing_when_every_type_is_over_the_bar(spark, props, sizes):
    """Nothing qualifies, so the caller keeps the full frame throughout."""
    efe = _extractor(spark)

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"huge"}
    )

    assert narrow is None
    assert kept_types == set()


def test_a_type_that_contributes_no_rows_is_narrow_safe(spark, props, sizes):
    """A type absent from sizing reads as zero rows and belongs in the frame.

    The narrow frame holds nothing for it, which is exactly what the full
    frame holds for it too -- so routing its edges to the narrow frame changes
    no answer.
    """
    efe = _extractor(spark)

    narrow, kept_types = efe._narrow_numeric_props(
        props.df, sizes, {"small_a", "no_properties_at_all"}
    )

    assert narrow is not None
    assert "no_properties_at_all" in kept_types
    assert efe._props_for(
        ("small_a", "rel", "no_properties_at_all"),
        narrow, kept_types, props.df,
    ) is narrow
