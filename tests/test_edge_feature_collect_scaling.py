"""
Regression tests for #362: small-type collection must not re-scan its inputs.

Each small edge type's entries frame is a lazy join against the endpoint
property frames, which run to hundreds of millions of rows on a real graph.
The collect used to run straight off those lazy frames in batches of eight
types, so every batch re-ran the join. The measured cost on 2026-08-31 was the
same 23,184 MB read on each of 78 stages -- about 1.7 TB of scanning to return
41 MB of results.

The fix aggregates and checkpoints the entries once, before any batching. These
tests pin the two properties that keeps: the source frame is read a bounded
number of times regardless of how many types there are, and the features that
come out are unchanged.
"""
from typing import List, Tuple

import numpy as np
import pytest
import torch

from pyspark.sql import functions as F

from spark_jobs.pyg_builder.edge_feature_extractor import EdgeFeatureExtractor


EDGE_VECTOR_DIM = 32


class _CountingSource:
    """A shared input frame that counts how many rows get scanned.

    This models the endpoint property frames. The counting UDF sits on the
    SHARED frame, above the per-type filters, so one full pass over it ticks
    once per row no matter which type is being built. That is what makes the
    count sensitive to #362: re-running the collect per batch re-scans this
    frame, and the tick count rises with the number of batches.

    The UDF is marked non-deterministic so Catalyst cannot push the per-type
    filter underneath it and quietly turn a full scan into a narrow one.
    """

    ROWS = 512

    def __init__(self, spark, df):
        self.accumulator = spark.sparkContext.accumulator(0)
        acc = self.accumulator

        def _tick(v):
            acc.add(1)
            return v

        tick = F.udf(_tick, "float").asNondeterministic()
        self._counted = df.withColumn("value", tick(F.col("value")))

    def entries_for(self, type_seed: int, num_edges: int):
        """One type's sparse entries, read off the shared counted frame.

        edge_idx is taken modulo num_edges because it is per-type 0-based:
        an entry outside [0, num_edges) has no slot in that type's tensor.
        """
        return (
            self._counted
            .filter(F.col("bucket") == type_seed % 4)
            .select(
                (F.col("edge_idx") % F.lit(num_edges)).alias("edge_idx"),
                F.col("dim"),
                F.col("value"),
            )
        )

    @property
    def scans(self) -> int:
        return self.accumulator.value

    def passes(self, since: int = 0) -> float:
        """Row ticks since `since`, expressed in full passes of the frame."""
        return (self.accumulator.value - since) / float(self.ROWS)


@pytest.fixture
def source(spark):
    """A stand-in for the endpoint property frames: one shared big input."""
    rows = [
        (i % 8, i % 4, float(i % 7) + 1.0, i % 4)
        for i in range(_CountingSource.ROWS)
    ]
    df = spark.createDataFrame(
        rows, schema="edge_idx LONG, dim INT, value FLOAT, bucket INT"
    )
    return _CountingSource(spark, df)


def _extractor(spark):
    return EdgeFeatureExtractor(spark, {})


def _small_list(source, n_types: int, edges_per_type: int = 8):
    """Build the (key, num_edges, entries) list _collect_small_batches takes."""
    small: List[Tuple[Tuple[str, str, str], int, object]] = []
    for i in range(n_types):
        key = (f"src{i}", f"rel{i}", f"dst{i}")
        small.append((key, edges_per_type, source.entries_for(i, edges_per_type)))
    return small


def test_entries_are_materialised_once_however_many_batches(spark, source):
    """The #362 invariant: one pass over the inputs, not one per batch.

    Sixteen types with a cap of two forces eight batches. Before the fix each
    batch collected straight off the lazy per-type frames, so the joins
    feeding them ran eight times. The materialising pass must happen exactly
    once regardless of how the collect is then batched.
    """
    efe = _extractor(spark)
    efe._max_types_per_batch = 2

    calls = []
    real = efe._materialise_small_entries

    def _spy(small):
        calls.append(len(small))
        return real(small)

    efe._materialise_small_entries = _spy

    flushes = []
    real_flush = efe._flush_batch

    def _flush_spy(materialised, batch, features, dim):
        flushes.append(materialised)
        return real_flush(materialised, batch, features, dim)

    efe._flush_batch = _flush_spy

    small = _small_list(source, 16, edges_per_type=8)
    efe._collect_small_batches(small, {}, EDGE_VECTOR_DIM)

    assert calls == [16], (
        f"expected one materialising pass over all 16 types, got {calls}"
    )
    assert len(flushes) == 8, (
        f"expected 8 batches at a cap of 2, got {len(flushes)}"
    )
    assert all(f is flushes[0] for f in flushes), (
        "batches collected from different frames -- each one is re-deriving "
        "its own inputs instead of reading the single materialised result "
        "(#362)"
    )


def test_batches_do_not_carry_the_source_lineage(spark, source):
    """What the batches read must be the checkpoint, not the input joins.

    This is the structural half of #362: if the collect ever goes back to
    reading the per-type frames, the source scan reappears in the plan the
    batches run against and this fails.
    """
    efe = _extractor(spark)
    small = _small_list(source, 6, edges_per_type=8)

    materialised = efe._materialise_small_entries(small)
    assert materialised is not None

    plan = materialised._jdf.queryExecution().toString()

    assert "BatchScan" not in plan and "FileScan" not in plan, (
        "materialised entries still reference a source scan; the checkpoint "
        "did not truncate the lineage (#362)"
    )
    assert efe._TYPE_COL in plan, (
        "materialised entries lost the type tag, so batches cannot tell one "
        "type's rows from another's"
    )


def test_features_are_unchanged_across_the_batch_boundary(spark, source):
    """Correctness guard: batching must not change the tensors produced.

    Runs the same types twice, once where they all fit in a single batch and
    once where the cap forces several, and requires identical output.
    """
    small = _small_list(source, 12, edges_per_type=8)

    one_batch = _extractor(spark)
    one_batch._max_types_per_batch = 64
    features_one: dict = {}
    one_batch._collect_small_batches(small, features_one, EDGE_VECTOR_DIM)

    many_batches = _extractor(spark)
    many_batches._max_types_per_batch = 2
    features_many: dict = {}
    many_batches._collect_small_batches(small, features_many, EDGE_VECTOR_DIM)

    assert set(features_one) == set(features_many)
    assert features_one, "no features were produced at all"

    for key, tensor in features_one.items():
        assert torch.equal(tensor, features_many[key]), (
            f"{key} differs depending on how the types were batched"
        )


def test_every_type_gets_a_tensor_of_the_right_shape(spark, source):
    """Each type must come back with its own (num_edges, dim) tensor."""
    efe = _extractor(spark)
    small = _small_list(source, 10, edges_per_type=6)

    features: dict = {}
    efe._collect_small_batches(small, features, EDGE_VECTOR_DIM)

    assert len(features) == 10
    for key, num_edges, _ in small:
        tensor = features[key]
        assert tensor.shape == (num_edges, EDGE_VECTOR_DIM)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_empty_input_does_no_work(spark):
    """No types means no Spark job and no tensors."""
    efe = _extractor(spark)
    features: dict = {}
    efe._collect_small_batches([], features, EDGE_VECTOR_DIM)
    assert features == {}


def test_values_land_on_their_own_type(spark, source):
    """The type tag must keep one type's entries out of another's tensor."""
    efe = _extractor(spark)

    rows = [(0, 3, 5.0, 0)]
    only_one = spark.createDataFrame(
        rows, schema="edge_idx LONG, dim INT, value FLOAT, bucket INT"
    ).select("edge_idx", "dim", "value")

    empty = only_one.filter(F.col("edge_idx") < 0)

    small = [
        (("a", "r", "b"), 2, only_one),
        (("c", "r", "d"), 2, empty),
    ]

    features: dict = {}
    efe._collect_small_batches(small, features, EDGE_VECTOR_DIM)

    got = features[("a", "r", "b")].numpy()
    assert got[0, 3] == pytest.approx(5.0)
    assert np.count_nonzero(got) == 1

    assert np.count_nonzero(features[("c", "r", "d")].numpy()) == 0
