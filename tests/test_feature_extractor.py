"""Value-level unit tests for FeatureExtractor node-feature encoding
(spark_jobs/pyg_builder/feature_extractor.py).

Only VectorLayout's descriptor geometry was previously tested — never the
*encoding*, i.e. that a given triple lands in the correct vector slot with the
correct value. These tests drive build_features over tiny synthetic triples on
the shared local SparkSession and assert the encoded rows at the value level:

  * class-identity one-hots occupy exactly the layout-reserved slots (the same
    F.hash(type_uri, seed) the encoder uses, reproduced here — never hardcoded);
  * numeric literals are z-score normalized to the expected value in the numeric
    slot (F.hash(predicate, 500));
  * categorical values set exactly the expected multi-hot slots;
  * _scatter_all_types places each type's rows at the right node IDs;
  * encoding the same fixture repeatedly yields identical vectors — the direct
    regression guard for the jolts_* categorical non-determinism (Issue open).

Vector width is pinned to 128 for speed; every assertion derives its slots from
VectorLayout(128), so nothing depends on the production 1024 default.
"""
import numpy as np
import pytest
import torch
from pyspark.sql import functions as F

from spark_jobs.pyg_builder.node_mapper import NodeMapper
from spark_jobs.pyg_builder.feature_extractor import (
    FeatureExtractor,
    VectorLayout,
    _HASH_SEEDS,
)

CPI_INDEX = "https://www.bls.gov/cpi/Index"        # -> cpi_Index
CPI_SERIES = "https://www.bls.gov/cpi/Series"      # -> cpi_Series
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
METRIC = "https://example.org/metric"              # numeric literal predicate
SECTOR = "https://example.org/sector"              # categorical predicate
REGION = "https://example.org/region"              # categorical predicate

VDIM = 128
CONFIG = {"feature_config": {"vector_dim": VDIM}}


def _node_features(spark, rows):
    """Run the real node-feature encoding; return {pyg_name: np.ndarray}."""
    triples = spark.createDataFrame(
        rows, schema="subject STRING, predicate STRING, object STRING"
    )
    node_id_df, counts = NodeMapper(spark, CONFIG).build_node_id_table(triples)
    tensors, _names = FeatureExtractor(spark, CONFIG).build_features(
        triples, node_id_df, counts
    )
    return {t: v.numpy() for t, v in tensors.items()}


def _hash_slot(spark, parts, dim, start):
    """Reproduce the encoder's slot: abs(F.hash(*parts)) % dim + start.

    `parts` are hashed together in order exactly as the encoder passes them
    (e.g. [type_uri, seed] or ["pred::value", seed]). Uses the same Spark
    primitive, so it tracks the implementation rather than reimplementing it.
    """
    cols = [F.lit(p) for p in parts]
    expr = F.abs(F.hash(*cols)) % F.lit(dim) + F.lit(start)
    return int(spark.range(1).select(expr.alias("d")).first()["d"])


def _seg(row, start, dim):
    return row[start:start + dim]


# ======================================================================
# Segment 1 — class identity one-hot
# ======================================================================

def test_class_identity_lands_in_reserved_slots(spark):
    x = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    ci_start, ci_dim = L.seg1_class_identity_start, L.seg1_class_identity_dim

    # Expected: each of the 4 hash seeds sets one slot to 1.0 (collisions sum).
    expected = np.zeros(ci_dim, dtype=np.float32)
    for seed in _HASH_SEEDS:
        slot = _hash_slot(spark, [CPI_INDEX, seed], ci_dim, ci_start)
        assert ci_start <= slot < ci_start + ci_dim
        expected[slot - ci_start] += 1.0

    np.testing.assert_allclose(_seg(x[0], ci_start, ci_dim), expected, atol=1e-6)
    # Total mass equals the number of seeds (each contributes 1.0).
    assert _seg(x[0], ci_start, ci_dim).sum() == pytest.approx(len(_HASH_SEEDS))


def test_class_identity_distinguishes_types(spark):
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_SERIES),
    ])
    L = VectorLayout(VDIM)
    ci_start, ci_dim = L.seg1_class_identity_start, L.seg1_class_identity_dim
    idx = _seg(feats["cpi_Index"][0], ci_start, ci_dim)
    ser = _seg(feats["cpi_Series"][0], ci_start, ci_dim)
    # Different type URIs must produce different class-identity fingerprints.
    assert not np.array_equal(idx, ser)


# ======================================================================
# Segment 3a — numeric literal z-score normalization
# ======================================================================

def test_numeric_literal_is_zscore_normalized(spark):
    # Values 10/20/30 → mean 20, sample std 10 → normalized -1/0/+1.
    # URIs sort n0<n1<n2 → node IDs 0/1/2 receive 10/20/30 respectively.
    x = _node_features(spark, [
        ("https://ex/n0", RDF_TYPE, CPI_INDEX),
        ("https://ex/n1", RDF_TYPE, CPI_INDEX),
        ("https://ex/n2", RDF_TYPE, CPI_INDEX),
        ("https://ex/n0", METRIC, "10"),
        ("https://ex/n1", METRIC, "20"),
        ("https://ex/n2", METRIC, "30"),
    ])["cpi_Index"]

    L = VectorLayout(VDIM)
    num_start, num_dim = L.seg3_numeric_start, L.seg3_numeric_dim
    slot = _hash_slot(spark, [METRIC, 500], num_dim, num_start)

    expected = [-1.0, 0.0, 1.0]
    for node_id, want in enumerate(expected):
        assert x[node_id, slot] == pytest.approx(want, abs=1e-5)
        # Single predicate → the numeric sub-segment holds exactly this value.
        assert _seg(x[node_id], num_start, num_dim).sum() == pytest.approx(
            want, abs=1e-5
        )


# ======================================================================
# Segment 3b — categorical multi-hot
# ======================================================================

def test_categorical_sets_expected_multihot_slots(spark):
    x = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
    ])["cpi_Index"]

    L = VectorLayout(VDIM)
    cat_start, cat_dim = L.seg3_categorical_start, L.seg3_categorical_dim

    expected = np.zeros(cat_dim, dtype=np.float32)
    for seed in _HASH_SEEDS:
        slot = _hash_slot(
            spark, [f"{SECTOR}::Energy", seed + 600], cat_dim, cat_start
        )
        assert cat_start <= slot < cat_start + cat_dim
        expected[slot - cat_start] += 1.0

    np.testing.assert_allclose(
        _seg(x[0], cat_start, cat_dim), expected, atol=1e-6
    )


def test_distinct_categorical_values_differ(spark):
    feats = _node_features(spark, [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        ("https://ex/b", SECTOR, "Finance"),
    ])["cpi_Index"]
    L = VectorLayout(VDIM)
    cat = (L.seg3_categorical_start, L.seg3_categorical_dim)
    # Nodes 0 and 1 carry different categorical values → different multi-hot.
    assert not np.array_equal(_seg(feats[0], *cat), _seg(feats[1], *cat))


# ======================================================================
# _scatter_all_types — per-type placement at the right node IDs
# ======================================================================

def test_scatter_places_each_type_at_its_own_node_ids(spark):
    feats = _node_features(spark, [
        ("https://ex/i0", RDF_TYPE, CPI_INDEX),
        ("https://ex/i1", RDF_TYPE, CPI_INDEX),
        ("https://ex/s0", RDF_TYPE, CPI_SERIES),
        # Distinct numeric values, monotonic in each type's node IDs.
        ("https://ex/i0", METRIC, "10"),
        ("https://ex/i1", METRIC, "20"),
        ("https://ex/s0", METRIC, "5"),
    ])
    L = VectorLayout(VDIM)
    slot = _hash_slot(
        spark, [METRIC, 500], L.seg3_numeric_dim, L.seg3_numeric_start
    )

    # Each type gets its own tensor with the right row count and width.
    assert feats["cpi_Index"].shape == (2, VDIM)
    assert feats["cpi_Series"].shape == (1, VDIM)
    # Within cpi_Index, the smaller input value (node 0: 10) normalizes below the
    # larger (node 1: 20) — monotonic regardless of the global mean/std, so rows
    # are placed at their own node IDs (not swapped across the type boundary).
    assert feats["cpi_Index"][0, slot] < feats["cpi_Index"][1, slot]
    assert np.isfinite(feats["cpi_Series"][0, slot])


# ======================================================================
# Determinism — regression guard for the jolts_* categorical bug
# ======================================================================

def test_categorical_encoding_is_deterministic_across_runs(spark):
    # Multiple categorical values per node exercise the multi-hot + group-by-sum
    # combine where the jolts_* non-determinism lives. Repeated encodings of the
    # same fixture must be bit-identical.
    rows = [
        ("https://ex/a", RDF_TYPE, CPI_INDEX),
        ("https://ex/b", RDF_TYPE, CPI_INDEX),
        ("https://ex/c", RDF_TYPE, CPI_INDEX),
        ("https://ex/a", SECTOR, "Energy"),
        ("https://ex/a", REGION, "West"),
        ("https://ex/b", SECTOR, "Energy"),
        ("https://ex/b", REGION, "East"),
        ("https://ex/c", SECTOR, "Finance"),
        ("https://ex/c", REGION, "West"),
    ]
    first = _node_features(spark, rows)["cpi_Index"]
    again = _node_features(spark, rows)["cpi_Index"]
    assert np.array_equal(first, again), (
        "categorical encoding is order-dependent (jolts_* regression)"
    )
