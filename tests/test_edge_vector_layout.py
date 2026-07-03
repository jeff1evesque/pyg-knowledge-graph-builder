"""
Unit tests for EdgeVectorLayout (edge feature vector geometry).

EdgeVectorLayout is the single source of truth for how the edge feature vector
is partitioned into segments/sub-segments, scaled proportionally from
edge_vector_dim. Mirror of the node-side VectorLayout tests — pure-Python
arithmetic, no SparkSession, no cluster.

They pin the *actual* computed boundaries (which differ from the class
docstring's hand-written 32-d example at seg3 due to rounding) and enforce the
invariants the class guarantees for any dimension.
"""
import pytest

from spark_jobs.pyg_builder.edge_feature_extractor import EdgeVectorLayout


# (start, dim) for every sub-segment, in vector order.
def _subsegments(layout):
    return [
        (layout.seg1_time_delta_start, layout.seg1_time_delta_dim),
        (layout.seg1_period_flags_start, layout.seg1_period_flags_dim),
        (layout.seg1_direction_start, layout.seg1_direction_dim),
        (layout.seg2_difference_start, layout.seg2_difference_dim),
        (layout.seg2_ratio_start, layout.seg2_ratio_dim),
        (layout.seg2_magnitude_start, layout.seg2_magnitude_dim),
        (layout.seg3_namespace_start, layout.seg3_namespace_dim),
        (layout.seg3_label_similarity_start, layout.seg3_label_similarity_dim),
        (layout.seg3_relation_identity_start, layout.seg3_relation_identity_dim),
    ]


# ======================================================================
# Invariants — must hold for every supported dimension
# ======================================================================

@pytest.mark.parametrize("dim", [9, 16, 32, 48, 64, 128, 256, 512, 1024])
def test_subsegments_tile_the_vector(dim):
    """Sub-segments are contiguous, gap-free, and sum exactly to edge_vector_dim."""
    subs = _subsegments(EdgeVectorLayout(dim))

    cursor = 0
    for start, size in subs:
        assert start == cursor, f"gap/overlap at {start}, expected {cursor}"
        assert size >= 1, "every sub-segment must have >= 1 dim"
        cursor += size
    assert cursor == dim, f"sub-segments sum to {cursor}, expected {dim}"


@pytest.mark.parametrize("dim", [9, 16, 32, 100, 512, 1024])
def test_segment_totals_sum_to_dim(dim):
    L = EdgeVectorLayout(dim)
    assert L.seg1_total + L.seg2_total + L.seg3_total == dim
    assert L.seg1_start == 0
    assert L.seg2_start == L.seg1_total
    assert L.seg3_start == L.seg1_total + L.seg2_total


# ======================================================================
# Minimum-dimension guard
# ======================================================================

@pytest.mark.parametrize("dim", [0, 1, 8])
def test_rejects_too_small_dim(dim):
    with pytest.raises(ValueError):
        EdgeVectorLayout(dim)


def test_accepts_minimum_dim():
    # 9 is the documented floor (one dim per sub-segment); must tile cleanly.
    L = EdgeVectorLayout(9)
    assert L.edge_vector_dim == 9
    for _, size in _subsegments(L):
        assert size >= 1


# ======================================================================
# Exact boundaries (ground truth — guards against silent drift)
# ======================================================================

def test_exact_layout_32():
    # 32 is the default EDGE_VECTOR_DIM.
    L = EdgeVectorLayout(32)
    assert (L.seg1_total, L.seg2_total, L.seg3_total) == (12, 12, 8)
    assert _subsegments(L) == [
        (0, 5),     # time_delta
        (5, 4),     # period_flags
        (9, 3),     # direction
        (12, 5),    # difference
        (17, 4),    # ratio
        (21, 3),    # magnitude
        (24, 3),    # namespace
        (27, 3),    # label_similarity
        (30, 2),    # relation_identity
    ]


def test_exact_layout_64():
    L = EdgeVectorLayout(64)
    assert (L.seg1_total, L.seg2_total, L.seg3_total) == (24, 24, 16)
    assert _subsegments(L) == [
        (0, 10),
        (10, 8),
        (18, 6),
        (24, 10),
        (34, 8),
        (42, 6),
        (48, 6),
        (54, 6),
        (60, 4),
    ]


def test_min_one_dim_clamps_at_small_dims():
    """At dim=16 the seg3 sub-segments hit the >=1 floor; still tiles exactly."""
    L = EdgeVectorLayout(16)
    subs = _subsegments(L)
    assert subs[-2] == (14, 1)   # label_similarity clamped to 1
    assert subs[-1] == (15, 1)   # relation_identity clamped to 1
    assert sum(size for _, size in subs) == 16


def test_doubling_dim_doubles_segment_totals():
    """Proportional scaling: 32 -> 64 doubles each segment total."""
    small = EdgeVectorLayout(32)
    big = EdgeVectorLayout(64)
    assert small.seg1_total * 2 == big.seg1_total
    assert small.seg2_total * 2 == big.seg2_total
    assert small.seg3_total * 2 == big.seg3_total


# ======================================================================
# to_dict round-trips the attributes
# ======================================================================

def test_to_dict_matches_attributes():
    L = EdgeVectorLayout(32)
    d = L.to_dict()
    assert d["edge_vector_dim"] == 32
    # Every key in to_dict must equal the corresponding attribute.
    for key, value in d.items():
        assert getattr(L, key) == value, f"{key} mismatch"
    # Spot-check a derived boundary survives serialization.
    assert d["seg3_relation_identity_start"] == 30
    assert d["seg3_relation_identity_dim"] == 2
