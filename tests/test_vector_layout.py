"""
Unit tests for VectorLayout (node feature vector geometry).

VectorLayout is the single source of truth for how the node feature vector is
partitioned into segments/sub-segments, scaled proportionally from vector_dim.
These are pure-Python arithmetic tests — no SparkSession, no cluster.

They pin the *actual* computed boundaries (which differ slightly from the
README's hand-written 1024 table at seg2/seg3 due to rounding), and enforce the
invariants the class guarantees for any dimension.
"""
import pytest

from spark_jobs.pyg_builder.feature_extractor import VectorLayout


# (start, dim) for every sub-segment, in vector order.
def _subsegments(layout):
    return [
        (layout.seg1_class_identity_start, layout.seg1_class_identity_dim),
        (layout.seg1_class_hierarchy_start, layout.seg1_class_hierarchy_dim),
        (layout.seg1_ontology_source_start, layout.seg1_ontology_source_dim),
        (layout.seg2_property_presence_start, layout.seg2_property_presence_dim),
        (layout.seg2_domain_range_start, layout.seg2_domain_range_dim),
        (layout.seg2_property_hierarchy_start, layout.seg2_property_hierarchy_dim),
        (layout.seg3_numeric_start, layout.seg3_numeric_dim),
        (layout.seg3_categorical_start, layout.seg3_categorical_dim),
    ]


# ======================================================================
# Invariants — must hold for every supported dimension
# ======================================================================

@pytest.mark.parametrize("dim", [32, 48, 64, 128, 256, 512, 1024, 2048, 4096])
def test_subsegments_tile_the_vector(dim):
    """Sub-segments are contiguous, gap-free, and sum exactly to vector_dim."""
    subs = _subsegments(VectorLayout(dim))

    # First starts at 0, each next starts where the previous ended.
    cursor = 0
    for start, size in subs:
        assert start == cursor, f"gap/overlap at {start}, expected {cursor}"
        assert size >= 1, "every sub-segment must have >= 1 dim"
        cursor += size
    assert cursor == dim, f"sub-segments sum to {cursor}, expected {dim}"


@pytest.mark.parametrize("dim", [32, 100, 512, 1024, 3000])
def test_segment_totals_sum_to_dim(dim):
    L = VectorLayout(dim)
    assert L.seg1_total + L.seg2_total + L.seg3_total == dim
    assert L.seg1_start == 0
    assert L.seg2_start == L.seg1_total
    assert L.seg3_start == L.seg1_total + L.seg2_total


# ======================================================================
# Minimum-dimension guard
# ======================================================================

@pytest.mark.parametrize("dim", [0, 1, 16, 31])
def test_rejects_too_small_dim(dim):
    with pytest.raises(ValueError):
        VectorLayout(dim)


def test_accepts_minimum_dim():
    # 32 is the documented floor; must construct and tile cleanly.
    L = VectorLayout(32)
    assert L.vector_dim == 32


# ======================================================================
# Exact boundaries (ground truth — guards against silent drift)
# ======================================================================
#
# A SNAPSHOT of the current composition, not a guarantee about it. The
# guarantees live in the invariant tests above, which hold for any set of
# fractions. Rebalancing what the vector carries is expected, and is *supposed*
# to fail these two: the point is that such a change is reviewed rather than
# silent. When you make one deliberately, update the numbers below -- the
# labelled form means the failure names the sub-segment that moved.
#
# Current tuning note: class_identity is 50% of segment 1 (128 dims at 1024),
# funded by class_hierarchy at 25%. It bounds how many ontology CLASSES the
# encoding can keep separable, and 64 dims could not cover a 118-class
# full-source build.

_SUBSEGMENT_NAMES = (
    "class_identity", "class_hierarchy", "ontology_source",
    "property_presence", "domain_range", "property_hierarchy",
    "numeric", "categorical",
)


def _labelled(layout):
    return dict(zip(_SUBSEGMENT_NAMES, _subsegments(layout)))


def test_exact_layout_1024():
    L = VectorLayout(1024)
    assert (L.seg1_total, L.seg2_total, L.seg3_total) == (256, 384, 384)
    # name -> (start, dim), the real computed values.
    assert _labelled(L) == {
        "class_identity": (0, 128),
        "class_hierarchy": (128, 64),
        "ontology_source": (192, 64),
        "property_presence": (256, 192),
        "domain_range": (448, 111),
        "property_hierarchy": (559, 81),
        "numeric": (640, 257),
        "categorical": (897, 127),
    }


def test_exact_layout_512():
    L = VectorLayout(512)
    assert (L.seg1_total, L.seg2_total, L.seg3_total) == (128, 192, 192)
    assert _labelled(L) == {
        "class_identity": (0, 64),
        "class_hierarchy": (64, 32),
        "ontology_source": (96, 32),
        "property_presence": (128, 96),
        "domain_range": (224, 56),
        "property_hierarchy": (280, 40),
        "numeric": (320, 129),
        "categorical": (449, 63),
    }


def test_halving_dim_halves_segment_totals():
    """Proportional scaling: 1024 -> 512 halves each segment total."""
    big = VectorLayout(1024)
    small = VectorLayout(512)
    assert small.seg1_total * 2 == big.seg1_total
    assert small.seg2_total * 2 == big.seg2_total
    assert small.seg3_total * 2 == big.seg3_total


# ======================================================================
# to_dict round-trips the attributes
# ======================================================================

def test_to_dict_matches_attributes():
    L = VectorLayout(1024)
    d = L.to_dict()
    assert d["vector_dim"] == 1024
    # Every key in to_dict must equal the corresponding attribute.
    for key, value in d.items():
        assert getattr(L, key) == value, f"{key} mismatch"
    # Spot-check a couple of derived boundaries survive serialization.
    assert d["seg3_categorical_start"] == 897
    assert d["seg3_categorical_dim"] == 127
