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
# Current tuning note: class_identity is 62.5% of segment 1 (160 dims at 1024),
# funded by class_hierarchy and ontology_source at 18.75% each. It bounds how
# many ontology CLASSES the encoding can keep separable -- 64 dims could not
# cover a 118-class full-source build, and 128 left only 32 classes of headroom
# over the 96 a full fixture build produces. Segment 1 is a quarter of the
# vector, so no split of it exceeds ~192 classes; past that the levers are
# feature_config.class_identity_dim or a larger vector_dim.

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
        "class_identity": (0, 160),
        "class_hierarchy": (160, 48),
        "ontology_source": (208, 48),
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
        "class_identity": (0, 80),
        "class_hierarchy": (80, 24),
        "ontology_source": (104, 24),
        "property_presence": (128, 96),
        "domain_range": (224, 56),
        "property_hierarchy": (280, 40),
        "numeric": (320, 129),
        "categorical": (449, 63),
    }


def test_exact_layout_1024_with_class_identity_override():
    """The #359 configuration, pinned.

    The 2026-08-30 graph carries 184 classes and the default split gives
    class_identity 160 dims, so both PyG legs died on the capacity guard. 192
    is the practical ceiling at this width: class_hierarchy keeps its
    fraction-derived 48, and everything above 192 comes out of ontology_source,
    which is down to 16 for a 26-entry namespace table already.

    Only segment 1 is redistributed. Every other boundary, and the vector
    width, must be identical to test_exact_layout_1024 -- that is the whole
    reason this lever exists rather than raising vector_dim, which would double
    the driver memory the feature tensors are collected into.
    """
    L = VectorLayout(1024, class_identity_dim=192)
    assert (L.seg1_total, L.seg2_total, L.seg3_total) == (256, 384, 384)
    assert _labelled(L) == {
        "class_identity": (0, 192),
        "class_hierarchy": (192, 48),
        "ontology_source": (240, 16),
        # unchanged from the default layout
        "property_presence": (256, 192),
        "domain_range": (448, 111),
        "property_hierarchy": (559, 81),
        "numeric": (640, 257),
        "categorical": (897, 127),
    }
    assert L.vector_dim == 1024


@pytest.mark.parametrize("ci_dim", [1, 100, 254])
def test_class_identity_override_still_tiles_the_vector(ci_dim):
    """An override must not open a gap or an overlap."""
    L = VectorLayout(1024, class_identity_dim=ci_dim)
    subsegments = _subsegments(L)
    assert L.seg1_class_identity_dim == ci_dim
    cursor = 0
    for start, dim in subsegments:
        assert start == cursor, f"gap or overlap at {start}, expected {cursor}"
        assert dim >= 1, "a sub-segment may not be zero-width"
        cursor += dim
    assert cursor == 1024


@pytest.mark.parametrize("ci_dim", [0, -1, 255, 1024])
def test_class_identity_override_out_of_range_is_rejected(ci_dim):
    """Rejected, not clamped.

    The point of the override is to guarantee a class budget, so a build that
    asks for a capacity the vector cannot hold must not quietly get a smaller
    one. 254 is the maximum at vector_dim=1024: class_hierarchy and
    ontology_source are indexed into and need at least one dim each.
    """
    with pytest.raises(ValueError, match="class_identity_dim"):
        VectorLayout(1024, class_identity_dim=ci_dim)


def test_class_identity_override_carries_the_class_count_that_failed():
    """184 classes fit in the override and do not fit in the default."""
    assert VectorLayout(1024).seg1_class_identity_dim < 184
    assert VectorLayout(1024, class_identity_dim=192).seg1_class_identity_dim >= 184


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
