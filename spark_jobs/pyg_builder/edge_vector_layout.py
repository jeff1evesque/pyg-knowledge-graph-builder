"""
Where each part of an edge feature vector lives.

``EdgeVectorLayout`` turns an ``edge_vector_dim`` into the start index and the
width of every segment and sub-segment. It does for edges what
``vector_layout.py`` does for nodes, over a different set of segments: the
encoders in ``edge_feature_extractor.py`` place a value by asking a layout for
the slot, and ``metadata_writer.py`` publishes the same layout in
feature_spec.json and encoding_config.json -- so what a model reads and what
the metadata says a slot holds cannot drift apart.

Pure arithmetic. No Spark, no torch, nothing that runs on an executor.
"""
from typing import Any, Dict


# ============================================
# Default edge vector dimension
# ============================================
EDGE_VECTOR_DIM = 32

# ============================================
# Segment proportions (fraction of total edge_vector_dim)
# ============================================
# These ratios are applied to any edge_vector_dim to compute boundaries.
#
# Segment 1: Temporal Signals — 37.5%
#   Sub-segments: time_delta=40%, period_flags=35%, direction=25%
#
# Segment 2: Numeric Contrast — 37.5%
#   Sub-segments: difference=40%, ratio=35%, magnitude=25%
#
# Segment 3: Relational Context — 25%
#   Sub-segments: namespace=40%, label_similarity=35%, relation_identity=25%
#
# These three are public because edge_feature_extractor.py publishes them as
# the segment_proportions of encoding_config.json. The sub-segment fractions
# below are read only here.
EDGE_SEG1_FRAC = 0.375
EDGE_SEG2_FRAC = 0.375
EDGE_SEG3_FRAC = 0.25  # = 1.0 - 0.375 - 0.375

# Sub-segment fractions within each segment
_EDGE_SEG1_TIME_DELTA_FRAC = 0.40
_EDGE_SEG1_PERIOD_FLAGS_FRAC = 0.35
_EDGE_SEG1_DIRECTION_FRAC = 0.25

_EDGE_SEG2_DIFFERENCE_FRAC = 0.40
_EDGE_SEG2_RATIO_FRAC = 0.35
_EDGE_SEG2_MAGNITUDE_FRAC = 0.25

_EDGE_SEG3_NAMESPACE_FRAC = 0.40
_EDGE_SEG3_LABEL_SIMILARITY_FRAC = 0.35
_EDGE_SEG3_RELATION_IDENTITY_FRAC = 0.25


class EdgeVectorLayout:
    """
    Computes all segment and sub-segment boundaries for edge feature
    vectors from a given edge_vector_dim. All boundaries are integer
    dim indices.

    Mirrors VectorLayout for node features — same guarantees:
    - All sub-segments are contiguous and non-overlapping
    - All sub-segments have at least 1 dimension (raises if
      edge_vector_dim is too small)
    - Sub-segment dims sum exactly to edge_vector_dim (no gaps,
      no overlap)

    Usage:
        layout = EdgeVectorLayout(32)
        layout.seg1_time_delta_start      # 0
        layout.seg1_time_delta_dim        # 5
        layout.seg3_relation_identity_start  # 26
        layout.seg3_relation_identity_dim    # 6
        layout.edge_vector_dim            # 32
    """

    def __init__(self, edge_vector_dim: int):
        if edge_vector_dim < 9:
            raise ValueError(
                f"edge_vector_dim must be >= 9, got {edge_vector_dim}. "
                f"Minimum needed for all sub-segments to have >= 1 dim."
            )

        self.edge_vector_dim = edge_vector_dim

        # --- Segment boundaries ---
        seg1_total = max(1, int(round(edge_vector_dim * EDGE_SEG1_FRAC)))
        seg2_total = max(1, int(round(edge_vector_dim * EDGE_SEG2_FRAC)))
        seg3_total = edge_vector_dim - seg1_total - seg2_total  # remainder

        if seg3_total < 1:
            raise ValueError(
                f"edge_vector_dim={edge_vector_dim} too small: seg3 "
                f"would have {seg3_total} dims"
            )

        self.seg1_start = 0
        self.seg1_total = seg1_total
        self.seg2_start = seg1_total
        self.seg2_total = seg2_total
        self.seg3_start = seg1_total + seg2_total
        self.seg3_total = seg3_total

        # --- Segment 1 sub-segments: Temporal Signals ---
        td_dim = max(1, int(round(seg1_total * _EDGE_SEG1_TIME_DELTA_FRAC)))
        pf_dim = max(
            1, int(round(seg1_total * _EDGE_SEG1_PERIOD_FLAGS_FRAC))
        )
        dir_dim = seg1_total - td_dim - pf_dim  # remainder

        if dir_dim < 1:
            dir_dim = 1
            pf_dim = seg1_total - td_dim - dir_dim

        self.seg1_time_delta_start = self.seg1_start
        self.seg1_time_delta_dim = td_dim
        self.seg1_period_flags_start = self.seg1_start + td_dim
        self.seg1_period_flags_dim = pf_dim
        self.seg1_direction_start = self.seg1_start + td_dim + pf_dim
        self.seg1_direction_dim = dir_dim

        # --- Segment 2 sub-segments: Numeric Contrast ---
        diff_dim = max(
            1, int(round(seg2_total * _EDGE_SEG2_DIFFERENCE_FRAC))
        )
        rat_dim = max(1, int(round(seg2_total * _EDGE_SEG2_RATIO_FRAC)))
        mag_dim = seg2_total - diff_dim - rat_dim  # remainder

        if mag_dim < 1:
            mag_dim = 1
            rat_dim = seg2_total - diff_dim - mag_dim

        self.seg2_difference_start = self.seg2_start
        self.seg2_difference_dim = diff_dim
        self.seg2_ratio_start = self.seg2_start + diff_dim
        self.seg2_ratio_dim = rat_dim
        self.seg2_magnitude_start = self.seg2_start + diff_dim + rat_dim
        self.seg2_magnitude_dim = mag_dim

        # --- Segment 3 sub-segments: Relational Context ---
        ns_dim = max(
            1, int(round(seg3_total * _EDGE_SEG3_NAMESPACE_FRAC))
        )
        ls_dim = max(
            1,
            int(round(seg3_total * _EDGE_SEG3_LABEL_SIMILARITY_FRAC)),
        )
        ri_dim = seg3_total - ns_dim - ls_dim  # remainder

        if ri_dim < 1:
            ri_dim = 1
            ls_dim = seg3_total - ns_dim - ri_dim

        self.seg3_namespace_start = self.seg3_start
        self.seg3_namespace_dim = ns_dim
        self.seg3_label_similarity_start = self.seg3_start + ns_dim
        self.seg3_label_similarity_dim = ls_dim
        self.seg3_relation_identity_start = (
            self.seg3_start + ns_dim + ls_dim
        )
        self.seg3_relation_identity_dim = ri_dim

        self._validate()

    def _validate(self):
        """Verify all sub-segments tile the full vector with no gaps."""
        total = (
            self.seg1_time_delta_dim
            + self.seg1_period_flags_dim
            + self.seg1_direction_dim
            + self.seg2_difference_dim
            + self.seg2_ratio_dim
            + self.seg2_magnitude_dim
            + self.seg3_namespace_dim
            + self.seg3_label_similarity_dim
            + self.seg3_relation_identity_dim
        )
        assert total == self.edge_vector_dim, (
            f"Sub-segment dims sum to {total}, "
            f"expected {self.edge_vector_dim}"
        )

        # Verify contiguity
        assert self.seg1_time_delta_start == 0
        assert (
            self.seg1_period_flags_start
            == self.seg1_time_delta_start + self.seg1_time_delta_dim
        )
        assert (
            self.seg1_direction_start
            == self.seg1_period_flags_start + self.seg1_period_flags_dim
        )
        assert (
            self.seg2_difference_start
            == self.seg1_direction_start + self.seg1_direction_dim
        )
        assert (
            self.seg2_ratio_start
            == self.seg2_difference_start + self.seg2_difference_dim
        )
        assert (
            self.seg2_magnitude_start
            == self.seg2_ratio_start + self.seg2_ratio_dim
        )
        assert (
            self.seg3_namespace_start
            == self.seg2_magnitude_start + self.seg2_magnitude_dim
        )
        assert (
            self.seg3_label_similarity_start
            == self.seg3_namespace_start + self.seg3_namespace_dim
        )
        assert (
            self.seg3_relation_identity_start
            == self.seg3_label_similarity_start
            + self.seg3_label_similarity_dim
        )
        assert (
            self.seg3_relation_identity_start
            + self.seg3_relation_identity_dim
            == self.edge_vector_dim
        )

        # Verify all dims >= 1
        for attr_name in dir(self):
            if attr_name.endswith("_dim") and not attr_name.startswith("_"):
                val = getattr(self, attr_name)
                assert val >= 1, f"{attr_name} = {val}, must be >= 1"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize layout to a JSON-compatible dict.

        Used by MetadataCollector for feature_spec.json and
        encoding_config.json. Contains all segment and sub-segment
        boundaries needed to reconstruct the layout.
        """
        return {
            "edge_vector_dim": self.edge_vector_dim,
            "seg1_start": self.seg1_start,
            "seg1_total": self.seg1_total,
            "seg1_time_delta_start": self.seg1_time_delta_start,
            "seg1_time_delta_dim": self.seg1_time_delta_dim,
            "seg1_period_flags_start": self.seg1_period_flags_start,
            "seg1_period_flags_dim": self.seg1_period_flags_dim,
            "seg1_direction_start": self.seg1_direction_start,
            "seg1_direction_dim": self.seg1_direction_dim,
            "seg2_start": self.seg2_start,
            "seg2_total": self.seg2_total,
            "seg2_difference_start": self.seg2_difference_start,
            "seg2_difference_dim": self.seg2_difference_dim,
            "seg2_ratio_start": self.seg2_ratio_start,
            "seg2_ratio_dim": self.seg2_ratio_dim,
            "seg2_magnitude_start": self.seg2_magnitude_start,
            "seg2_magnitude_dim": self.seg2_magnitude_dim,
            "seg3_start": self.seg3_start,
            "seg3_total": self.seg3_total,
            "seg3_namespace_start": self.seg3_namespace_start,
            "seg3_namespace_dim": self.seg3_namespace_dim,
            "seg3_label_similarity_start": (
                self.seg3_label_similarity_start
            ),
            "seg3_label_similarity_dim": (
                self.seg3_label_similarity_dim
            ),
            "seg3_relation_identity_start": (
                self.seg3_relation_identity_start
            ),
            "seg3_relation_identity_dim": (
                self.seg3_relation_identity_dim
            ),
        }

    def summary(self) -> str:
        """Human-readable layout summary."""
        lines = [
            f"EdgeVectorLayout(edge_vector_dim={self.edge_vector_dim})",
            f"  Segment 1: Temporal Signals "
            f"[{self.seg1_start}–{self.seg2_start - 1}] "
            f"({self.seg1_total} dims)",
            f"    Time Delta:        "
            f"[{self.seg1_time_delta_start}–"
            f"{self.seg1_time_delta_start + self.seg1_time_delta_dim - 1}] "
            f"({self.seg1_time_delta_dim} dims)",
            f"    Period Flags:      "
            f"[{self.seg1_period_flags_start}–"
            f"{self.seg1_period_flags_start + self.seg1_period_flags_dim - 1}] "
            f"({self.seg1_period_flags_dim} dims)",
            f"    Direction:         "
            f"[{self.seg1_direction_start}–"
            f"{self.seg1_direction_start + self.seg1_direction_dim - 1}] "
            f"({self.seg1_direction_dim} dims)",
            f"  Segment 2: Numeric Contrast "
            f"[{self.seg2_start}–{self.seg3_start - 1}] "
            f"({self.seg2_total} dims)",
            f"    Difference:        "
            f"[{self.seg2_difference_start}–"
            f"{self.seg2_difference_start + self.seg2_difference_dim - 1}] "
            f"({self.seg2_difference_dim} dims)",
            f"    Ratio:             "
            f"[{self.seg2_ratio_start}–"
            f"{self.seg2_ratio_start + self.seg2_ratio_dim - 1}] "
            f"({self.seg2_ratio_dim} dims)",
            f"    Magnitude:         "
            f"[{self.seg2_magnitude_start}–"
            f"{self.seg2_magnitude_start + self.seg2_magnitude_dim - 1}] "
            f"({self.seg2_magnitude_dim} dims)",
            f"  Segment 3: Relational Context "
            f"[{self.seg3_start}–{self.edge_vector_dim - 1}] "
            f"({self.seg3_total} dims)",
            f"    Namespace:         "
            f"[{self.seg3_namespace_start}–"
            f"{self.seg3_namespace_start + self.seg3_namespace_dim - 1}] "
            f"({self.seg3_namespace_dim} dims)",
            f"    Label Similarity:  "
            f"[{self.seg3_label_similarity_start}–"
            f"{self.seg3_label_similarity_start + self.seg3_label_similarity_dim - 1}] "
            f"({self.seg3_label_similarity_dim} dims)",
            f"    Relation Identity: "
            f"[{self.seg3_relation_identity_start}–"
            f"{self.seg3_relation_identity_start + self.seg3_relation_identity_dim - 1}] "
            f"({self.seg3_relation_identity_dim} dims)",
        ]
        return "\n".join(lines)
