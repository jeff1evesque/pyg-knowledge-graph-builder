"""
Where each part of a node feature vector lives.

``VectorLayout`` turns a ``vector_dim`` into the start index and the width of
every segment and sub-segment. Nothing else computes those numbers: the
encoders in ``feature_extractor.py`` place a value by asking a layout for the
slot, and ``metadata_writer.py`` publishes the same layout in
feature_spec.json and encoding_config.json -- so what a model reads and what
the metadata says a slot holds cannot drift apart.

Pure arithmetic. No Spark, no torch, nothing that runs on an executor.
"""
from typing import Any, Dict, Optional


# ============================================
# Default vector dimension
# ============================================
VECTOR_DIM = 1024

# ============================================
# Segment proportions (fraction of total vector_dim)
# ============================================
# These ratios are applied to any vector_dim to compute boundaries.
#
# Segment 1: Ontology Structure — 25%
#   Sub-segments: class_identity=25%, class_hierarchy=50%, ontology_source=25%
#
# Segment 2: Property Schema — 37.5%
#   Sub-segments: property_presence=50%, domain_range=~29%, property_hierarchy=~21%
#
# Segment 3: Literal Values — 37.5%
#   Sub-segments: numeric=~67%, categorical=~33%
#
# These three are public because feature_extractor.py publishes them as the
# segment_proportions of encoding_config.json. The sub-segment fractions below
# are read only here.
SEG1_FRAC = 0.25
SEG2_FRAC = 0.375
SEG3_FRAC = 0.375  # = 1.0 - 0.25 - 0.375

# Sub-segment fractions within each segment
#
# class_identity holds a multi-hot code per class, and a d-dim segment holds at
# most d linearly independent codes -- so this fraction, not the property count,
# is what bounds how many ontology CLASSES the encoding can represent. At the
# original 0.25 it was 64 dims of a 1024-d vector: enough for the 44 classes of
# a two-source run, but not for the 118 of a full sec+noaa+market+BLS build,
# where identity stopped being recoverable while every code stayed distinct (so
# nothing looked wrong).
#
# The dims come from within segment 1 rather than from vector_dim, which would
# have raised driver memory for every node in the graph. Both donors have slack:
#
#   * class_hierarchy encodes rdfs:subClassOf chains, and no raw source declares
#     any -- it is populated only when --enable_ontology_mapping derives them
#     from class naming and the curated class_mappings table (33 axioms, 21 of
#     86 node types, on the e2e fixtures) and is permanently zero when mapping
#     is off. Which of the two produced a given build is recorded in
#     ontology_schema.json (ontology_mapping_enabled) rather than left to be
#     guessed from an empty list -- see feature_extractor's
#     _collect_ontology_schema_metadata.
#   * ontology_source is `onto_idx % dim` over NAMESPACE_PREFIXES, so it needs
#     only as many dims as there are namespaces (31). At 0.1875 and the default
#     vector_dim it keeps 48 -- collision-free with room for 17 more sources.
#     Below the default it aliases namespaces, as it already did at 256 (16
#     dims); vector_dim is documented as a resolution dial and 1024 is the
#     production recommendation, so that is the existing trade rather than a
#     new one -- but 512 crosses the namespace count where it previously did
#     not (32 dims before, 24 now).
#
# 0.625 gives 160 dims at the default vector_dim. That is NOT a permanent
# answer: segment 1 is a quarter of the vector, so no split of it can carry more
# than ~192 classes, while the class count grows with every source added (SEC
# alone runs to the high hundreds on real filing volume). It buys headroom over
# the 96 classes a full fixture build produces and moves the real lever into
# config -- feature_config.class_identity_dim, or a larger vector_dim -- with
# an over-subscribed build now failing outright rather than shipping features a
# model cannot read. See feature_extractor's _check_class_identity_capacity.
#
# Changing this width changes every class's slots (`hash % dim`), so it
# invalidates models trained against a previous layout. That is why it is a
# tuning constant with a published value in encoding_config.json rather than a
# width recomputed per build -- one that moved whenever a class appeared would
# re-map every existing class for no benefit.
#
# Deriving it per build was tried and removed. Sizing class_identity to the
# class count means taking dims from class_hierarchy and ontology_source, and
# neither has a stated requirement to size against -- ontology_source indexes a
# fixed 26-entry namespace table and is EXPECTED to collide, so "what it needs"
# is not a measurable quantity there. Any split that moves dims off them is a
# guess. The build fails instead, naming the vector_dim that would fit; see
# feature_extractor's _check_class_identity_capacity.
_SEG1_CLASS_IDENTITY_FRAC = 0.625
_SEG1_CLASS_HIERARCHY_FRAC = 0.1875
_SEG1_ONTOLOGY_SOURCE_FRAC = 0.1875

_SEG2_PROPERTY_PRESENCE_FRAC = 0.50
_SEG2_DOMAIN_RANGE_FRAC = 0.29
_SEG2_PROPERTY_HIERARCHY_FRAC = 0.21

_SEG3_NUMERIC_FRAC = 0.67
_SEG3_CATEGORICAL_FRAC = 0.33


class VectorLayout:
    """
    Computes all segment and sub-segment boundaries from a given
    vector_dim. All boundaries are integer dim indices.

    Guarantees:
    - All sub-segments are contiguous and non-overlapping
    - All sub-segments have at least 1 dimension (raises if vector_dim
      is too small)
    - Sub-segment dims sum exactly to vector_dim (no gaps, no overlap)

    Usage:
        layout = VectorLayout(1024)
        layout.seg1_class_identity_start  # 0
        layout.seg1_class_identity_dim    # 160
        layout.seg3_categorical_start     # 896
        layout.seg3_categorical_dim       # 128
        layout.vector_dim                 # 1024

    class_identity_dim overrides the fraction-derived width of the
    class_identity sub-segment. It is the lever for a build whose class count
    exceeds what the default split carries -- see _SEG1_CLASS_IDENTITY_FRAC.
    The dims are taken from the rest of segment 1, so vector_dim and every
    other segment boundary are unchanged and driver memory does not move.
    """

    def __init__(
        self,
        vector_dim: int,
        class_identity_dim: Optional[int] = None,
    ):
        if vector_dim < 32:
            raise ValueError(
                f"vector_dim must be >= 32, got {vector_dim}. "
                f"Minimum needed for all sub-segments to have >= 1 dim."
            )

        self.vector_dim = vector_dim

        # --- Segment boundaries ---
        seg1_total = max(1, int(round(vector_dim * SEG1_FRAC)))
        seg2_total = max(1, int(round(vector_dim * SEG2_FRAC)))
        seg3_total = vector_dim - seg1_total - seg2_total  # remainder

        if seg3_total < 1:
            raise ValueError(
                f"vector_dim={vector_dim} too small: seg3 would have "
                f"{seg3_total} dims"
            )

        self.seg1_start = 0
        self.seg1_total = seg1_total
        self.seg2_start = seg1_total
        self.seg2_total = seg2_total
        self.seg3_start = seg1_total + seg2_total
        self.seg3_total = seg3_total

        # --- Segment 1 sub-segments ---
        ci_dim = max(1, int(round(seg1_total * _SEG1_CLASS_IDENTITY_FRAC)))

        if class_identity_dim is not None:
            # Two dims have to remain for class_hierarchy and ontology_source,
            # which are also indexed into and cannot be zero-width. Rejected
            # rather than clamped: a build that asked for a capacity the vector
            # cannot hold must not quietly get a smaller one, since the whole
            # point of the override is to guarantee a class budget.
            if not 1 <= class_identity_dim <= seg1_total - 2:
                raise ValueError(
                    f"feature_config.class_identity_dim={class_identity_dim} "
                    f"does not fit: segment 1 is {seg1_total} dims at "
                    f"vector_dim={vector_dim}, and class_hierarchy plus "
                    f"ontology_source need at least 1 each, so the maximum is "
                    f"{seg1_total - 2}. Raise feature_config.vector_dim to "
                    f"carry more classes -- segment 1 scales with it."
                )
            ci_dim = class_identity_dim

        ch_dim = max(1, int(round(seg1_total * _SEG1_CLASS_HIERARCHY_FRAC)))
        os_dim = seg1_total - ci_dim - ch_dim  # remainder

        if os_dim < 1:
            os_dim = 1
            ch_dim = seg1_total - ci_dim - os_dim

        self.seg1_class_identity_start = self.seg1_start
        self.seg1_class_identity_dim = ci_dim
        self.seg1_class_hierarchy_start = self.seg1_start + ci_dim
        self.seg1_class_hierarchy_dim = ch_dim
        self.seg1_ontology_source_start = self.seg1_start + ci_dim + ch_dim
        self.seg1_ontology_source_dim = os_dim

        # --- Segment 2 sub-segments ---
        pp_dim = max(1, int(round(seg2_total * _SEG2_PROPERTY_PRESENCE_FRAC)))
        dr_dim = max(1, int(round(seg2_total * _SEG2_DOMAIN_RANGE_FRAC)))
        ph_dim = seg2_total - pp_dim - dr_dim  # remainder

        if ph_dim < 1:
            ph_dim = 1
            dr_dim = seg2_total - pp_dim - ph_dim

        self.seg2_property_presence_start = self.seg2_start
        self.seg2_property_presence_dim = pp_dim
        self.seg2_domain_range_start = self.seg2_start + pp_dim
        self.seg2_domain_range_dim = dr_dim
        self.seg2_property_hierarchy_start = self.seg2_start + pp_dim + dr_dim
        self.seg2_property_hierarchy_dim = ph_dim

        # --- Segment 3 sub-segments ---
        num_dim = max(1, int(round(seg3_total * _SEG3_NUMERIC_FRAC)))
        cat_dim = seg3_total - num_dim  # remainder

        if cat_dim < 1:
            cat_dim = 1
            num_dim = seg3_total - cat_dim

        self.seg3_numeric_start = self.seg3_start
        self.seg3_numeric_dim = num_dim
        self.seg3_categorical_start = self.seg3_start + num_dim
        self.seg3_categorical_dim = cat_dim

        self._validate()

    def _validate(self):
        """Verify all sub-segments tile the full vector with no gaps."""
        total = (
            self.seg1_class_identity_dim
            + self.seg1_class_hierarchy_dim
            + self.seg1_ontology_source_dim
            + self.seg2_property_presence_dim
            + self.seg2_domain_range_dim
            + self.seg2_property_hierarchy_dim
            + self.seg3_numeric_dim
            + self.seg3_categorical_dim
        )
        assert total == self.vector_dim, (
            f"Sub-segment dims sum to {total}, expected {self.vector_dim}"
        )

        # Verify contiguity
        assert self.seg1_class_identity_start == 0
        assert (
            self.seg1_class_hierarchy_start
            == self.seg1_class_identity_start + self.seg1_class_identity_dim
        )
        assert (
            self.seg1_ontology_source_start
            == self.seg1_class_hierarchy_start + self.seg1_class_hierarchy_dim
        )
        assert (
            self.seg2_property_presence_start
            == self.seg1_ontology_source_start + self.seg1_ontology_source_dim
        )
        assert (
            self.seg2_domain_range_start
            == self.seg2_property_presence_start
            + self.seg2_property_presence_dim
        )
        assert (
            self.seg2_property_hierarchy_start
            == self.seg2_domain_range_start + self.seg2_domain_range_dim
        )
        assert (
            self.seg3_numeric_start
            == self.seg2_property_hierarchy_start
            + self.seg2_property_hierarchy_dim
        )
        assert (
            self.seg3_categorical_start
            == self.seg3_numeric_start + self.seg3_numeric_dim
        )
        assert (
            self.seg3_categorical_start + self.seg3_categorical_dim
            == self.vector_dim
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
            "vector_dim": self.vector_dim,
            "seg1_start": self.seg1_start,
            "seg1_total": self.seg1_total,
            "seg1_class_identity_start": self.seg1_class_identity_start,
            "seg1_class_identity_dim": self.seg1_class_identity_dim,
            "seg1_class_hierarchy_start": (
                self.seg1_class_hierarchy_start
            ),
            "seg1_class_hierarchy_dim": self.seg1_class_hierarchy_dim,
            "seg1_ontology_source_start": (
                self.seg1_ontology_source_start
            ),
            "seg1_ontology_source_dim": self.seg1_ontology_source_dim,
            "seg2_start": self.seg2_start,
            "seg2_total": self.seg2_total,
            "seg2_property_presence_start": (
                self.seg2_property_presence_start
            ),
            "seg2_property_presence_dim": (
                self.seg2_property_presence_dim
            ),
            "seg2_domain_range_start": self.seg2_domain_range_start,
            "seg2_domain_range_dim": self.seg2_domain_range_dim,
            "seg2_property_hierarchy_start": (
                self.seg2_property_hierarchy_start
            ),
            "seg2_property_hierarchy_dim": (
                self.seg2_property_hierarchy_dim
            ),
            "seg3_start": self.seg3_start,
            "seg3_total": self.seg3_total,
            "seg3_numeric_start": self.seg3_numeric_start,
            "seg3_numeric_dim": self.seg3_numeric_dim,
            "seg3_categorical_start": self.seg3_categorical_start,
            "seg3_categorical_dim": self.seg3_categorical_dim,
        }

    def summary(self) -> str:
        """Human-readable layout summary."""
        lines = [
            f"VectorLayout(vector_dim={self.vector_dim})",
            f"  Segment 1: Ontology Structure "
            f"[{self.seg1_start}–{self.seg2_start - 1}] "
            f"({self.seg1_total} dims)",
            f"    Class Identity:    "
            f"[{self.seg1_class_identity_start}–"
            f"{self.seg1_class_identity_start + self.seg1_class_identity_dim - 1}] "
            f"({self.seg1_class_identity_dim} dims)",
            f"    Class Hierarchy:   "
            f"[{self.seg1_class_hierarchy_start}–"
            f"{self.seg1_class_hierarchy_start + self.seg1_class_hierarchy_dim - 1}] "
            f"({self.seg1_class_hierarchy_dim} dims)",
            f"    Ontology Source:   "
            f"[{self.seg1_ontology_source_start}–"
            f"{self.seg1_ontology_source_start + self.seg1_ontology_source_dim - 1}] "
            f"({self.seg1_ontology_source_dim} dims)",
            f"  Segment 2: Property Schema "
            f"[{self.seg2_start}–{self.seg3_start - 1}] "
            f"({self.seg2_total} dims)",
            f"    Property Presence: "
            f"[{self.seg2_property_presence_start}–"
            f"{self.seg2_property_presence_start + self.seg2_property_presence_dim - 1}] "
            f"({self.seg2_property_presence_dim} dims)",
            f"    Domain/Range:      "
            f"[{self.seg2_domain_range_start}–"
            f"{self.seg2_domain_range_start + self.seg2_domain_range_dim - 1}] "
            f"({self.seg2_domain_range_dim} dims)",
            f"    Property Hierarchy:"
            f"[{self.seg2_property_hierarchy_start}–"
            f"{self.seg2_property_hierarchy_start + self.seg2_property_hierarchy_dim - 1}] "
            f"({self.seg2_property_hierarchy_dim} dims)",
            f"  Segment 3: Literal Values "
            f"[{self.seg3_start}–{self.vector_dim - 1}] "
            f"({self.seg3_total} dims)",
            f"    Numeric Values:    "
            f"[{self.seg3_numeric_start}–"
            f"{self.seg3_numeric_start + self.seg3_numeric_dim - 1}] "
            f"({self.seg3_numeric_dim} dims)",
            f"    Categorical Values:"
            f"[{self.seg3_categorical_start}–"
            f"{self.seg3_categorical_start + self.seg3_categorical_dim - 1}] "
            f"({self.seg3_categorical_dim} dims)",
        ]
        return "\n".join(lines)
