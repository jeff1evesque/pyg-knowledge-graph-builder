"""
Edge Feature Extractor — Derived Edge Feature Vectors

Constructs fixed-width edge feature vectors by deriving signals from
endpoint node properties during PyG construction. No changes to the
enrichment pipeline or Parquet artifacts are required — all signals
are computed from literal properties already present in the triples.

Edge features are selective: only edge types with meaningful per-instance
variation get feature vectors. Edge types where the relation name alone
carries sufficient signal (e.g., belongsToSector, owl:sameAs) are left
featureless. Within a single edge type, all edges share the same
feature dimensionality (PyG requirement).

Vector structure (default 32-d, configurable via edge_vector_dim):

  Segment 1 — Temporal Signals    (37.5% of dims): time delta between
              endpoints, consecutiveness, same-period flags, temporal
              direction indicators

  Segment 2 — Numeric Contrast    (37.5% of dims): differences, ratios,
              and relative magnitudes of numeric properties shared by
              both endpoints (e.g., moneyness for option-stock edges,
              severity delta for escalation edges)

  Segment 3 — Relational Context  (25% of dims): same-namespace flag,
              label similarity, cross-source indicators, relation type
              identity hash

All segment and sub-segment boundaries scale proportionally with
edge_vector_dim via EdgeVectorLayout. Passing edge_vector_dim=64
doubles resolution; passing edge_vector_dim=16 halves it. The default
32 is recommended for production.

All encoding runs on Spark executors using pure Spark column
expressions. Only compact [num_edges, edge_vector_dim] float32 arrays
are collected to the driver, one edge type at a time.

Architecture — no double-join, no driver round-trip:
  EdgeFeatureExtractor receives the cached resolved edges DataFrame
  directly from EdgeMapper (via constructor.py). The expensive
  double-join (triples × node_id_df) runs exactly once in EdgeMapper.
  A deterministic edge_idx is assigned via Window functions on executors
  using the same sort order (src_id ASC, dst_id ASC) that EdgeMapper
  used for collection, ensuring feature tensor rows align with
  edge_index tensor columns. No data round-trips through the driver.

Driver memory safety:
  Edge feature tensors are much smaller than node features (32-d vs
  1024-d). At 32 dims × 4 bytes × 2M edges, a single edge type's
  tensor is ~256 MB — well within driver memory. For very large edge
  types (>1M edges), chunked collection is available to bound peak
  Pandas memory. Collection is per-edge-type with immediate Pandas
  release and gc.collect() between types.

Which edge types get features (configurable, defaults below):

  High value (default ON):
    - precedes / temporal sequences: time delta, same-year, direction
    - hasUnderlyingPriceObservation: moneyness, time to expiry
    - escalatesTo: severity delta, time between alerts

  Medium value (default OFF, enable via config):
    - correlatesWith: label similarity, same-namespace
    - leadsTo / impacts: estimated lag, cross-source flag
    - straddleWith / spreadWith: strike distance, same-expiry

  No features (always skip):
    - belongsToSector, owl:sameAs, hasParent, rdf:type

Why this requires zero enrichment changes:
  During PyG construction, the EdgeMapper already joins each edge's
  subject and object against the node ID table. At that point, both
  endpoint nodes are resolved. The EdgeFeatureExtractor receives this
  cached resolved edges DataFrame and joins against the literal triples
  to access endpoint node properties — all on Spark executors, all
  without touching a single enrichment module.
"""
import logging
import gc
import time
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_jobs.settle import settle


logger = logging.getLogger(__name__)

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
_EDGE_SEG1_FRAC = 0.375
_EDGE_SEG2_FRAC = 0.375
_EDGE_SEG3_FRAC = 0.25  # = 1.0 - 0.375 - 0.375

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

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_NON_FEATURE_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2002/07/owl#imports",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
}

# ============================================
# Edge type classification
# ============================================
# Relation name fragments that identify temporal edges
_TEMPORAL_RELATION_FRAGMENTS = (
    "precedes", "follows", "hasNext", "hasPrevious",
    "temporallyRelated",
)

# Relation name fragments that identify option-stock edges
_OPTION_STOCK_RELATION_FRAGMENTS = (
    "hasUnderlyingPriceObservation", "hasUnderlying",
)

# Relation name fragments that identify escalation edges
_ESCALATION_RELATION_FRAGMENTS = (
    "escalatesTo", "escalatesFrom", "severityChange",
)

# Relation name fragments that identify correlation edges
#
# "Correlation" is a suffix fragment, not a whole relation name: the
# cross-source linkers emit one relation per sector
# (bls_enrichment_energySectorCorrelation,
# bls_enrichment_employmentSizeSectorCorrelation, ...), and matching only
# the hand-written names below classified all 22 of them "generic" —
# which no enabled_categories set contains, so every one was silently
# dropped from featurization.
_CORRELATION_RELATION_FRAGMENTS = (
    "correlatesWith", "relatedTo", "Correlation",
)

# Relation name fragments that identify causal edges
_CAUSAL_RELATION_FRAGMENTS = (
    "leadsTo", "impacts", "causes", "affects",
)

# Relation name fragments that identify option strategy edges
_STRATEGY_RELATION_FRAGMENTS = (
    "straddleWith", "spreadWith", "strangleWith",
)

# Edge types that should never get features
_SKIP_RELATION_FRAGMENTS = (
    "belongsToSector", "sameAs", "hasParent", "hasChild",
    "equivalentClass", "equivalentProperty", "imports",
    "refersToCompany", "hasRegion", "affectsRegion",
    "sameEventType", "affectsSameRegion",
)

# Every category _classify_relation can return. "skip" is a verdict, not a
# selectable category: structural edges are excluded before enabled_categories
# is consulted, so naming it here would promise something the classifier
# refuses to honour.
_FEATURIZABLE_CATEGORIES = frozenset({
    "temporal", "option_stock", "escalation", "correlation", "causal",
    "strategy", "generic",
})

# Default categories. "generic" is deliberately out — it is the fallback for
# relations no fragment matched, so enabling it featurizes essentially every
# edge type in the graph. It is selectable (see _resolve_enabled_categories),
# just not by default.
#
# "correlation" is IN, despite being off historically: the cross-source linkers
# make correlation the dominant relation family on real data (22 of 46 relations
# and 542 of 704 edge types on a two-source build), so leaving it out meant the
# default config produced a graph with no edge features at all.
#
# "causal" is IN as of #359, and the reason it was not before is that it had
# nothing to featurize. leadsTo is the only relation that classifies causal, and
# it emitted zero triples until #350 woke the BLS leg, then 393,860,192 -- an
# edge per market snapshot -- until #359 moved it onto the sector. It is now 570
# edges in exactly two edge types on the 2026-08-30 graph:
#
#     Grouping -leadsTo-> EquitySector   300
#     Category -leadsTo-> EquitySector   270
#
# Note that enabling a category does not create edge types. Those edges are in
# the graph either way; the category only decides whether they carry an
# edge_attr, so the cost here is two small tensors against the 2,401.9 MB of
# edge features that build already writes. Causal is also the directional
# sibling of correlation, and having one in the default set and not the other
# was only ever an accident of leadsTo being dead.
_DEFAULT_ENABLED_CATEGORIES = (
    "temporal", "option_stock", "escalation", "correlation", "causal",
)

# Hash seeds for deterministic encoding
_EDGE_HASH_SEEDS = [0, 7, 13, 31]

# ============================================
# Driver memory safety constants
# ============================================
_CHUNK_EDGE_THRESHOLD = 1_000_000

# Fraction of spark.driver.maxResultSize one collect may aim for. The estimate
# below counts tensor bytes, not the serialized row overhead Spark actually
# measures, so the remainder is headroom for what it does not model.
_RESULT_SIZE_TARGET_FRACTION = 0.6

# What one slot costs on the wire, which is not what it costs in the tensor.
# A collect returns the SPARSE form -- one row per non-zero slot, carrying
# edge_idx (8) + dim (4) + value (4) + the batch's type id (4) -- so 20 bytes
# travel to deliver 4. Sizing a collect by the dense tensor's 4 bytes
# under-counts it by 5x at worst.
#
# Worst case rather than the measured fill rate, deliberately. On the 2026-08
# graph the edge features are 51.6% full overall and the two market types that
# carry 99.9% of the edges are ~50-53%, so a measured constant would be about
# 2x kinder -- and would silently stop being true when the data changes. The
# cost of assuming full is smaller collects, which is what the chunking is for.
_SPARSE_ENTRY_BYTES = 20

# Upper bound on how many edge types may be unioned into a single
# batched collect.
#
# Small edge types are collected in batches so the driver pays one
# Spark round-trip per batch instead of one per type. The edge budget
# alone is not a sufficient cap: every type contributes its own join
# subtree to the batch's Catalyst plan, so a few hundred tiny types
# would satisfy any edge budget while producing a plan large enough to
# reproduce the plan-size blowup that forced the enrichment phases to
# settle. This caps plan width independently of row count.
_MAX_EDGE_TYPES_PER_BATCH = 8

# Predicate patterns that identify a month or a year property. Hoisted
# out of _encode_temporal_signals so _endpoint_property_rows can count
# exactly the rows that method filters to — a count taken with a
# different pattern would size the wrong frame.
_MONTH_PREDICATE_PATTERN = "(?i)(hasMonth|month|hasMonthNum|monthNumber)"
_YEAR_PREDICATE_PATTERN = "(?i)(hasYear|year|hasYearNum|yearNumber)"

# ============================================
# Broadcast sizing
# ============================================
# Approximate width, in bytes, of one row of each endpoint property
# frame. Used to turn Spark's byte-denominated broadcast threshold into
# a row budget, since a row count is what we can measure cheaply.
#
# The temporal frames carry a long node id and a double value. The
# segment-2 frame carries the predicate URI as well, which is most of
# its width — these are full URIs, not local names.
_TEMPORAL_ROW_BYTES = 24
_NUMERIC_ROW_BYTES = 128

# Spark's own default, used when the session does not report a value.
_DEFAULT_BROADCAST_THRESHOLD = 10 * 1024 * 1024

_BYTE_SUFFIXES = {
    "b": 1,
    "k": 1024, "kb": 1024,
    "m": 1024 ** 2, "mb": 1024 ** 2,
    "g": 1024 ** 3, "gb": 1024 ** 3,
    "t": 1024 ** 4, "tb": 1024 ** 4,
    "p": 1024 ** 5, "pb": 1024 ** 5,
}


def _parse_byte_string(value: str) -> Optional[int]:
    """
    Parse a Spark byte setting ("10MB", "10485760", "-1") into bytes.

    Returns None when the value cannot be read, so the caller can fall
    back rather than treat an unparseable setting as zero.
    """
    text = str(value).strip().lower()
    if not text:
        return None

    for suffix, multiplier in sorted(
        _BYTE_SUFFIXES.items(), key=lambda kv: -len(kv[0])
    ):
        if text.endswith(suffix):
            number = text[: -len(suffix)].strip()
            try:
                return int(float(number) * multiplier)
            except ValueError:
                return None

    try:
        return int(float(text))
    except ValueError:
        return None


class _PropertyRows:
    """
    Row counts of the endpoint property frames for one node type.

    ``month`` and ``year`` are the subsets segment 1 filters to;
    ``total`` is the whole per-type frame segment 2 uses.
    """

    __slots__ = ("total", "month", "year")

    def __init__(self, total: int = 0, month: int = 0, year: int = 0):
        self.total = total
        self.month = month
        self.year = year


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
        seg1_total = max(1, int(round(edge_vector_dim * _EDGE_SEG1_FRAC)))
        seg2_total = max(1, int(round(edge_vector_dim * _EDGE_SEG2_FRAC)))
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

    # Add this method to the existing EdgeVectorLayout class

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


def _classify_relation(relation: str) -> str:
    """
    Classify a relation name into a feature category.

    Returns one of: 'temporal', 'option_stock', 'escalation',
    'correlation', 'causal', 'strategy', 'skip', 'generic'.

    Classification is based on substring matching against known
    relation name fragments. The order of checks matters — skip
    is checked first to ensure structural edges are never featurized.
    """
    rel_lower = relation.lower()

    for frag in _SKIP_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "skip"

    for frag in _TEMPORAL_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "temporal"

    for frag in _OPTION_STOCK_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "option_stock"

    for frag in _ESCALATION_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "escalation"

    for frag in _CORRELATION_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "correlation"

    for frag in _CAUSAL_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "causal"

    for frag in _STRATEGY_RELATION_FRAGMENTS:
        if frag.lower() in rel_lower:
            return "strategy"

    return "generic"


def _resolve_enabled_categories(configured: Any) -> Set[str]:
    """
    Validate and normalize edge_feature_config.enabled_categories.

    A name that no category ever matches is not a harmless typo: the
    extractor logs "Building N-d edge feature vectors", featurizes nothing,
    and the job exits 0 with a graph whose edge_attr is silently absent.
    An hour of cluster time later, the only evidence is a zero in
    graph_schema.json. So reject it at construction instead.

    Args:
        configured: The configured value, or None to take the default.

    Returns:
        The resolved category set.

    Raises:
        ValueError: If a name is not a featurizable category. "skip" is
            called out separately because it is a real classifier verdict
            and the confusion is worth a specific message.
    """
    if configured is None:
        return set(_DEFAULT_ENABLED_CATEGORIES)

    requested = set(configured)
    if "skip" in requested:
        raise ValueError(
            "enabled_categories contains 'skip', which is a classification "
            "verdict rather than a selectable category: structural edges "
            "(sameAs, hasParent, belongsToSector, ...) are excluded before "
            "enabled_categories is consulted and cannot be featurized. "
            f"Remove it. Selectable: {sorted(_FEATURIZABLE_CATEGORIES)}"
        )

    unknown = requested - _FEATURIZABLE_CATEGORIES
    if unknown:
        raise ValueError(
            f"enabled_categories contains unknown categor"
            f"{'ies' if len(unknown) > 1 else 'y'} {sorted(unknown)}. "
            f"No relation ever classifies as such a name, so edge features "
            f"would be silently empty. "
            f"Selectable: {sorted(_FEATURIZABLE_CATEGORIES)}"
        )

    return requested


class EdgeFeatureExtractor:
    """
    Builds fixed-width edge feature vectors derived from endpoint node
    properties.

    All heavy computation runs on Spark executors. Only compact float
    arrays are collected to the driver, one edge type at a time.

    Architecture — no double-join, no driver round-trip:
      Receives the cached resolved edges DataFrame directly from
      EdgeMapper (via constructor.py). The expensive double-join
      (triples × node_id_df) runs exactly once in EdgeMapper. A
      deterministic edge_idx is assigned via Window functions on
      executors using the same sort order (src_id ASC, dst_id ASC)
      that EdgeMapper used for collection, ensuring feature tensor
      rows align with edge_index tensor columns. No data round-trips
      through the driver.

    Edge features are selective — only edge types with meaningful
    per-instance variation get feature vectors. The set of featurized
    edge types is configurable via enabled_categories.

    Driver memory safety:
      - Edge feature tensors are much smaller than node features
        (32-d vs 1024-d, and edges are just pairs)
      - At 32 dims × 4 bytes × 2M edges, a single type is ~256 MB
      - Collection is per-edge-type with immediate Pandas release
      - For very large edge types (>1M edges), chunked collection
        bounds peak Pandas memory
      - gc.collect() between types reclaims fragmented memory
    """

    # Column identifying an edge type inside a batched collect. It travels
    # with every sparse entry, because edge_idx is only unique within a
    # type. One int, not the type triple's three strings — see _flush_batch.
    _TYPE_COL: str = "edge_type_id"

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        edge_feat_config = config.get("edge_feature_config", {})
        self._edge_vector_dim = edge_feat_config.get(
            "edge_vector_dim", EDGE_VECTOR_DIM
        )
        self._enabled = edge_feat_config.get("enabled", True)
        self._chunk_threshold = edge_feat_config.get(
            "chunk_edge_threshold", _CHUNK_EDGE_THRESHOLD
        )
        self._max_types_per_batch = edge_feat_config.get(
            "max_edge_types_per_batch", _MAX_EDGE_TYPES_PER_BATCH
        )
        # (edge_vector_dim, budget) from the last _max_edges_per_collect call.
        # The split loop asks once per edge type -- hundreds of py4j round
        # trips to read a conf value that cannot change mid-build.
        self._edges_per_collect: Optional[Tuple[int, int]] = None

        # Which categories of edge types to featurize. Validated here so a
        # name that can never match fails at construction, not after an
        # hour of cluster time that produced no edge_attr.
        self._enabled_categories: Set[str] = _resolve_enabled_categories(
            edge_feat_config.get("enabled_categories")
        )

        # Compute layout from edge_vector_dim — all segment boundaries
        # scale proportionally
        self._layout = EdgeVectorLayout(self._edge_vector_dim)

        # The bar a property frame has to clear to be broadcast. Read
        # from the session rather than hardcoded, so a deployment that
        # raises Spark's threshold gets the wider hint it asked for.
        self._broadcast_limit = self._read_broadcast_threshold()

    def _read_broadcast_threshold(self) -> int:
        """
        Spark's own byte bar for auto-broadcasting a join side.

        A negative value means broadcasting is disabled, which we honour
        by never hinting.
        """
        # Asked for without a default on purpose: Spark then answers with
        # its own ("10485760b" when nothing set it), so a deployment that
        # configured the threshold is honoured and one that did not still
        # gets the same bar its joins are planned against.
        try:
            raw = self.spark.conf.get(
                "spark.sql.autoBroadcastJoinThreshold"
            )
        except Exception:
            raw = None

        if raw is None:
            return _DEFAULT_BROADCAST_THRESHOLD

        parsed = _parse_byte_string(raw)
        if parsed is None:
            logger.warning(
                f"  Could not read spark.sql.autoBroadcastJoinThreshold "
                f"({raw!r}); using Spark's default of "
                f"{_DEFAULT_BROADCAST_THRESHOLD:,} bytes to size "
                f"endpoint property broadcasts"
            )
            return _DEFAULT_BROADCAST_THRESHOLD

        return parsed

    def _maybe_broadcast(
        self, frame: DataFrame, rows: int, bytes_per_row: int
    ) -> DataFrame:
        """
        Hint a broadcast join only when the frame actually fits.

        The endpoint property frames reach these joins through an
        anti-join and an aggregation, so Catalyst's size estimate for
        them is far too pessimistic and no join would auto-broadcast.
        That is why the hint is here at all. But the hint is a promise,
        not a request: Spark collects the build side to the driver
        whatever its size, and the task results are charged against
        spark.driver.maxResultSize. On a node type with 10M nodes that
        is tens of GB arriving at the driver during a count.

        So hint against a measured row count, and let anything larger
        plan as an ordinary shuffle join.

        Returns the frame with or without the hint.
        """
        if self._broadcast_limit <= 0:
            return frame
        if rows * bytes_per_row > self._broadcast_limit:
            return frame
        return F.broadcast(frame)

    def get_layout(self) -> EdgeVectorLayout:
        """Return the EdgeVectorLayout for metadata registration."""
        return self._layout

    def get_encoding_config(self) -> Dict[str, Any]:
        """
        Return edge feature encoding configuration.

        All values here are the same constants used in the encoding
        methods below. Called by constructor.py after
        build_edge_features() to merge into the combined
        encoding_config.json.
        """
        layout = self._layout
        return {
            "edge_features": {
                "total_dim": layout.edge_vector_dim,
                "enabled_categories": sorted(
                    list(self._enabled_categories)
                ),
                "segment_proportions": {
                    "temporal_signals": _EDGE_SEG1_FRAC,
                    "numeric_contrast": _EDGE_SEG2_FRAC,
                    "relational_context": _EDGE_SEG3_FRAC,
                },
                "temporal_signals": {
                    "time_delta_seed": 0,
                    "abs_time_delta_seed": 7,
                    "normalization_divisor": 12.0,
                },
                "numeric_contrast": {
                    "difference_seed": 500,
                    "ratio_seed": 501,
                    "magnitude_seed": 502,
                    "ratio_clamp": 10.0,
                },
                "cross_property": {
                    "moneyness_seed": 510,
                    "log_moneyness_seed": 511,
                    "strike_stock_diff_seed": 512,
                    "severity_delta_seed": 520,
                },
                "relational_context": {
                    "namespace_seed_offset": 700,
                    "dst_namespace_seed_offset": 710,
                    "label_hash_seed": 800,
                    "category_label_seed": 810,
                    "relation_seed_offset": 900,
                    "category_hash_seed": 950,
                    "relation_weight": 1.0,
                    "category_weight": 0.5,
                },
                "classification_fragments": {
                    "temporal": list(
                        _TEMPORAL_RELATION_FRAGMENTS
                    ),
                    "option_stock": list(
                        _OPTION_STOCK_RELATION_FRAGMENTS
                    ),
                    "escalation": list(
                        _ESCALATION_RELATION_FRAGMENTS
                    ),
                    "correlation": list(
                        _CORRELATION_RELATION_FRAGMENTS
                    ),
                    "causal": list(_CAUSAL_RELATION_FRAGMENTS),
                    "strategy": list(
                        _STRATEGY_RELATION_FRAGMENTS
                    ),
                    "skip": list(_SKIP_RELATION_FRAGMENTS),
                },
            },
        }
        # NOTE: no "checksum" key here -- see the same note in
        # feature_extractor.get_encoding_config(). It was also silently
        # overwriting the node-side one: constructor.py merges the two configs
        # with a shallow dict.update(), so whichever ran last won and the other
        # total never reached encoding_config.json at all.

    def get_edge_classification(
        self,
        edge_indices: Dict[Tuple[str, str, str], "torch.Tensor"],
    ) -> Tuple[
        Dict[str, str],
        List[str],
        List[str],
    ]:
        """
        Classify all edge types and return classification metadata.

        This is a lightweight driver-side operation — just substring
        matching on relation names, ~1 μs per type.

        Args:
            edge_indices: Dict from EdgeMapper

        Returns:
            Tuple of:
            - categories: Dict[relation_name -> category]
            - types_with_features: List of edge type key strings
            - types_without_features: List of edge type key strings
        """
        categories: Dict[str, str] = {}
        types_with: List[str] = []
        types_without: List[str] = []

        for (src, rel, dst) in edge_indices:
            cat = _classify_relation(rel)
            categories[rel] = cat
            key_str = f"({src}, {rel}, {dst})"

            if cat == "skip" or cat not in self._enabled_categories:
                types_without.append(key_str)
            else:
                types_with.append(key_str)

        return categories, types_with, types_without

    def build_edge_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        edges_final_df: DataFrame,
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build edge feature vectors for eligible edge types.

        Uses the cached resolved edges DataFrame from EdgeMapper
        directly — no double-join replay. Joins endpoint literal
        properties on executors to derive per-edge features. Only
        compact float tensors cross to the driver.

        Args:
            triples_df: Enriched triples DataFrame
                (subject, predicate, object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
                on executors
            edges_final_df: Resolved edges DataFrame from EdgeMapper
                (src_type, src_id, relation, dst_type, dst_id) — cached
                on executors. Reused here to avoid replaying the
                expensive double-join.
            edge_indices: Dict from EdgeMapper — maps
                (src_type, relation, dst_type) -> LongTensor[2, N].
                Used only for edge counts and eligibility checking,
                not pushed back to Spark.

        Returns:
            Dict mapping (src_type, relation, dst_type) ->
                FloatTensor[num_edges, edge_vector_dim]
            Only contains entries for edge types that received features.
            Edge types not in this dict should not have edge_attr set
            in HeteroData.
        """
        if not self._enabled:
            logger.info("  Edge features disabled by config")
            return {}

        phase_start = time.monotonic()

        layout = self._layout
        edge_vector_dim = layout.edge_vector_dim

        logger.info(
            f"  Building {edge_vector_dim}-d edge feature vectors"
        )
        logger.info(f"  {layout.summary()}")
        logger.info(
            f"  Enabled categories: {sorted(self._enabled_categories)}"
        )

        # ============================================
        # Classify each edge type — determine which need features
        # ============================================
        eligible_types: List[Tuple[Tuple[str, str, str], str]] = []
        skipped = 0
        # What the graph actually contains, for the diagnostic below. Without
        # it a zero-eligible run cannot be told apart from a graph that has
        # no featurizable edges at all.
        observed: Dict[str, int] = {}

        for edge_type_key in edge_indices:
            src_type, relation, dst_type = edge_type_key
            category = _classify_relation(relation)
            observed[category] = observed.get(category, 0) + 1

            if category == "skip":
                skipped += 1
                continue

            if category not in self._enabled_categories:
                skipped += 1
                continue

            eligible_types.append((edge_type_key, category))

        logger.info(
            f"  {len(eligible_types)} edge types eligible for features, "
            f"{skipped} skipped"
        )

        if not eligible_types:
            # Edge features were asked for and none were produced. Every
            # artifact of this run still looks healthy — exit 0, a valid
            # .pt, graph_schema.json honestly reporting
            # edge_types_with_features: 0 — so nothing else in the pipeline
            # will ever say this out loud.
            missed = sorted(
                (cat for cat in observed if cat != "skip"
                 and cat not in self._enabled_categories),
                key=lambda cat: -observed[cat],
            )
            logger.warning(
                "  NO EDGE TYPES WERE FEATURIZED, but edge features are "
                "enabled. Every edge type in this graph will have no "
                "edge_attr, and the job will still succeed."
            )
            logger.warning(
                f"    enabled categories : {sorted(self._enabled_categories)}"
            )
            logger.warning(
                "    categories present : "
                + ", ".join(
                    f"{cat}={observed[cat]}" for cat in sorted(
                        observed, key=lambda c: -observed[c]
                    )
                )
            )
            if missed:
                logger.warning(
                    f"    add {missed} to edge_feature_config."
                    f"enabled_categories to featurize the "
                    f"{sum(observed[c] for c in missed)} edge type(s) that "
                    f"classified into them"
                )
            return {}

        # ============================================
        # Assign deterministic edge_idx on executors.
        # Same ordering (src_id ASC, dst_id ASC) as EdgeMapper's
        # collection sort, ensuring feature tensor rows align with
        # edge_index tensor columns.
        # ============================================
        logger.info(
            "  Assigning deterministic edge_idx on executors..."
        )

        edges_with_idx = edges_final_df.withColumn(
            "edge_idx",
            (
                F.row_number().over(
                    Window.partitionBy(
                        "src_type", "relation", "dst_type"
                    ).orderBy("src_id", "dst_id")
                )
                - 1
            ).cast("long"),
        )

        # Materialize *and truncate* the logical plan, for the same
        # reason the enrichment phases settle. .cache() materializes
        # the rows but leaves the plan intact, so every per-edge-type
        # query below re-analyzes the full enrichment lineage on the
        # driver. That analysis is pure driver CPU: flat per edge type,
        # independent of how many edges the type has, and unaffected by
        # how the collects are batched or how the joins are planned.
        # It was the dominant cost of this phase.
        edges_with_idx = settle(edges_with_idx)

        # ============================================
        # Pre-extract endpoint literal properties (on executors)
        # These are reused across all edge types.
        # ============================================
        logger.info("  Extracting endpoint literal properties...")

        numeric_props_df = self._extract_node_numeric_properties(
            triples_df, node_id_df
        )
        label_df = self._extract_node_labels(triples_df, node_id_df)

        # Resolve the segment-2 shared/cross-property decision for every
        # edge type in one pass, instead of one probe job per type.
        shared_numeric_types = self._types_with_shared_numerics(
            edges_with_idx, numeric_props_df
        )

        # Size the frames the encoders broadcast, once for the whole
        # phase rather than once per edge type.
        property_rows = self._endpoint_property_rows(numeric_props_df)

        # The encoders filter the numeric frame by endpoint type, and
        # every one of those filters is a scan. Narrow it once so they
        # scan something small -- see _narrow_numeric_props.
        endpoint_types: Set[str] = set()
        for (src_type, _, dst_type), _ in eligible_types:
            endpoint_types.add(src_type)
            endpoint_types.add(dst_type)

        narrow_props_df, narrow_types = self._narrow_numeric_props(
            numeric_props_df, property_rows, endpoint_types
        )

        # ============================================
        # Build features for all eligible edge types
        #
        # Each type's sparse (edge_idx, dim, value) entries are built
        # lazily; the collect is then done in bounded BATCHES rather
        # than one Spark action per type. The per-type driver
        # round-trip previously dominated pipeline wall-clock — the
        # same defect #188 fixed on the node side.
        # ============================================
        edge_features: Dict[Tuple[str, str, str], torch.Tensor] = {}

        collect_start = time.monotonic()

        # (edge_type_key, num_edges, entries_df) awaiting collection
        large: List[Tuple[Tuple[str, str, str], int, DataFrame]] = []
        small: List[Tuple[Tuple[str, str, str], int, DataFrame]] = []

        for edge_type_key, category in eligible_types:
            num_edges = edge_indices[edge_type_key].shape[1]

            logger.info(
                f"    [{edge_type_key}] {num_edges:,} edges, "
                f"category={category}"
            )

            if num_edges == 0:
                edge_features[edge_type_key] = torch.zeros(
                    0, edge_vector_dim, dtype=torch.float32
                )
                continue

            props_df = self._props_for(
                edge_type_key,
                narrow_props_df,
                narrow_types,
                numeric_props_df,
            )

            entries = self._build_edge_type_entries(
                edge_type_key=edge_type_key,
                category=category,
                edges_with_idx=edges_with_idx,
                numeric_props_df=props_df,
                label_df=label_df,
                has_shared_numerics=(
                    edge_type_key in shared_numeric_types
                ),
                property_rows=property_rows,
            )

            if entries is None:
                # No segment produced entries for this type — its
                # feature vector is legitimately all zeros, and no
                # Spark work is needed to say so.
                edge_features[edge_type_key] = torch.zeros(
                    num_edges, edge_vector_dim, dtype=torch.float32
                )
                continue

            # "Large" is whatever will not fit in one collect, which is a
            # question about bytes rather than rows -- the same edge count is
            # a different amount of driver memory at a different
            # edge_vector_dim (#340).
            if num_edges > self._max_edges_per_collect(edge_vector_dim):
                large.append((edge_type_key, num_edges, entries))
            else:
                small.append((edge_type_key, num_edges, entries))

        # Large types stay on the streaming path, collected one at a
        # time by edge_idx range. This preserves the #186 driver-memory
        # guarantee: a single huge type never materialises whole.
        for edge_type_key, num_edges, entries in large:
            logger.info(
                f"    [{edge_type_key}] {num_edges:,} edges "
                f"(chunked collection)"
            )
            tensor = np.zeros(
                (num_edges, edge_vector_dim), dtype=np.float32
            )
            self._collect_and_scatter_chunked(
                self._aggregate_entries(entries),
                tensor,
                num_edges,
                edge_vector_dim,
            )
            edge_features[edge_type_key] = (
                torch.from_numpy(tensor).contiguous()
            )
            gc.collect()

        self._collect_small_batches(
            small, edge_features, edge_vector_dim
        )

        collect_elapsed = time.monotonic() - collect_start

        logger.info(
            f"    Collected {len(small)} small + {len(large)} large "
            f"edge types"
        )

        # Cleanup cached intermediates
        edges_with_idx.unpersist()
        for df in [numeric_props_df, narrow_props_df, label_df]:
            if df is not None:
                try:
                    df.unpersist()
                except Exception:
                    pass

        total_mb = sum(
            t.numel() * 4 / (1024 * 1024)
            for t in edge_features.values()
        )
        logger.info(
            f"  Edge features complete: {len(edge_features)} types, "
            f"{total_mb:.1f} MB total"
        )
        # Explicit phase timings. These replace the old practice of
        # inferring cost from the span between the first and last
        # per-edge-type log line, which stops being measurable once the
        # per-type loop is collapsed into batched collects.
        logger.info(
            f"  [timing] edge feature collect: "
            f"{collect_elapsed:.1f}s "
            f"({len(eligible_types)} eligible types)"
        )
        logger.info(
            f"  [timing] edge feature phase total: "
            f"{time.monotonic() - phase_start:.1f}s"
        )

        return edge_features

    # ================================================================
    # Endpoint property extraction (all on executors)
    # ================================================================

    def _extract_node_numeric_properties(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Extract numeric literal properties for all nodes, keyed by
        (node_type, node_id, predicate, numeric_value).

        Used to derive numeric contrast signals between edge endpoints.

        Isolates literal triples via anti-join against node URIs (same
        pattern as feature_extractor.py), then parses numeric values
        via cast("double"). Aggregates multiple values per
        (node, predicate) pair by mean.

        Returns a cached DataFrame on executors.
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        # Isolate literal triples via anti-join against node URIs.
        # Triples where the object is a known node URI are edge triples,
        # not literal properties.
        literal_triples = triples_df.join(
            node_id_df.select(F.col("uri").alias("_obj_uri")),
            triples_df["object"] == F.col("_obj_uri"),
            "left_anti",
        ).filter(~F.col("predicate").isin(excluded_list))

        # Parse numeric values — strip datatype suffix before casting
        candidates = literal_triples.withColumn(
            "numeric_value",
            F.split(F.col("object"), r"\^\^").getItem(0).cast("double"),
        ).filter(F.col("numeric_value").isNotNull())

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        # Join with node IDs and aggregate per (node, predicate)
        numeric_df = (
            candidates
            .join(
                node_lookup,
                candidates["subject"] == node_lookup["_node_uri"],
                "inner",
            )
            .drop("_node_uri")
            .groupBy("node_type", "node_id", "predicate")
            .agg(F.mean("numeric_value").alias("numeric_value"))
        )

        # Truncate the plan, not just materialize it — this frame is
        # referenced by every edge type's segment-1/2 encoding. See the
        # note on edges_with_idx in build_edge_features.
        numeric_df = settle(numeric_df)
        count = numeric_df.count()
        logger.info(
            f"    Endpoint numeric properties: {count:,} "
            f"(node, property) pairs"
        )

        return numeric_df

    def _endpoint_property_rows(
        self,
        numeric_props_df: DataFrame,
    ) -> Dict[str, _PropertyRows]:
        """
        Count the endpoint property rows each node type contributes.

        The encoders filter this frame by the edge's endpoint type, and
        broadcast the result. How big that result is depends entirely on
        the node type: on real data one type can hold 98% of the graph's
        nodes and tens of millions of property rows, while the next
        holds a few thousand. Nothing in the frame's shape says which,
        so it has to be measured.

        Counted per (type, month, year) in a single pass, because
        segment 1 broadcasts the month and year subsets while segment 2
        broadcasts the whole per-type frame. numeric_props_df is already
        settled, so this is one job for the phase.

        Returns node_type -> _PropertyRows. A type absent from the frame
        is absent here too, and reads as zero rows.
        """
        rows = (
            numeric_props_df
            .groupBy("node_type")
            .agg(
                F.count(F.lit(1)).alias("total_rows"),
                F.sum(
                    F.when(
                        F.col("predicate").rlike(_MONTH_PREDICATE_PATTERN),
                        F.lit(1),
                    ).otherwise(F.lit(0))
                ).alias("month_rows"),
                F.sum(
                    F.when(
                        F.col("predicate").rlike(_YEAR_PREDICATE_PATTERN),
                        F.lit(1),
                    ).otherwise(F.lit(0))
                ).alias("year_rows"),
            )
            .collect()
        )

        sizes = {
            row["node_type"]: _PropertyRows(
                total=row["total_rows"],
                month=row["month_rows"],
                year=row["year_rows"],
            )
            for row in rows
        }

        if sizes:
            widest = max(sizes.items(), key=lambda kv: kv[1].total)
            logger.info(
                f"    Endpoint property frames sized for "
                f"{len(sizes)} node type(s); widest is {widest[0]} at "
                f"{widest[1].total:,} rows (broadcast bar "
                f"{self._broadcast_limit:,} bytes)"
            )

        return sizes

    def _narrow_numeric_props(
        self,
        numeric_props_df: DataFrame,
        property_rows: Dict[str, _PropertyRows],
        endpoint_types: Set[str],
    ) -> Tuple[Optional[DataFrame], Set[str]]:
        """
        Materialise one small frame holding every endpoint type whose
        properties the encoders were already going to broadcast.

        The encoders filter the numeric property frame by endpoint type,
        and each filter is a full scan of it: segment 1 takes four per
        temporal edge type, segment 2 two more. That was affordable while
        the collect ran one edge type at a time, because each scan was
        its own small job. It stopped being affordable when the small
        types were unioned into a single job -- 61 types put several
        hundred scans of a 305,127,393-row frame into one plan, which is
        the ~380 stages the 2026-09-01 run spent 83 minutes on without
        finishing.

        Narrowing does not make the plan narrower; the per-type filters
        still happen. It makes them cheap, by giving them a frame of a
        few hundred thousand rows to scan instead of three hundred
        million.

        A type is included when its rows fit the broadcast bar -- the
        same measured test _maybe_broadcast applies -- so the frame holds
        exactly the types whose joins were meant to be broadcasts. The
        few above the bar (on the 2026-08 graph only
        market_quotes_OptionSnapshot, at 304,600,530 rows) keep the full
        frame and the shuffle joins they were already planning.

        Returns (frame, the types it holds). The frame is None when no
        type qualifies, and the caller then uses the full frame
        throughout.
        """
        if self._broadcast_limit <= 0:
            return None, set()

        # A type absent from property_rows contributes no rows, so it
        # reads as zero here and is included -- the narrow frame holds
        # nothing for it, which is what the full frame holds too.
        small_types = {
            node_type
            for node_type in endpoint_types
            if property_rows.get(node_type, _PropertyRows()).total
            * _NUMERIC_ROW_BYTES <= self._broadcast_limit
        }

        if not small_types:
            return None, set()

        narrow = settle(
            numeric_props_df.filter(
                F.col("node_type").isin(sorted(small_types))
            )
        )

        kept = narrow.count()
        logger.info(
            f"    Endpoint numeric properties narrowed to "
            f"{len(small_types)} of {len(endpoint_types)} endpoint "
            f"type(s): {kept:,} rows, which is what the per-type filters "
            f"now scan"
        )

        return narrow, small_types

    def _props_for(
        self,
        edge_type_key: Tuple[str, str, str],
        narrow_props_df: Optional[DataFrame],
        narrow_types: Set[str],
        numeric_props_df: DataFrame,
    ) -> DataFrame:
        """
        Pick the numeric property frame one edge type is allowed to read.

        Both endpoints have to be in the narrow frame. A type that is
        missing from it reads as having no properties at all, so this
        edge type's segment-1 and segment-2 features would come back
        zeros -- wrong, and with nothing in the log to say so. Falling
        back to the full frame costs that one type its scans and keeps
        the answer right.
        """
        if narrow_props_df is None:
            return numeric_props_df

        src_type, _, dst_type = edge_type_key
        if src_type in narrow_types and dst_type in narrow_types:
            return narrow_props_df

        return numeric_props_df

    def _extract_node_labels(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Extract rdfs:label values for all nodes, keyed by
        (node_type, node_id, label).

        Used for label similarity signals on correlation and causal
        edges. Labels are lowercased for case-insensitive word overlap
        computation.

        Returns a cached DataFrame on executors.
        """
        rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        label_df = (
            triples_df
            .filter(F.col("predicate") == rdfs_label)
            .join(
                node_lookup,
                triples_df["subject"] == node_lookup["_node_uri"],
                "inner",
            )
            .drop("_node_uri")
            .select(
                "node_type",
                "node_id",
                F.lower(F.col("object")).alias("label"),
            )
            .distinct()
        )

        # Plan truncation, same rationale as numeric_df above.
        label_df = settle(label_df)
        count = label_df.count()
        logger.info(f"    Endpoint labels: {count:,} (node, label) pairs")

        return label_df

    def _types_with_shared_numerics(
        self,
        edges_with_idx: DataFrame,
        numeric_props_df: DataFrame,
    ) -> Set[Tuple[str, str, str]]:
        """
        Determine, in ONE distributed pass, which edge types have at
        least one edge whose endpoints share a numeric predicate.

        Segment 2 needs this to choose between the shared-property
        encoding and the cross-property fallback. It used to be answered
        per edge type with a `head(1)` probe, which forced that type's
        endpoint joins to execute — one Spark action per type, and the
        dominant cost of the whole edge-feature phase (measured ~6s per
        type on the e2e fixtures, flat regardless of edge count).
        Answering it once for every type is the same question asked in
        one job instead of N.

        Equivalence with the old probe: for a given type, the probe
        joined that type's edges to its endpoints' numerics on src_id,
        then on dst_id AND matching predicate. Because every edge of a
        type shares the same (src_type, dst_type), filtering the
        property table by node_type and joining on node_id alone is the
        same as joining on the (node_type, node_id) pair — which is what
        this does for all types at once.

        Returns the set of (src_type, relation, dst_type) keys with at
        least one shared-predicate edge.
        """
        src_numerics = numeric_props_df.select(
            F.col("node_type").alias("_s_nt"),
            F.col("node_id").alias("_s_nid"),
            F.col("predicate").alias("_s_pred"),
        )
        dst_numerics = numeric_props_df.select(
            F.col("node_type").alias("_d_nt"),
            F.col("node_id").alias("_d_nid"),
            F.col("predicate").alias("_d_pred"),
        )

        shared = (
            edges_with_idx
            .join(
                src_numerics,
                (F.col("src_type") == F.col("_s_nt"))
                & (F.col("src_id") == F.col("_s_nid")),
                "inner",
            )
            .join(
                dst_numerics,
                (F.col("dst_type") == F.col("_d_nt"))
                & (F.col("dst_id") == F.col("_d_nid"))
                & (F.col("_s_pred") == F.col("_d_pred")),
                "inner",
            )
            .select("src_type", "relation", "dst_type")
            .distinct()
        )

        keys = {
            (row["src_type"], row["relation"], row["dst_type"])
            for row in shared.collect()
        }
        logger.info(
            f"    {len(keys)} edge types have endpoint-shared numeric "
            f"properties (resolved in one pass)"
        )
        return keys

    # ================================================================
    # Per-edge-type vector assembly (on executors, collect to driver)
    # ================================================================

    def _build_edge_type_entries(
        self,
        edge_type_key: Tuple[str, str, str],
        category: str,
        edges_with_idx: DataFrame,
        numeric_props_df: DataFrame,
        label_df: DataFrame,
        has_shared_numerics: bool,
        property_rows: Dict[str, _PropertyRows],
    ) -> Optional[DataFrame]:
        """
        Build the lazy sparse entries for all edges of one type.

        Each segment is computed on executors as a DataFrame of
        (edge_idx, dim, value) sparse entries, which are unioned but
        deliberately NOT aggregated or collected here. The caller unions
        several types together and issues one grouped collect per batch,
        so the driver round-trip is paid once per batch rather than once
        per edge type.

        No data round-trips through the driver — edges_with_idx was
        received from EdgeMapper's cache, not reconstructed.

        Returns DataFrame(edge_idx, dim, value), or None when no segment
        produced entries (caller supplies a zero tensor).
        """
        src_type, relation, dst_type = edge_type_key

        # ============================================
        # Filter resolved edges to this type (on executors)
        # ============================================
        edge_df = (
            edges_with_idx
            .filter(
                (F.col("src_type") == src_type)
                & (F.col("relation") == relation)
                & (F.col("dst_type") == dst_type)
            )
            .select("edge_idx", "src_id", "dst_id",
                     "src_type", "dst_type")
        )

        # ============================================
        # Compute all three segments (lazy on executors)
        # ============================================
        src_rows = property_rows.get(src_type, _PropertyRows())
        dst_rows = property_rows.get(dst_type, _PropertyRows())

        seg1_entries = self._encode_temporal_signals(
            edge_df=edge_df,
            src_type=src_type,
            dst_type=dst_type,
            category=category,
            numeric_props_df=numeric_props_df,
            src_rows=src_rows,
            dst_rows=dst_rows,
        )

        seg2_entries = self._encode_numeric_contrast(
            edge_df=edge_df,
            src_type=src_type,
            dst_type=dst_type,
            category=category,
            numeric_props_df=numeric_props_df,
            has_shared_numerics=has_shared_numerics,
            src_rows=src_rows,
            dst_rows=dst_rows,
        )

        seg3_entries = self._encode_relational_context(
            edge_df=edge_df,
            src_type=src_type,
            dst_type=dst_type,
            relation=relation,
            category=category,
            label_df=label_df,
        )

        # ============================================
        # Union all segments
        # ============================================
        all_parts = [
            df for df in [seg1_entries, seg2_entries, seg3_entries]
            if df is not None
        ]

        if not all_parts:
            return None

        combined = all_parts[0]
        for df in all_parts[1:]:
            combined = combined.unionAll(df)

        return combined

    # ================================================================
    # Segment 1: Temporal Signals encoding
    # ================================================================

    def _encode_temporal_signals(
        self,
        edge_df: DataFrame,
        src_type: str,
        dst_type: str,
        category: str,
        numeric_props_df: DataFrame,
        src_rows: _PropertyRows,
        dst_rows: _PropertyRows,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 1: time delta, period flags, and temporal
        direction between edge endpoints.

        For temporal edges: computes month/year delta from endpoint
        properties. Month and year values are extracted by regex-matching
        predicate URIs containing "month" or "year" patterns.

        For non-temporal edges: encodes a hash of the relation category
        into the time delta sub-segment as a type indicator, so the GNN
        can still distinguish edge categories in this segment.

        All dim indices are derived from self._layout.

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Time Delta ---
        td_start = layout.seg1_time_delta_start
        td_dim = layout.seg1_time_delta_dim

        if category == "temporal":
            # Find month/year numeric properties by predicate pattern.
            # These patterns match predicates like cpi:hasMonth,
            # market:monthNumber, etc.
            month_preds = numeric_props_df.filter(
                F.col("predicate").rlike(_MONTH_PREDICATE_PATTERN)
            )
            year_preds = numeric_props_df.filter(
                F.col("predicate").rlike(_YEAR_PREDICATE_PATTERN)
            )

            # Source endpoint temporal properties
            src_month = (
                month_preds
                .filter(F.col("node_type") == src_type)
                .select(
                    F.col("node_id").alias("_src_nid"),
                    F.col("numeric_value").alias("src_month"),
                )
            )
            src_year = (
                year_preds
                .filter(F.col("node_type") == src_type)
                .select(
                    F.col("node_id").alias("_src_nid"),
                    F.col("numeric_value").alias("src_year"),
                )
            )

            # Destination endpoint temporal properties
            dst_month = (
                month_preds
                .filter(F.col("node_type") == dst_type)
                .select(
                    F.col("node_id").alias("_dst_nid"),
                    F.col("numeric_value").alias("dst_month"),
                )
            )
            dst_year = (
                year_preds
                .filter(F.col("node_type") == dst_type)
                .select(
                    F.col("node_id").alias("_dst_nid"),
                    F.col("numeric_value").alias("dst_year"),
                )
            )

            # Join temporal properties to edges (on executors).
            # Left joins ensure edges without temporal properties
            # still get zero-valued features rather than being dropped.
            # The endpoint property side is broadcast when it is small
            # enough to be — see _maybe_broadcast. For a node type of a
            # few thousand nodes that is every time, and it saves one
            # shuffle stage per join per edge type. For a node type of
            # ten million it is never, and the join shuffles.
            temporal_edges = (
                edge_df
                .join(
                    self._maybe_broadcast(
                        src_month, src_rows.month, _TEMPORAL_ROW_BYTES
                    ),
                    edge_df["src_id"] == src_month["_src_nid"],
                    "left",
                )
                .drop("_src_nid")
                .join(
                    self._maybe_broadcast(
                        src_year, src_rows.year, _TEMPORAL_ROW_BYTES
                    ),
                    edge_df["src_id"] == src_year["_src_nid"],
                    "left",
                )
                .drop("_src_nid")
                .join(
                    self._maybe_broadcast(
                        dst_month, dst_rows.month, _TEMPORAL_ROW_BYTES
                    ),
                    edge_df["dst_id"] == dst_month["_dst_nid"],
                    "left",
                )
                .drop("_dst_nid")
                .join(
                    self._maybe_broadcast(
                        dst_year, dst_rows.year, _TEMPORAL_ROW_BYTES
                    ),
                    edge_df["dst_id"] == dst_year["_dst_nid"],
                    "left",
                )
                .drop("_dst_nid")
            )

            # Compute month delta: (dst_year - src_year) * 12
            #                       + (dst_month - src_month)
            # This gives a signed integer representing the number of
            # months between endpoints. Positive = dst is later.
            temporal_edges = temporal_edges.withColumn(
                "month_delta",
                F.when(
                    F.col("src_year").isNotNull()
                    & F.col("dst_year").isNotNull()
                    & F.col("src_month").isNotNull()
                    & F.col("dst_month").isNotNull(),
                    (
                        (F.col("dst_year") - F.col("src_year"))
                        * F.lit(12.0)
                        + (F.col("dst_month") - F.col("src_month"))
                    ),
                ).otherwise(F.lit(0.0)),
            )

            # Normalize month delta: divide by 12 to put in
            # ~[-1, 1] range for typical monthly data.
            # A 1-month gap = 0.083, a 1-year gap = 1.0.
            temporal_edges = temporal_edges.withColumn(
                "norm_delta",
                F.col("month_delta") / F.lit(12.0),
            )

            # Signed normalized delta in a hashed slot
            td_encoded = temporal_edges.select(
                F.col("edge_idx"),
                (
                    F.abs(F.hash(F.lit("time_delta"), F.lit(0)))
                    % F.lit(td_dim)
                    + F.lit(td_start)
                ).alias("dim"),
                F.col("norm_delta").alias("value"),
            )
            parts.append(td_encoded)

            # Absolute delta in a second slot — useful for the GNN
            # to learn "how far apart" regardless of direction
            if td_dim >= 2:
                td_abs_encoded = temporal_edges.select(
                    F.col("edge_idx"),
                    (
                        F.abs(F.hash(F.lit("abs_time_delta"), F.lit(7)))
                        % F.lit(td_dim)
                        + F.lit(td_start)
                    ).alias("dim"),
                    F.abs(F.col("norm_delta")).alias("value"),
                )
                parts.append(td_abs_encoded)

            # --- Sub-segment 1b: Period Flags ---
            pf_start = layout.seg1_period_flags_start
            pf_dim = layout.seg1_period_flags_dim

            # Same-year flag: 1.0 if both endpoints are in the same year
            same_year = temporal_edges.select(
                F.col("edge_idx"),
                F.lit(pf_start).alias("dim"),
                F.when(
                    F.col("src_year").isNotNull()
                    & F.col("dst_year").isNotNull()
                    & (F.col("src_year") == F.col("dst_year")),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)).alias("value"),
            )
            parts.append(same_year)

            # Consecutive-month flag: 1.0 if endpoints are exactly
            # 1 month apart (the most common temporal edge pattern)
            if pf_dim >= 2:
                consecutive = temporal_edges.select(
                    F.col("edge_idx"),
                    F.lit(pf_start + 1).alias("dim"),
                    F.when(
                        F.abs(F.col("month_delta")) == F.lit(1.0),
                        F.lit(1.0),
                    ).otherwise(F.lit(0.0)).alias("value"),
                )
                parts.append(consecutive)

            # Same-quarter flag: 1.0 if both endpoints are in the
            # same calendar quarter and year
            if pf_dim >= 3:
                same_quarter = temporal_edges.select(
                    F.col("edge_idx"),
                    F.lit(pf_start + 2).alias("dim"),
                    F.when(
                        F.col("src_month").isNotNull()
                        & F.col("dst_month").isNotNull()
                        & (
                            F.floor(
                                (F.col("src_month") - 1) / F.lit(3.0)
                            )
                            == F.floor(
                                (F.col("dst_month") - 1) / F.lit(3.0)
                            )
                        )
                        & (F.col("src_year") == F.col("dst_year")),
                        F.lit(1.0),
                    ).otherwise(F.lit(0.0)).alias("value"),
                )
                parts.append(same_quarter)

            # --- Sub-segment 1c: Direction ---
            dir_start = layout.seg1_direction_start

            # Forward direction indicator: +1 if dst is later in time,
            # -1 if earlier, 0 if same or unknown
            direction = temporal_edges.select(
                F.col("edge_idx"),
                F.lit(dir_start).alias("dim"),
                F.when(
                    F.col("month_delta") > 0, F.lit(1.0)
                ).when(
                    F.col("month_delta") < 0, F.lit(-1.0)
                ).otherwise(F.lit(0.0)).alias("value"),
            )
            parts.append(direction)

        else:
            # Non-temporal edges: encode category as a hash indicator
            # in the time delta sub-segment so the GNN can still
            # distinguish edge categories in this segment
            cat_indicator = edge_df.select(
                F.col("edge_idx"),
                (
                    F.abs(F.hash(F.lit(category), F.lit(50)))
                    % F.lit(td_dim)
                    + F.lit(td_start)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(cat_indicator)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("edge_idx").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Segment 2: Numeric Contrast encoding
    # ================================================================

    def _encode_numeric_contrast(
        self,
        edge_df: DataFrame,
        src_type: str,
        dst_type: str,
        category: str,
        numeric_props_df: DataFrame,
        has_shared_numerics: bool,
        src_rows: _PropertyRows,
        dst_rows: _PropertyRows,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 2: numeric differences, ratios, and magnitudes
        between edge endpoints.

        For each numeric property shared by both endpoints (same
        predicate URI), computes:
        - Difference (dst - src) hashed into difference sub-segment
        - Ratio (dst / src, clamped to [-10, 10]) hashed into ratio
          sub-segment
        - Average magnitude hashed into magnitude sub-segment

        If no shared properties exist, falls back to cross-property
        derivation for specific categories (e.g., moneyness for
        option-stock edges, severity delta for escalation edges).

        All dim indices are derived from self._layout.

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        if not has_shared_numerics:
            # No edge of this type has endpoints sharing a numeric
            # predicate — derive contrast across different properties
            # instead. Decided once for all types by
            # _types_with_shared_numerics, so this branch costs no
            # Spark job of its own.
            return self._encode_cross_property_contrast(
                edge_df, src_type, dst_type, category, numeric_props_df
            )

        # Get numeric properties for source and destination types
        src_numerics = (
            numeric_props_df
            .filter(F.col("node_type") == src_type)
            .select(
                F.col("node_id").alias("_src_nid"),
                F.col("predicate").alias("src_pred"),
                F.col("numeric_value").alias("src_val"),
            )
        )

        dst_numerics = (
            numeric_props_df
            .filter(F.col("node_type") == dst_type)
            .select(
                F.col("node_id").alias("_dst_nid"),
                F.col("predicate").alias("dst_pred"),
                F.col("numeric_value").alias("dst_val"),
            )
        )

        # Join edge endpoints with their numeric properties.
        # Inner join on predicate ensures we only get properties
        # that both endpoints share.
        # Broadcast on the same terms as the segment-1 joins, and note
        # that these frames are the widest of the six: no predicate
        # filter narrows them, so a type contributes every numeric
        # property of every one of its nodes.
        edge_with_src = (
            edge_df
            .join(
                self._maybe_broadcast(
                    src_numerics, src_rows.total, _NUMERIC_ROW_BYTES
                ),
                edge_df["src_id"] == src_numerics["_src_nid"],
                "inner",
            )
            .drop("_src_nid")
        )

        edge_with_both = (
            edge_with_src
            .join(
                self._maybe_broadcast(
                    dst_numerics, dst_rows.total, _NUMERIC_ROW_BYTES
                ),
                (edge_with_src["dst_id"] == dst_numerics["_dst_nid"])
                & (edge_with_src["src_pred"] == dst_numerics["dst_pred"]),
                "inner",
            )
            .drop("_dst_nid", "dst_pred")
            .withColumnRenamed("src_pred", "predicate")
        )

        # --- Sub-segment 2a: Difference (dst - src) ---
        diff_start = layout.seg2_difference_start
        diff_dim = layout.seg2_difference_dim

        diff_entries = edge_with_both.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.col("predicate"), F.lit(500)))
                % F.lit(diff_dim)
                + F.lit(diff_start)
            ).alias("dim"),
            (F.col("dst_val") - F.col("src_val")).alias("value"),
        )
        parts.append(diff_entries)

        # --- Sub-segment 2b: Ratio (dst / src, clamped) ---
        rat_start = layout.seg2_ratio_start
        rat_dim = layout.seg2_ratio_dim

        # Safe ratio: clamp to [-10, 10] to avoid extreme values
        # from near-zero denominators. Division by zero returns 0.0.
        ratio_entries = edge_with_both.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.col("predicate"), F.lit(501)))
                % F.lit(rat_dim)
                + F.lit(rat_start)
            ).alias("dim"),
            F.when(
                F.abs(F.col("src_val")) > F.lit(1e-8),
                F.greatest(
                    F.lit(-10.0),
                    F.least(
                        F.lit(10.0),
                        F.col("dst_val") / F.col("src_val"),
                    ),
                ),
            ).otherwise(F.lit(0.0)).alias("value"),
        )
        parts.append(ratio_entries)

        # --- Sub-segment 2c: Magnitude (average absolute value) ---
        mag_start = layout.seg2_magnitude_start
        mag_dim = layout.seg2_magnitude_dim

        mag_entries = edge_with_both.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.col("predicate"), F.lit(502)))
                % F.lit(mag_dim)
                + F.lit(mag_start)
            ).alias("dim"),
            (
                (F.abs(F.col("src_val")) + F.abs(F.col("dst_val")))
                / F.lit(2.0)
            ).alias("value"),
        )
        parts.append(mag_entries)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("edge_idx").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    def _encode_cross_property_contrast(
        self,
        edge_df: DataFrame,
        src_type: str,
        dst_type: str,
        category: str,
        numeric_props_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Derive numeric contrast from different properties on source
        and destination endpoints for specific edge categories.

        This handles the case where endpoints don't share the same
        predicate URI but have semantically related properties that
        can be meaningfully compared:

        For option_stock edges:
          - Source (option node): strikePrice property
          - Destination (stock node): observedPrice property
          - Derived: moneyness = strike / stock_price
          - Derived: log-moneyness = log(moneyness)
          - Derived: strike-stock difference

        For escalation edges:
          - Source (alert node): severity/severityLevel property
          - Destination (alert node): severity/severityLevel property
          - Derived: severity delta = dst_severity - src_severity

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        if category == "option_stock":
            # Source (option): look for strike price property
            src_strike = (
                numeric_props_df
                .filter(F.col("node_type") == src_type)
                .filter(
                    F.col("predicate").rlike(
                        "(?i)(strikePrice|strike)"
                    )
                )
                .select(
                    F.col("node_id").alias("_src_nid"),
                    F.col("numeric_value").alias("strike_price"),
                )
            )

            # Destination (stock): look for observed price property
            dst_price = (
                numeric_props_df
                .filter(F.col("node_type") == dst_type)
                .filter(
                    F.col("predicate").rlike(
                        "(?i)(observedPrice|price|lastPrice|closePrice)"
                    )
                )
                .select(
                    F.col("node_id").alias("_dst_nid"),
                    F.col("numeric_value").alias("stock_price"),
                )
            )

            moneyness_edges = (
                edge_df
                .join(
                    src_strike,
                    edge_df["src_id"] == src_strike["_src_nid"],
                    "inner",
                )
                .drop("_src_nid")
                .join(
                    dst_price,
                    edge_df["dst_id"] == dst_price["_dst_nid"],
                    "inner",
                )
                .drop("_dst_nid")
            )

            if moneyness_edges.head(1):
                diff_start = layout.seg2_difference_start
                diff_dim = layout.seg2_difference_dim

                # Moneyness = strike / stock_price
                # ATM ≈ 1.0, ITM call < 1.0, OTM call > 1.0
                moneyness = moneyness_edges.withColumn(
                    "moneyness",
                    F.when(
                        F.abs(F.col("stock_price")) > F.lit(1e-8),
                        F.col("strike_price") / F.col("stock_price"),
                    ).otherwise(F.lit(1.0)),
                )

                # Moneyness in difference sub-segment
                m_entry = moneyness.select(
                    F.col("edge_idx"),
                    (
                        F.abs(F.hash(F.lit("moneyness"), F.lit(510)))
                        % F.lit(diff_dim)
                        + F.lit(diff_start)
                    ).alias("dim"),
                    F.col("moneyness").alias("value"),
                )
                parts.append(m_entry)

                # Log-moneyness in ratio sub-segment — more useful
                # for GNNs because it's symmetric around ATM (log(1)=0)
                rat_start = layout.seg2_ratio_start
                rat_dim = layout.seg2_ratio_dim

                log_m = moneyness.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(F.lit("log_moneyness"), F.lit(511))
                        )
                        % F.lit(rat_dim)
                        + F.lit(rat_start)
                    ).alias("dim"),
                    F.log(
                        F.greatest(F.col("moneyness"), F.lit(1e-8))
                    ).alias("value"),
                )
                parts.append(log_m)

                # Strike-stock absolute difference in magnitude
                # sub-segment — raw dollar distance
                mag_start = layout.seg2_magnitude_start
                mag_dim = layout.seg2_magnitude_dim

                price_diff = moneyness.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(
                                F.lit("strike_stock_diff"), F.lit(512)
                            )
                        )
                        % F.lit(mag_dim)
                        + F.lit(mag_start)
                    ).alias("dim"),
                    (
                        F.col("strike_price") - F.col("stock_price")
                    ).alias("value"),
                )
                parts.append(price_diff)

        elif category == "escalation":
            # Look for severity-related numeric properties on both
            # endpoints. NOAA alerts may encode severity as a numeric
            # level (1=Minor, 2=Moderate, 3=Severe, 4=Extreme).
            src_severity = (
                numeric_props_df
                .filter(F.col("node_type") == src_type)
                .filter(
                    F.col("predicate").rlike(
                        "(?i)(severity|severityLevel|severityNum)"
                    )
                )
                .select(
                    F.col("node_id").alias("_src_nid"),
                    F.col("numeric_value").alias("src_severity"),
                )
            )

            dst_severity = (
                numeric_props_df
                .filter(F.col("node_type") == dst_type)
                .filter(
                    F.col("predicate").rlike(
                        "(?i)(severity|severityLevel|severityNum)"
                    )
                )
                .select(
                    F.col("node_id").alias("_dst_nid"),
                    F.col("numeric_value").alias("dst_severity"),
                )
            )

            sev_edges = (
                edge_df
                .join(
                    src_severity,
                    edge_df["src_id"] == src_severity["_src_nid"],
                    "inner",
                )
                .drop("_src_nid")
                .join(
                    dst_severity,
                    edge_df["dst_id"] == dst_severity["_dst_nid"],
                    "inner",
                )
                .drop("_dst_nid")
            )

            if sev_edges.head(1):
                diff_start = layout.seg2_difference_start
                diff_dim = layout.seg2_difference_dim

                # Severity delta: positive means escalation (dst is
                # more severe), negative means de-escalation
                sev_delta = sev_edges.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(F.lit("severity_delta"), F.lit(520))
                        )
                        % F.lit(diff_dim)
                        + F.lit(diff_start)
                    ).alias("dim"),
                    (
                        F.col("dst_severity") - F.col("src_severity")
                    ).alias("value"),
                )
                parts.append(sev_delta)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("edge_idx").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Segment 3: Relational Context encoding
    # ================================================================

    def _encode_relational_context(
        self,
        edge_df: DataFrame,
        src_type: str,
        dst_type: str,
        relation: str,
        category: str,
        label_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 3: namespace signals, label similarity, and
        relation type identity.

        Namespace signals are derived from the PyG node type name
        prefix (e.g., "cpi_Index" → "cpi"). This is a driver-side
        string operation on the type name, not a Spark UDF — the
        result is broadcast as a literal to all executors.

        Label similarity is computed for correlation and causal edges
        using Jaccard word overlap between endpoint rdfs:label values.
        For other edge types, a category indicator hash is used instead.

        Relation identity hashes the relation name and category into
        fixed slots, giving the GNN a fingerprint of the relationship
        type within the vector.

        All dim indices are derived from self._layout.

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Namespace Signals ---
        ns_start = layout.seg3_namespace_start
        ns_dim = layout.seg3_namespace_dim

        # Same-namespace flag: are src and dst from the same ontology?
        # Derive from node_type prefix (e.g., "cpi_Index" → "cpi").
        # This is a driver-side string operation — the result is
        # broadcast as F.lit() to all executors.
        src_prefix = (
            src_type.split("_")[0] if "_" in src_type else src_type
        )
        dst_prefix = (
            dst_type.split("_")[0] if "_" in dst_type else dst_type
        )
        same_ns = 1.0 if src_prefix == dst_prefix else 0.0

        same_ns_entry = edge_df.select(
            F.col("edge_idx"),
            F.lit(ns_start).alias("dim"),
            F.lit(same_ns).alias("value"),
        )
        parts.append(same_ns_entry)

        # Cross-source flag (inverse of same-namespace)
        if ns_dim >= 2:
            cross_source_entry = edge_df.select(
                F.col("edge_idx"),
                F.lit(ns_start + 1).alias("dim"),
                F.lit(1.0 - same_ns).alias("value"),
            )
            parts.append(cross_source_entry)

        # Hash source and destination namespace prefixes into remaining
        # namespace slots for finer-grained namespace identity
        if ns_dim >= 3:
            for seed_offset in _EDGE_HASH_SEEDS[:2]:
                src_ns_hash = edge_df.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(
                                F.lit(src_prefix),
                                F.lit(seed_offset + 700),
                            )
                        )
                        % F.lit(max(1, ns_dim - 2))
                        + F.lit(ns_start + 2)
                    ).alias("dim"),
                    F.lit(1.0).alias("value"),
                )
                parts.append(src_ns_hash)

                dst_ns_hash = edge_df.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(
                                F.lit(dst_prefix),
                                F.lit(seed_offset + 710),
                            )
                        )
                        % F.lit(max(1, ns_dim - 2))
                        + F.lit(ns_start + 2)
                    ).alias("dim"),
                    F.lit(0.5).alias("value"),
                )
                parts.append(dst_ns_hash)

        # --- Sub-segment 3b: Label Similarity ---
        ls_start = layout.seg3_label_similarity_start
        ls_dim = layout.seg3_label_similarity_dim

        if category in ("correlation", "causal"):
            # Join labels for both endpoints to compute word overlap.
            # This distinguishes strong correlations (exact keyword
            # match like "Energy" ↔ "Energy") from weak ones
            # ("Food at Home" ↔ "Food Manufacturing").
            src_labels = (
                label_df
                .filter(F.col("node_type") == src_type)
                .select(
                    F.col("node_id").alias("_src_nid"),
                    F.col("label").alias("src_label"),
                )
            )
            dst_labels = (
                label_df
                .filter(F.col("node_type") == dst_type)
                .select(
                    F.col("node_id").alias("_dst_nid"),
                    F.col("label").alias("dst_label"),
                )
            )

            label_edges = (
                edge_df
                .join(
                    src_labels,
                    edge_df["src_id"] == src_labels["_src_nid"],
                    "left",
                )
                .drop("_src_nid")
                .join(
                    dst_labels,
                    edge_df["dst_id"] == dst_labels["_dst_nid"],
                    "left",
                )
                .drop("_dst_nid")
            )

            # Compute Jaccard word overlap as a simple similarity
            # measure: |intersection| / |union| of word sets.
            # This is a rough approximation — good enough for a
            # feature signal. All computed via Spark array functions
            # on executors.
            label_edges = (
                label_edges
                .withColumn(
                    "src_words",
                    F.split(
                        F.coalesce(F.col("src_label"), F.lit("")),
                        r"\s+",
                    ),
                )
                .withColumn(
                    "dst_words",
                    F.split(
                        F.coalesce(F.col("dst_label"), F.lit("")),
                        r"\s+",
                    ),
                )
                .withColumn(
                    "shared_words",
                    F.size(
                        F.array_intersect("src_words", "dst_words")
                    ),
                )
                .withColumn(
                    "total_words",
                    F.size(
                        F.array_union("src_words", "dst_words")
                    ),
                )
                .withColumn(
                    "label_sim",
                    F.when(
                        F.col("total_words") > 0,
                        F.col("shared_words").cast("double")
                        / F.col("total_words").cast("double"),
                    ).otherwise(F.lit(0.0)),
                )
            )

            # Jaccard similarity in the first label similarity slot
            sim_entry = label_edges.select(
                F.col("edge_idx"),
                F.lit(ls_start).alias("dim"),
                F.col("label_sim").alias("value"),
            )
            parts.append(sim_entry)

            # Hash the concatenated labels into remaining slots for
            # finer-grained label pair identity
            if ls_dim >= 2:
                shared_hash = label_edges.select(
                    F.col("edge_idx"),
                    (
                        F.abs(
                            F.hash(
                                F.concat(
                                    F.coalesce(
                                        F.col("src_label"), F.lit("")
                                    ),
                                    F.lit("::"),
                                    F.coalesce(
                                        F.col("dst_label"), F.lit("")
                                    ),
                                ),
                                F.lit(800),
                            )
                        )
                        % F.lit(max(1, ls_dim - 1))
                        + F.lit(ls_start + 1)
                    ).alias("dim"),
                    F.col("label_sim").alias("value"),
                )
                parts.append(shared_hash)
        else:
            # Non-correlation edges: encode a category indicator
            # so the GNN can still distinguish categories in this
            # sub-segment
            cat_label_entry = edge_df.select(
                F.col("edge_idx"),
                (
                    F.abs(F.hash(F.lit(category), F.lit(810)))
                    % F.lit(ls_dim)
                    + F.lit(ls_start)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(cat_label_entry)

        # --- Sub-segment 3c: Relation Identity ---
        ri_start = layout.seg3_relation_identity_start
        ri_dim = layout.seg3_relation_identity_dim

        # Hash the relation name into the relation identity sub-segment.
        # Two hash functions for reduced collision probability.
        for seed_offset in _EDGE_HASH_SEEDS[:2]:
            rel_hash = edge_df.select(
                F.col("edge_idx"),
                (
                    F.abs(
                        F.hash(F.lit(relation), F.lit(seed_offset + 900))
                    )
                    % F.lit(ri_dim)
                    + F.lit(ri_start)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(rel_hash)

        # Hash the category into a separate slot at half weight,
        # so edges of the same category share bits but edges of the
        # same specific relation share stronger bits
        cat_hash = edge_df.select(
            F.col("edge_idx"),
            (
                F.abs(F.hash(F.lit(category), F.lit(950)))
                % F.lit(ri_dim)
                + F.lit(ri_start)
            ).alias("dim"),
            F.lit(0.5).alias("value"),
        )
        parts.append(cat_hash)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("edge_idx").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Collection helpers (mirror node feature_extractor pattern)
    # ================================================================

    def _aggregate_entries(
        self,
        entries: DataFrame,
        key_cols: Tuple[str, ...] = (),
    ) -> DataFrame:
        """
        Sum sparse values that land on the same slot, on executors.

        Hash collisions from different sub-segments landing on the same
        dim are summed — same pattern as node feature vectors.

        key_cols carries the edge-type columns when several types share
        one aggregation; it is empty when aggregating a single type.
        """
        return (
            entries
            .groupBy(*key_cols, "edge_idx", "dim")
            .agg(F.sum("value").alias("value"))
            .select(
                *key_cols,
                F.col("edge_idx").cast("long"),
                F.col("dim").cast("int"),
                F.col("value").cast("float"),
            )
        )

    def _max_edges_per_collect(self, edge_vector_dim: int) -> int:
        """How many edges one collect may carry, sized in bytes not rows.

        The cap used to be a row count alone, but spark.driver.maxResultSize
        counts bytes. The same edge budget therefore meant 32x the bytes at
        edge_vector_dim 1024 that it meant at 32, so a run that fit at one
        width failed at another with nothing in the failure naming the width
        (#340).

        Spent in what the collect actually returns -- the sparse rows, at
        _SPARSE_ENTRY_BYTES each -- and not in what the dense tensor costs
        afterwards. Those differ by 5x, and only the first is what the cap
        measures.

        The row cap stays as a second, independent ceiling: it bounds the
        Pandas intermediaries, which the byte estimate does not describe.
        Spark reads maxResultSize 0 as unbounded and -1 as unset; either way
        there is no cap to size against and the row cap is the only one left.
        """
        cached = self._edges_per_collect
        if cached is not None and cached[0] == edge_vector_dim:
            return cached[1]

        rows = max(1, self._chunk_threshold)
        configured = self.spark.conf.get(
            "spark.driver.maxResultSize", None
        )
        cap = (
            None if configured is None
            else _parse_byte_string(configured)
        )
        if cap is None or cap <= 0:
            budget = rows
        else:
            per_edge = max(1, edge_vector_dim * _SPARSE_ENTRY_BYTES)
            by_bytes = int(cap * _RESULT_SIZE_TARGET_FRACTION) // per_edge
            # One edge must stay collectable whatever the cap says, or the
            # batch cannot progress at all.
            budget = max(1, min(by_bytes, rows))

        self._edges_per_collect = (edge_vector_dim, budget)
        return budget

    def _collect_small_batches(
        self,
        small: List[Tuple[Tuple[str, str, str], int, DataFrame]],
        edge_features: Dict[Tuple[str, str, str], torch.Tensor],
        edge_vector_dim: int,
    ) -> None:
        """
        Collect small edge types in bounded batches — one grouped
        toPandas per batch instead of one per edge type.

        Batches are capped on two independent axes: estimated bytes (so a
        batch's collect stays inside spark.driver.maxResultSize) and type
        count (so the batch's Catalyst plan stays bounded in width). Either
        cap alone is insufficient — see _MAX_EDGE_TYPES_PER_BATCH.

        The entries are aggregated and materialised ONCE before any of that
        batching happens. Each type's entries frame carries its own join
        against the endpoint property frames, which run to hundreds of
        millions of rows, so a batch that collects straight from those
        frames re-runs the join. #362 measured the cost: every batch read
        the same 23,184 MB and returned 0.53 MB, about 1.7 TB of repeated
        scanning to collect 41 MB. One eager checkpoint pays for the scan
        once; the batched collects then read only the aggregated entries,
        which are small by construction.
        """
        if not small:
            return

        materialised = self._materialise_small_entries(small)
        if materialised is None:
            return

        budget = self._max_edges_per_collect(edge_vector_dim)
        max_types = max(1, self._max_types_per_batch)

        batch: List[Tuple[int, Tuple[str, str, str], int]] = []
        batch_edges = 0
        num_batches = 0

        for type_id, (edge_type_key, num_edges, _) in enumerate(small):
            if batch and (
                batch_edges + num_edges > budget
                or len(batch) >= max_types
            ):
                self._flush_batch(
                    materialised, batch, edge_features, edge_vector_dim
                )
                num_batches += 1
                batch = []
                batch_edges = 0
            batch.append((type_id, edge_type_key, num_edges))
            batch_edges += num_edges

        if batch:
            self._flush_batch(
                materialised, batch, edge_features, edge_vector_dim
            )
            num_batches += 1

        materialised.unpersist(blocking=False)

        logger.info(
            f"    {len(small)} small edge types collected in "
            f"{num_batches} batch(es) "
            f"(budget {budget:,} edges, at most "
            f"{budget * edge_vector_dim * _SPARSE_ENTRY_BYTES / 1024 ** 2:,.0f}"
            f" MB collected per batch)"
        )

    def _materialise_small_entries(
        self,
        small: List[Tuple[Tuple[str, str, str], int, DataFrame]],
    ) -> Optional[DataFrame]:
        """
        Tag every small type's entries with its index, union them, and
        aggregate and materialise the result in one Spark job.

        This is the whole point of #362. The per-type entries frames are
        lazy joins against the endpoint property frames; collecting from
        them in N batches scans those frames N times. Aggregating first
        collapses the sparse entries to at most one row per (type, edge,
        dim), and the eager checkpoint pins that small result so the
        batched collects never touch the big frames again.

        The union is wide — one subtree per type — but it is planned and
        run exactly once here, rather than once per batch, which is the
        trade _MAX_EDGE_TYPES_PER_BATCH was guarding against on the
        collect path.
        """
        tagged: Optional[DataFrame] = None
        for type_id, (_, _, entries) in enumerate(small):
            part = entries.select(
                F.lit(type_id).cast("int").alias(self._TYPE_COL),
                F.col("edge_idx"),
                F.col("dim"),
                F.col("value"),
            )
            tagged = part if tagged is None else tagged.unionAll(part)

        if tagged is None:
            return None

        # This is the one expensive action on the small-type path, and it
        # used to run without saying anything. The 2026-09-01 run spent 83
        # minutes inside it, and the notebook's watchdog -- seeing no log
        # line move -- killed it as a hang while its tasks were still
        # finishing normally. Say when it starts and what it cost.
        logger.info(
            f"    Materialising entries for {len(small)} small edge "
            f"type(s) in one pass..."
        )
        started = time.monotonic()

        materialised = settle(
            self._aggregate_entries(tagged, (self._TYPE_COL,))
        )

        logger.info(
            f"    Materialised {materialised.count():,} aggregated "
            f"entries in {time.monotonic() - started:.1f}s"
        )

        return materialised

    def _flush_batch(
        self,
        materialised: DataFrame,
        batch: List[Tuple[int, Tuple[str, str, str], int]],
        edge_features: Dict[Tuple[str, str, str], torch.Tensor],
        edge_vector_dim: int,
    ) -> None:
        """
        Collect one batch of edge types from the materialised entries and
        scatter them into per-type dense tensors on the driver.

        edge_idx is per-type 0-based (assigned by a window partitioned
        on the type triple), so a type key has to travel with every row
        and the scatter has to be done per group — unlike the node side,
        where node_id alone identifies the row.

        The key is the type's index in the whole small list, as one int,
        not the three strings of the type triple. Those strings repeat on
        every sparse entry, and a batch holds up to _chunk_threshold edges
        times their non-zero dims — millions of string objects for the
        driver to build, where an int costs 4 bytes. Same reasoning as the
        edge-index collect in edge_mapper.py.

        The filter is what keeps the batch bounded. It reads the checkpoint
        written by _materialise_small_entries, so it costs a scan of the
        aggregated entries and never touches the endpoint frames (#362).
        """
        type_ids = [type_id for type_id, _, _ in batch]

        started = time.monotonic()
        pdf = materialised.filter(
            F.col(self._TYPE_COL).isin(type_ids)
        ).toPandas()
        logger.info(
            f"      batch of {len(batch)} type(s): {len(pdf):,} entries "
            f"in {time.monotonic() - started:.1f}s"
        )

        groups = (
            {int(k): g for k, g in pdf.groupby(self._TYPE_COL)}
            if not pdf.empty else {}
        )

        for type_id, edge_type_key, num_edges in batch:
            tensor = np.zeros(
                (num_edges, edge_vector_dim), dtype=np.float32
            )
            g = groups.get(type_id)
            if g is not None and not g.empty:
                edge_idxs = g["edge_idx"].values
                dims = g["dim"].values
                values = g["value"].values

                valid_mask = (dims >= 0) & (dims < edge_vector_dim)
                tensor[
                    edge_idxs[valid_mask], dims[valid_mask]
                ] = values[valid_mask]

            edge_features[edge_type_key] = (
                torch.from_numpy(tensor).contiguous()
            )

        # Reclaim the batch's Pandas intermediaries before the next
        # batch allocates its own. Driver-side only — all executor work
        # for this batch is complete.
        del pdf
        gc.collect()

    def _collect_and_scatter_chunked(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        num_edges: int,
        edge_vector_dim: int,
    ) -> None:
        """
        Collect sparse entries in chunks by edge_idx range and scatter
        into the pre-allocated dense array incrementally.

        Each chunk's Pandas DataFrame is scattered into the dense array
        and immediately freed. This bounds peak Pandas memory to
        ~chunk_size edges' sparse entries regardless of total edge count.
        """
        chunk_size = self._max_edges_per_collect(edge_vector_dim)

        combined = combined.cache()
        total_entries = combined.count()

        logger.info(
            f"      Chunked collection: {num_edges:,} edges, "
            f"{total_entries:,} sparse entries, "
            f"chunk size {chunk_size:,}"
        )

        num_chunks = (num_edges + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            lo = chunk_idx * chunk_size
            hi = min(lo + chunk_size, num_edges)

            chunk_df = combined.filter(
                (F.col("edge_idx") >= lo) & (F.col("edge_idx") < hi)
            )

            pdf = chunk_df.toPandas()

            if not pdf.empty:
                edge_idxs = pdf["edge_idx"].values
                dims = pdf["dim"].values
                values = pdf["value"].values

                valid_mask = (dims >= 0) & (dims < edge_vector_dim)
                tensor[
                    edge_idxs[valid_mask], dims[valid_mask]
                ] = values[valid_mask]

            del pdf
            gc.collect()

            if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                logger.info(
                    f"      Chunk {chunk_idx + 1}/{num_chunks} complete "
                    f"(edges {lo:,}–{hi:,})"
                )

        combined.unpersist()