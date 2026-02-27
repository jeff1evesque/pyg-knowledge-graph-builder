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
doubles resolution; passing edge_vector_dim=16 halves it.

All encoding runs on Spark executors using pure Spark column
expressions. Only compact [num_edges, edge_vector_dim] float32 arrays
are collected to the driver, one edge type at a time.

Architecture — no driver round-trip:
  Edge resolution (triples × node_id_df double-join) is reconstructed
  on executors from the same inputs EdgeMapper uses. The resolved edge
  DataFrame stays on executors throughout feature computation. Only
  the final dense float32 tensor per edge type crosses to the driver.
  The edge_index tensor (already on the driver from EdgeMapper) is used
  solely for ordering alignment — a monotonic edge_idx column assigned
  via Window functions on executors ensures the feature tensor rows
  match the edge_index columns.

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
"""
import logging
import gc
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from glue_jobs.utils.rdf_utils import NAMESPACE_PREFIXES

logger = logging.getLogger(__name__)

# ============================================
# Default edge vector dimension
# ============================================
EDGE_VECTOR_DIM = 32

# ============================================
# Segment proportions (fraction of total edge_vector_dim)
# ============================================
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

_EXCLUDED_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#imports",
}

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
_TEMPORAL_RELATION_FRAGMENTS = (
    "precedes", "follows", "hasNext", "hasPrevious",
    "temporallyRelated",
)

_OPTION_STOCK_RELATION_FRAGMENTS = (
    "hasUnderlyingPriceObservation", "hasUnderlying",
)

_ESCALATION_RELATION_FRAGMENTS = (
    "escalatesTo", "escalatesFrom", "severityChange",
)

_CORRELATION_RELATION_FRAGMENTS = (
    "correlatesWith", "relatedTo",
)

_CAUSAL_RELATION_FRAGMENTS = (
    "leadsTo", "impacts", "causes", "affects",
)

_STRATEGY_RELATION_FRAGMENTS = (
    "straddleWith", "spreadWith", "strangleWith",
)

_SKIP_RELATION_FRAGMENTS = (
    "belongsToSector", "sameAs", "hasParent", "hasChild",
    "equivalentClass", "equivalentProperty", "imports",
    "refersToCompany", "hasRegion", "affectsRegion",
    "sameEventType", "affectsSameRegion",
)

# Hash seeds for deterministic encoding
_EDGE_HASH_SEEDS = [0, 7, 13, 31]

# Driver memory safety constants
_CHUNK_EDGE_THRESHOLD = 1_000_000


class EdgeVectorLayout:
    """
    Computes all segment and sub-segment boundaries for edge feature
    vectors from a given edge_vector_dim. All boundaries are integer
    dim indices.

    Mirrors VectorLayout for node features — same guarantees:
    - All sub-segments are contiguous and non-overlapping
    - All sub-segments have at least 1 dimension (raises if
      edge_vector_dim is too small)
    - Sub-segment dims sum exactly to edge_vector_dim

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
        seg3_total = edge_vector_dim - seg1_total - seg2_total

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
        dir_dim = seg1_total - td_dim - pf_dim

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
        mag_dim = seg2_total - diff_dim - rat_dim

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
        ri_dim = seg3_total - ns_dim - ls_dim

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


def _build_predicate_to_relation_expr(
    pred_col: str = "predicate",
) -> F.Column:
    """
    Build a pure-Spark Column expression that converts a predicate URI
    to a PyG-compatible relation name. No Python UDF.

    Identical to edge_mapper._build_predicate_to_relation_expr — we
    replicate it here to avoid a circular import and because both
    modules need the same conversion independently.
    """
    col = F.col(pred_col)
    expr = None

    for namespace, prefix in NAMESPACE_PREFIXES:
        ns_len = len(namespace)
        local_name = F.substring(col, ns_len + 1, 1000)
        local_name = F.regexp_replace(local_name, r"^[/#]+|[/#]+$", "")
        relation_name = F.concat(F.lit(f"{prefix}_"), local_name)

        condition = col.startswith(namespace) & (F.length(local_name) > 0)

        if expr is None:
            expr = F.when(condition, relation_name)
        else:
            expr = expr.when(condition, relation_name)

    fallback_local = F.regexp_extract(col, r"[#/]([^#/]+)$", 1)
    fallback_name = F.concat(F.lit("unknown_"), fallback_local)

    expr = expr.otherwise(
        F.when(
            F.length(fallback_local) > 0, fallback_name
        ).otherwise(
            F.concat(
                F.lit("unknown_"), F.abs(F.hash(col)).cast("string")
            )
        )
    )

    return expr


class EdgeFeatureExtractor:
    """
    Builds fixed-width edge feature vectors derived from endpoint node
    properties.

    All heavy computation runs on Spark executors. Only compact float
    arrays are collected to the driver, one edge type at a time.

    Architecture — no driver round-trip:
      The resolved edge DataFrame is reconstructed on executors by
      replaying the same double-join that EdgeMapper performs (triples
      × node_id_df). This avoids pushing driver-side tensors back to
      Spark. A deterministic edge_idx is assigned via Window functions
      on executors using the same sort order (src_id, dst_id) that
      EdgeMapper used for deduplication, ensuring the feature tensor
      rows align with the edge_index tensor columns.

    Edge features are selective — only edge types with meaningful
    per-instance variation get feature vectors. The set of featurized
    edge types is configurable.

    Driver memory safety:
      - Edge feature tensors are much smaller than node features
        (32-d vs 1024-d, and edges are just pairs)
      - Collection is per-edge-type with immediate Pandas release
      - For very large edge types, chunked collection is available
    """

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

        # Which categories of edge types to featurize
        self._enabled_categories: Set[str] = set(
            edge_feat_config.get(
                "enabled_categories",
                ["temporal", "option_stock", "escalation"],
            )
        )

        self._layout = EdgeVectorLayout(self._edge_vector_dim)

    def build_edge_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build edge feature vectors for eligible edge types.

        Reconstructs the resolved edge DataFrame on executors (same
        double-join as EdgeMapper), then derives per-edge features
        from endpoint literal properties. Only compact float tensors
        cross to the driver.

        Args:
            triples_df: Enriched triples DataFrame
                (subject, predicate, object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
            edge_indices: Dict from EdgeMapper — maps
                (src_type, relation, dst_type) -> LongTensor[2, N].
                Used only for edge counts and ordering validation,
                not pushed back to Spark.

        Returns:
            Dict mapping (src_type, relation, dst_type) ->
                FloatTensor[num_edges, edge_vector_dim]
            Only contains entries for edge types that received features.
        """
        if not self._enabled:
            logger.info("  Edge features disabled by config")
            return {}

        layout = self._layout
        edge_vector_dim = layout.edge_vector_dim

        logger.info(
            f"  Building {edge_vector_dim}-d edge feature vectors"
        )
        logger.info(f"  {layout.summary()}")
        logger.info(
            f"  Enabled categories: {sorted(self._enabled_categories)}"
        )

        # Classify each edge type — determine which need features
        eligible_types: List[Tuple[Tuple[str, str, str], str]] = []
        skipped = 0

        for edge_type_key in edge_indices:
            src_type, relation, dst_type = edge_type_key
            category = _classify_relation(relation)

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
            return {}

        # ============================================
        # Reconstruct resolved edges on executors
        # (same double-join as EdgeMapper, stays on executors)
        # ============================================
        logger.info("  Reconstructing resolved edges on executors...")

        resolved_edges_df = self._reconstruct_resolved_edges(
            triples_df, node_id_df
        )

        # ============================================
        # Pre-extract endpoint literal properties (on executors)
        # ============================================
        logger.info("  Extracting endpoint literal properties...")

        numeric_props_df = self._extract_node_numeric_properties(
            triples_df, node_id_df
        )
        label_df = self._extract_node_labels(triples_df, node_id_df)

        # ============================================
        # Build features per eligible edge type
        # ============================================
        edge_features: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for edge_type_key, category in eligible_types:
            src_type, relation, dst_type = edge_type_key
            num_edges = edge_indices[edge_type_key].shape[1]

            logger.info(
                f"    [{edge_type_key}] {num_edges:,} edges, "
                f"category={category}"
            )

            tensor = self._build_edge_type_vectors(
                edge_type_key=edge_type_key,
                num_edges=num_edges,
                category=category,
                resolved_edges_df=resolved_edges_df,
                numeric_props_df=numeric_props_df,
                label_df=label_df,
            )

            if tensor is not None:
                edge_features[edge_type_key] = tensor
                feat_mb = tensor.numel() * 4 / (1024 * 1024)
                logger.info(
                    f"      → shape {list(tensor.shape)}, "
                    f"{feat_mb:.1f} MB"
                )

            gc.collect()

        # Cleanup
        for df in [resolved_edges_df, numeric_props_df, label_df]:
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

        return edge_features

    # ================================================================
    # Resolved edge reconstruction (entirely on executors)
    # ================================================================

    def _reconstruct_resolved_edges(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Reconstruct the resolved edge DataFrame on executors by
        replaying the same double-join that EdgeMapper performs.

        Returns a cached DataFrame with columns:
            (src_type, src_id, relation, dst_type, dst_id, edge_idx)

        The edge_idx column is a deterministic 0-indexed integer per
        (src_type, relation, dst_type) group, assigned via Window
        functions using the same sort order (src_id, dst_id) that
        EdgeMapper used. This ensures feature tensor rows align with
        edge_index tensor columns.

        All work runs on executors — no data crosses to the driver.
        """
        excluded_list = list(_EXCLUDED_PREDICATES)
        edge_triples = triples_df.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        src_lookup = node_id_df.select(
            F.col("uri").alias("_src_uri"),
            F.col("node_id").alias("src_id"),
            F.col("node_type").alias("src_type"),
        )

        dst_lookup = node_id_df.select(
            F.col("uri").alias("_dst_uri"),
            F.col("node_id").alias("dst_id"),
            F.col("node_type").alias("dst_type"),
        )

        edges_resolved = (
            edge_triples
            .join(
                src_lookup,
                edge_triples["subject"] == src_lookup["_src_uri"],
                "inner",
            )
            .drop("_src_uri")
            .join(
                dst_lookup,
                F.col("object") == dst_lookup["_dst_uri"],
                "inner",
            )
            .drop("_dst_uri")
        )

        # Derive relation names (pure Spark, no UDF)
        edges_resolved = edges_resolved.withColumn(
            "relation", _build_predicate_to_relation_expr("predicate")
        )

        # Select, deduplicate, and assign deterministic edge_idx
        edges_deduped = (
            edges_resolved
            .select(
                "src_type",
                F.col("src_id").cast("long"),
                "relation",
                "dst_type",
                F.col("dst_id").cast("long"),
            )
            .dropDuplicates(
                ["src_type", "src_id", "relation", "dst_type", "dst_id"]
            )
        )

        # Assign per-edge-type 0-indexed edge_idx using the same
        # deterministic ordering (src_id, dst_id) that EdgeMapper
        # collects in. This ensures row i of the feature tensor
        # corresponds to column i of the edge_index tensor.
        edges_with_idx = edges_deduped.withColumn(
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

        edges_with_idx = edges_with_idx.cache()
        total = edges_with_idx.count()
        logger.info(
            f"    Reconstructed {total:,} resolved edges on executors"
        )

        return edges_with_idx

    # ================================================================
    # Endpoint property extraction (on executors)
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
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        # Isolate literal triples via anti-join against node URIs
        literal_triples = triples_df.join(
            node_id_df.select(F.col("uri").alias("_obj_uri")),
            triples_df["object"] == F.col("_obj_uri"),
            "left_anti",
        ).filter(~F.col("predicate").isin(excluded_list))

        # Parse numeric values
        candidates = literal_triples.withColumn(
            "numeric_value",
            F.split(F.col("object"), r"\^\^").getItem(0).cast("double"),
        ).filter(F.col("numeric_value").isNotNull())

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

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

        numeric_df = numeric_df.cache()
        count = numeric_df.count()
        logger.info(
            f"    Endpoint numeric properties: {count:,} "
            f"(node, property) pairs"
        )

        return numeric_df

    def _extract_node_labels(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Extract rdfs:label values for all nodes, keyed by
        (node_type, node_id, label).

        Used for label similarity signals on correlation edges.
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

        label_df = label_df.cache()
        count = label_df.count()
        logger.info(f"    Endpoint labels: {count:,} (node, label) pairs")

        return label_df

    # ================================================================
    # Per-edge-type vector assembly
    # ================================================================

    def _build_edge_type_vectors(
        self,
        edge_type_key: Tuple[str, str, str],
        num_edges: int,
        category: str,
        resolved_edges_df: DataFrame,
        numeric_props_df: DataFrame,
        label_df: DataFrame,
    ) -> Optional[torch.Tensor]:
        """
        Build feature vectors for all edges of one type.

        Filters the resolved_edges_df to this edge type (on executors),
        joins endpoint properties, computes derived features as sparse
        (edge_idx, dim, value) entries, aggregates on executors, then
        collects to driver.

        No data round-trips through the driver — resolved_edges_df
        was reconstructed entirely on executors.
        """
        layout = self._layout
        edge_vector_dim = layout.edge_vector_dim
        src_type, relation, dst_type = edge_type_key

        if num_edges == 0:
            return torch.zeros(
                0, edge_vector_dim, dtype=torch.float32
            )

        # ============================================
        # Filter resolved edges to this type (on executors)
        # ============================================
        edge_df = (
            resolved_edges_df
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
        seg1_entries = self._encode_temporal_signals(
            edge_df=edge_df,
            src_type=src_type,
            dst_type=dst_type,
            category=category,
            numeric_props_df=numeric_props_df,
        )

        seg2_entries = self._encode_numeric_contrast(
            edge_df=edge_df,
            src_type=src_type,
            dst_type=dst_type,
            category=category,
            numeric_props_df=numeric_props_df,
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
            return torch.zeros(
                num_edges, edge_vector_dim, dtype=torch.float32
            )

        combined = all_parts[0]
        for df in all_parts[1:]:
            combined = combined.unionAll(df)

        # Aggregate on executors: sum values at same (edge_idx, dim)
        combined = (
            combined
            .groupBy("edge_idx", "dim")
            .agg(F.sum("value").alias("value"))
            .select(
                F.col("edge_idx").cast("long"),
                F.col("dim").cast("int"),
                F.col("value").cast("float"),
            )
        )

        # ============================================
        # Collect to driver and scatter into dense array
        # ============================================
        tensor = np.zeros(
            (num_edges, edge_vector_dim), dtype=np.float32
        )

        if num_edges <= self._chunk_threshold:
            self._collect_and_scatter(
                combined, tensor, edge_vector_dim
            )
        else:
            self._collect_and_scatter_chunked(
                combined, tensor, num_edges, edge_vector_dim
            )

        return torch.from_numpy(tensor).contiguous()

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
    ) -> Optional[DataFrame]:
        """
        Encode Segment 1: time delta, period flags, and temporal
        direction between edge endpoints.

        For temporal edges: computes month/year delta from endpoint
        properties.
        For non-temporal edges: encodes a hash of the relation category
        into the time delta sub-segment as a type indicator.

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Time Delta ---
        td_start = layout.seg1_time_delta_start
        td_dim = layout.seg1_time_delta_dim

        if category == "temporal":
            # Join source endpoint month/year properties
            month_preds = numeric_props_df.filter(
                F.col("predicate").rlike(
                    "(?i)(hasMonth|month|hasMonthNum|monthNumber)"
                )
            )
            year_preds = numeric_props_df.filter(
                F.col("predicate").rlike(
                    "(?i)(hasYear|year|hasYearNum|yearNumber)"
                )
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

            # Join temporal properties to edges (on executors)
            temporal_edges = (
                edge_df
                .join(
                    src_month,
                    edge_df["src_id"] == src_month["_src_nid"],
                    "left",
                )
                .drop("_src_nid")
                .join(
                    src_year,
                    edge_df["src_id"] == src_year["_src_nid"],
                    "left",
                )
                .drop("_src_nid")
                .join(
                    dst_month,
                    edge_df["dst_id"] == dst_month["_dst_nid"],
                    "left",
                )
                .drop("_dst_nid")
                .join(
                    dst_year,
                    edge_df["dst_id"] == dst_year["_dst_nid"],
                    "left",
                )
                .drop("_dst_nid")
            )

            # Compute month delta: (dst_year - src_year) * 12
            #                       + (dst_month - src_month)
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
            # ~[-1, 1] range for typical monthly data
            temporal_edges = temporal_edges.withColumn(
                "norm_delta",
                F.col("month_delta") / F.lit(12.0),
            )

            # Hash the normalized delta into time_delta sub-segment
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

            # Also encode absolute delta in a second slot
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

            # Same-year flag
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

            # Consecutive-month flag
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

            # Same-quarter flag
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

            # Forward direction indicator (+1 if dst is later, -1 if
            # earlier, 0 if same or unknown)
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
            # in the time delta sub-segment
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
    ) -> Optional[DataFrame]:
        """
        Encode Segment 2: numeric differences, ratios, and magnitudes
        between edge endpoints.

        For each numeric property shared by both endpoints, computes:
        - Difference (dst - src) hashed into difference sub-segment
        - Ratio (dst / src, clamped) hashed into ratio sub-segment
        - Average magnitude hashed into magnitude sub-segment

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

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

        # Join edge endpoints with their numeric properties
        # Only keep properties that both endpoints share (same predicate)
        edge_with_src = (
            edge_df
            .join(
                src_numerics,
                edge_df["src_id"] == src_numerics["_src_nid"],
                "inner",
            )
            .drop("_src_nid")
        )

        edge_with_both = (
            edge_with_src
            .join(
                dst_numerics,
                (edge_with_src["dst_id"] == dst_numerics["_dst_nid"])
                & (edge_with_src["src_pred"] == dst_numerics["dst_pred"]),
                "inner",
            )
            .drop("_dst_nid", "dst_pred")
            .withColumnRenamed("src_pred", "predicate")
        )

        if not edge_with_both.head(1):
            # No shared numeric properties — try cross-property
            # derivation for specific categories
            return self._encode_cross_property_contrast(
                edge_df, src_type, dst_type, category, numeric_props_df
            )

        # --- Sub-segment 2a: Difference ---
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

        # --- Sub-segment 2b: Ratio ---
        rat_start = layout.seg2_ratio_start
        rat_dim = layout.seg2_ratio_dim

        # Safe ratio: clamp to [-10, 10] to avoid extreme values
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

        # --- Sub-segment 2c: Magnitude ---
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

        For option_stock edges: moneyness = strike_price / observed_price
        For escalation edges: severity delta from severity level properties
        """
        layout = self._layout
        parts: List[DataFrame] = []

        if category == "option_stock":
            # Source (option): look for strike price
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

            # Destination (stock): look for observed price
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
                # Moneyness = strike / stock_price
                diff_start = layout.seg2_difference_start
                diff_dim = layout.seg2_difference_dim

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

                # Log-moneyness in ratio sub-segment
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

                # Strike-stock difference in magnitude sub-segment
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
            # endpoints
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

        Returns DataFrame(edge_idx: long, dim: int, value: float)
        or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Namespace Signals ---
        ns_start = layout.seg3_namespace_start
        ns_dim = layout.seg3_namespace_dim

        # Same-namespace flag: are src and dst from the same ontology?
        # Derive from node_type prefix (e.g., "cpi_Index" → "cpi")
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
        # namespace slots
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
            # Join labels for both endpoints
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

            # Compute word overlap as a simple similarity measure:
            # Split labels into words, count shared words / total words
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

            sim_entry = label_edges.select(
                F.col("edge_idx"),
                F.lit(ls_start).alias("dim"),
                F.col("label_sim").alias("value"),
            )
            parts.append(sim_entry)

            # Hash shared words into remaining label similarity slots
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

        # Hash the relation name into the relation identity sub-segment
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

        # Hash the category into a separate slot
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

    def _collect_and_scatter(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        edge_vector_dim: int,
    ) -> None:
        """
        Collect all sparse entries and scatter into dense array.
        Used for small-to-medium edge types.
        """
        pdf = combined.toPandas()

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
        """
        chunk_size = self._chunk_threshold

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