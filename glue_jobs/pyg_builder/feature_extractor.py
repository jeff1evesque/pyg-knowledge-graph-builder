"""
Feature Extractor — Ontology-Aware Fixed-Width Node Feature Vectors

Constructs universal fixed-width feature vectors (default 1024-d) that
encode three layers of information for every node:

  Segment 1 — Ontology Structure  (25% of dims):  class identity,
              class hierarchy (rdfs:subClassOf chains), ontology/source
              membership

  Segment 2 — Property Schema     (37.5% of dims): property presence
              (which ontology-defined properties this node has),
              domain/range signals, property hierarchy

  Segment 3 — Literal Values      (37.5% of dims): numeric values in
              hashed slots (z-score normalized), categorical values as
              multi-hot hash encodings

All segment and sub-segment boundaries scale proportionally with
vector_dim. Passing vector_dim=512 produces a 512-d vector with the
same three-segment structure at half resolution. Passing vector_dim=2048
doubles resolution. The default 1024 is recommended for production.

All encoding runs on Spark executors using deterministic hash-based
functions expressed as pure Spark column expressions. Only the final
[num_nodes, vector_dim] float32 array per node type is collected to
the driver.

Driver memory safety:
  The dense tensor for a node type is num_nodes × vector_dim × 4 bytes.
  For large types (>500K nodes), this can exceed available driver memory.
  To prevent OOM:
  - Sparse (node_id, dim, value) entries are collected via toPandas()
    and scattered directly into a pre-allocated numpy array
  - The Pandas DataFrame is deleted immediately after scatter
  - For very large types, collection is chunked by node_id range
    so that at most ~500K nodes' sparse entries are in Pandas at once
  - The dense tensor itself is unavoidable (PyG requires it), but we
    ensure only ONE type's tensor + its Pandas intermediary coexist

Why this replaces the old per-type variable-width approach:
  - Universal width enables shared GNN layers across all node types
  - Ontology structure gives the GNN a type fingerprint beyond raw literals
  - Property presence distinguishes "missing" from "inapplicable"
  - Hash-based encoding avoids vocabulary management across 100+ ontologies
"""
import logging
import gc
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from glue_jobs.utils.rdf_utils import ONTOLOGY_NAMESPACE_INDICES

logger = logging.getLogger(__name__)

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
_SEG1_FRAC = 0.25
_SEG2_FRAC = 0.375
_SEG3_FRAC = 0.375  # = 1.0 - 0.25 - 0.375

# Sub-segment fractions within each segment
_SEG1_CLASS_IDENTITY_FRAC = 0.25
_SEG1_CLASS_HIERARCHY_FRAC = 0.50
_SEG1_ONTOLOGY_SOURCE_FRAC = 0.25

_SEG2_PROPERTY_PRESENCE_FRAC = 0.50
_SEG2_DOMAIN_RANGE_FRAC = 0.29
_SEG2_PROPERTY_HIERARCHY_FRAC = 0.21

_SEG3_NUMERIC_FRAC = 0.67
_SEG3_CATEGORICAL_FRAC = 0.33

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUB_PROPERTY_OF = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"

_NON_FEATURE_PREDICATES = {
    RDF_TYPE,
    RDFS_SUBCLASS_OF,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUB_PROPERTY_OF,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2002/07/owl#imports",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
}

# ============================================
# Driver memory safety constants
# ============================================
_CHUNK_NODE_THRESHOLD = 500_000

# Number of hash functions for multi-hot categorical encoding
_NUM_CATEGORICAL_HASHES = 4

# Seed offsets for independent hash functions
_HASH_SEEDS = [0, 7, 13, 31]


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
        layout.seg1_class_identity_dim    # 64
        layout.seg3_categorical_start     # 896
        layout.seg3_categorical_dim       # 128
        layout.vector_dim                 # 1024
    """

    def __init__(self, vector_dim: int):
        if vector_dim < 32:
            raise ValueError(
                f"vector_dim must be >= 32, got {vector_dim}. "
                f"Minimum needed for all sub-segments to have >= 1 dim."
            )

        self.vector_dim = vector_dim

        # --- Segment boundaries ---
        seg1_total = max(1, int(round(vector_dim * _SEG1_FRAC)))
        seg2_total = max(1, int(round(vector_dim * _SEG2_FRAC)))
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


class FeatureExtractor:
    """
    Builds universal fixed-width ontology-aware feature vectors for all
    nodes.

    All heavy computation runs on Spark executors. Only compact float
    arrays are collected to the driver, one node type at a time.

    Driver memory safety:
      - Dense tensor is pre-allocated once per type (num_nodes × vector_dim × 4B)
      - Sparse entries collected in chunks for large types
      - Pandas intermediaries freed immediately after scatter
      - Only one type's tensor is being built at a time
      - gc.collect() between types to reclaim fragmented memory
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        feat_config = config.get("feature_config", {})
        self._normalize = feat_config.get("normalize", True)
        self._vector_dim = feat_config.get("vector_dim", VECTOR_DIM)
        self._chunk_threshold = feat_config.get(
            "chunk_node_threshold", _CHUNK_NODE_THRESHOLD
        )

        # Compute layout from vector_dim — all segment boundaries
        # scale proportionally
        self._layout = VectorLayout(self._vector_dim)

    def build_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """
        Build universal fixed-width feature vectors for all node types.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate, object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
            node_counts: Dict[str, int] of node type counts

        Returns:
            Tuple of:
            - Dict[node_type -> FloatTensor[num_nodes, vector_dim]]
            - Dict[node_type -> List[str]] segment description lists
        """
        layout = self._layout
        vector_dim = layout.vector_dim

        logger.info(
            f"  Building {vector_dim}-d ontology-aware feature vectors"
        )
        logger.info(f"  {layout.summary()}")

        # Log driver memory budget estimate
        total_dense_bytes = sum(
            n * vector_dim * 4 for n in node_counts.values()
        )
        total_dense_mb = total_dense_bytes / (1024 * 1024)
        logger.info(
            f"  Estimated total dense tensor memory: "
            f"{total_dense_mb:,.1f} MB across {len(node_counts)} types"
        )

        # ============================================
        # Pre-compute ontology structure tables (on executors)
        # ============================================
        logger.info("  Extracting ontology structure from triples...")

        class_hierarchy_df = self._extract_class_hierarchy(triples_df)
        property_schema_df = self._extract_property_schema(triples_df)
        property_hierarchy_df = self._extract_property_hierarchy(triples_df)

        # ============================================
        # Pre-compute per-node property presence (on executors)
        # ============================================
        logger.info("  Computing per-node property presence...")

        node_properties_df = self._compute_node_properties(
            triples_df, node_id_df
        )

        # ============================================
        # Pre-compute literal values (on executors)
        # ============================================
        logger.info("  Extracting literal values...")

        numeric_df = self._extract_numeric_literals(triples_df, node_id_df)
        categorical_df = self._extract_categorical_literals(
            triples_df, node_id_df
        )

        # ============================================
        # Pre-compute normalization stats for numeric values
        # ============================================
        norm_stats = None
        if self._normalize and numeric_df is not None:
            logger.info("  Computing normalization statistics...")
            norm_stats = self._compute_normalization_stats(numeric_df)

        # ============================================
        # Get type URIs for ontology encoding
        # ============================================
        type_uri_df = self._get_type_uri_mapping(triples_df, node_id_df)

        # ============================================
        # Build vectors per node type (on executors, collect per type)
        # Process largest types first so OOM fails fast if it will fail
        # ============================================
        logger.info("  Assembling feature vectors per node type...")

        feature_tensors: Dict[str, torch.Tensor] = {}
        feature_names: Dict[str, List[str]] = {}

        sorted_types = sorted(
            node_counts.items(), key=lambda x: -x[1]
        )

        for node_type, num_nodes in sorted_types:
            if num_nodes == 0:
                continue

            dense_mb = num_nodes * vector_dim * 4 / (1024 * 1024)
            needs_chunking = num_nodes > self._chunk_threshold
            logger.info(
                f"    [{node_type}] {num_nodes:,} nodes, "
                f"dense tensor: {dense_mb:,.1f} MB"
                f"{' (chunked collection)' if needs_chunking else ''}"
            )

            tensor = self._build_type_vectors(
                node_type=node_type,
                num_nodes=num_nodes,
                node_id_df=node_id_df,
                type_uri_df=type_uri_df,
                class_hierarchy_df=class_hierarchy_df,
                property_schema_df=property_schema_df,
                property_hierarchy_df=property_hierarchy_df,
                node_properties_df=node_properties_df,
                numeric_df=numeric_df,
                categorical_df=categorical_df,
                norm_stats=norm_stats,
            )

            feature_tensors[node_type] = tensor
            feature_names[node_type] = [
                f"ontology_structure"
                f"[{layout.seg1_start}:{layout.seg2_start}]",
                f"property_schema"
                f"[{layout.seg2_start}:{layout.seg3_start}]",
                f"literal_values"
                f"[{layout.seg3_start}:{layout.vector_dim}]",
            ]

            # Force GC between types to reclaim Pandas/numpy intermediaries
            gc.collect()

        # Cleanup cached intermediates
        for df in [numeric_df, categorical_df, class_hierarchy_df,
                    property_schema_df]:
            if df is not None:
                try:
                    df.unpersist()
                except Exception:
                    pass

        return feature_tensors, feature_names

    # ================================================================
    # Ontology structure extraction (all on executors)
    # ================================================================

    def _extract_class_hierarchy(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:subClassOf chains from triples.

        Returns DataFrame(class_uri, superclass_uri, depth) where depth
        indicates distance in the hierarchy (1 = direct superclass).

        Computes transitive closure up to depth 10 via iterative joins
        on executors.
        """
        direct = (
            triples_df
            .filter(F.col("predicate") == RDFS_SUBCLASS_OF)
            .select(
                F.col("subject").alias("class_uri"),
                F.col("object").alias("superclass_uri"),
            )
            .distinct()
        )

        if not direct.head(1):
            logger.info("    No rdfs:subClassOf triples found")
            schema = "class_uri string, superclass_uri string, depth int"
            return self.spark.createDataFrame([], schema)

        current = direct.withColumn("depth", F.lit(1))
        all_hierarchy = current

        max_depth = 10
        for d in range(2, max_depth + 1):
            next_level = (
                current
                .select(
                    F.col("class_uri"),
                    F.col("superclass_uri").alias("_mid"),
                )
                .join(
                    direct.select(
                        F.col("class_uri").alias("_mid"),
                        F.col("superclass_uri"),
                    ),
                    "_mid",
                    "inner",
                )
                .drop("_mid")
                .withColumn("depth", F.lit(d))
                .distinct()
            )

            next_level = next_level.join(
                all_hierarchy.select("class_uri", "superclass_uri"),
                ["class_uri", "superclass_uri"],
                "left_anti",
            )

            if not next_level.head(1):
                break

            all_hierarchy = all_hierarchy.unionAll(next_level)
            current = next_level

        all_hierarchy = all_hierarchy.cache()
        count = all_hierarchy.count()
        logger.info(
            f"    Class hierarchy: {count:,} (class, superclass) pairs"
        )

        return all_hierarchy

    def _extract_property_schema(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:domain and rdfs:range declarations.

        Returns DataFrame(property_uri, domain_uri, range_uri).
        """
        domain_df = (
            triples_df
            .filter(F.col("predicate") == RDFS_DOMAIN)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("domain_uri"),
            )
            .distinct()
        )

        range_df = (
            triples_df
            .filter(F.col("predicate") == RDFS_RANGE)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("range_uri"),
            )
            .distinct()
        )

        schema_df = domain_df.join(range_df, "property_uri", "full_outer")
        schema_df = schema_df.cache()

        count = schema_df.count()
        logger.info(
            f"    Property schema: {count:,} properties with domain/range"
        )

        return schema_df

    def _extract_property_hierarchy(
        self, triples_df: DataFrame
    ) -> DataFrame:
        """
        Extract rdfs:subPropertyOf relationships.

        Returns DataFrame(property_uri, super_property_uri).
        """
        prop_hier = (
            triples_df
            .filter(F.col("predicate") == RDFS_SUB_PROPERTY_OF)
            .select(
                F.col("subject").alias("property_uri"),
                F.col("object").alias("super_property_uri"),
            )
            .distinct()
        )

        return prop_hier

    # ================================================================
    # Per-node property presence (on executors)
    # ================================================================

    def _compute_node_properties(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Compute which properties each node has (regardless of value).

        Returns DataFrame(node_type, node_id, predicate) — one row per
        (node, property) pair where the node is the subject.
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        node_props = (
            triples_df
            .filter(~F.col("predicate").isin(excluded_list))
            .select("subject", "predicate")
            .distinct()
            .join(
                node_lookup,
                F.col("subject") == F.col("_node_uri"),
                "inner",
            )
            .drop("_node_uri", "subject")
            .select("node_type", "node_id", "predicate")
        )

        return node_props

    # ================================================================
    # Literal value extraction (on executors)
    # ================================================================

    def _extract_numeric_literals(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Extract numeric literal properties joined with node IDs.

        Returns DataFrame(node_type, node_id, predicate, numeric_value)
        or None if no numeric literals found.
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        literal_triples = triples_df.join(
            node_id_df.select(F.col("uri").alias("_obj_uri")),
            triples_df["object"] == F.col("_obj_uri"),
            "left_anti",
        )

        literal_triples = literal_triples.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        candidates = literal_triples.withColumn(
            "numeric_value",
            F.split(F.col("object"), r"\^\^").getItem(0).cast("double"),
        ).filter(F.col("numeric_value").isNotNull())

        if not candidates.head(1):
            logger.info("    No numeric literals found")
            return None

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
            f"    Numeric literals: {count:,} (node, property) pairs"
        )

        return numeric_df

    def _extract_categorical_literals(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Extract categorical (non-numeric) literal properties.

        Returns DataFrame(node_type, node_id, predicate, cat_value)
        or None.
        """
        excluded_list = list(_NON_FEATURE_PREDICATES)

        literal_triples = triples_df.join(
            node_id_df.select(F.col("uri").alias("_obj_uri")),
            triples_df["object"] == F.col("_obj_uri"),
            "left_anti",
        )

        literal_triples = literal_triples.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        non_numeric = literal_triples.withColumn(
            "_try_numeric",
            F.split(F.col("object"), r"\^\^").getItem(0).cast("double"),
        ).filter(
            F.col("_try_numeric").isNull()
        ).drop("_try_numeric")

        if not non_numeric.head(1):
            logger.info("    No categorical literals found")
            return None

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        cat_df = (
            non_numeric
            .join(
                node_lookup,
                non_numeric["subject"] == node_lookup["_node_uri"],
                "inner",
            )
            .drop("_node_uri")
            .select(
                "node_type", "node_id", "predicate",
                F.col("object").alias("cat_value"),
            )
        )

        return cat_df

    # ================================================================
    # Normalization statistics (on executors)
    # ================================================================

    def _compute_normalization_stats(
        self, numeric_df: DataFrame
    ) -> DataFrame:
        """
        Compute per-predicate mean and stddev for z-score normalization.

        Returns DataFrame(predicate, mu, sigma) — small table, broadcast
        joined downstream.
        """
        stats = (
            numeric_df
            .groupBy("predicate")
            .agg(
                F.mean("numeric_value").alias("mu"),
                F.stddev("numeric_value").alias("sigma"),
            )
            .withColumn(
                "sigma",
                F.when(
                    (F.col("sigma").isNull()) | (F.col("sigma") == 0.0),
                    F.lit(1.0),
                ).otherwise(F.col("sigma")),
            )
            .withColumn(
                "mu",
                F.coalesce(F.col("mu"), F.lit(0.0)),
            )
        )

        return stats

    # ================================================================
    # Type URI mapping
    # ================================================================

    def _get_type_uri_mapping(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> DataFrame:
        """
        Get all rdf:type URIs for each node.

        Returns DataFrame(node_type, node_id, type_uri) with all
        rdf:type URIs per node.
        """
        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        type_uri_df = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(
                F.col("subject").alias("_node_uri"),
                F.col("object").alias("type_uri"),
            )
            .join(node_lookup, "_node_uri", "inner")
            .drop("_node_uri")
            .select("node_type", "node_id", "type_uri")
            .distinct()
        )

        return type_uri_df

    # ================================================================
    # Per-type vector assembly (on executors, collect to driver)
    # ================================================================

    def _build_type_vectors(
        self,
        node_type: str,
        num_nodes: int,
        node_id_df: DataFrame,
        type_uri_df: DataFrame,
        class_hierarchy_df: DataFrame,
        property_schema_df: DataFrame,
        property_hierarchy_df: DataFrame,
        node_properties_df: DataFrame,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        norm_stats: Optional[DataFrame],
    ) -> torch.Tensor:
        """
        Build the full vector for all nodes of one type.

        Each segment is computed on executors as a DataFrame of
        (node_id, dim, value) sparse entries. These are aggregated on
        executors (sum at same node_id+dim), then collected to the
        driver and scattered into a pre-allocated dense numpy array.

        For large types (>chunk_threshold nodes), collection is split
        into node_id ranges to bound peak Pandas memory on the driver.
        """
        layout = self._layout
        vector_dim = layout.vector_dim

        # Filter all inputs to this node type
        type_nodes = node_id_df.filter(
            F.col("node_type") == node_type
        ).select("uri", "node_id")

        type_type_uris = type_uri_df.filter(
            F.col("node_type") == node_type
        )

        type_node_props = node_properties_df.filter(
            F.col("node_type") == node_type
        )

        # ============================================
        # Compute all three segments (lazy on executors)
        # ============================================
        seg1_entries = self._encode_ontology_structure(
            node_type=node_type,
            type_nodes=type_nodes,
            type_type_uris=type_type_uris,
            class_hierarchy_df=class_hierarchy_df,
        )

        seg2_entries = self._encode_property_schema(
            node_type=node_type,
            type_nodes=type_nodes,
            type_type_uris=type_type_uris,
            type_node_props=type_node_props,
            property_schema_df=property_schema_df,
            property_hierarchy_df=property_hierarchy_df,
        )

        seg3_entries = self._encode_literal_values(
            node_type=node_type,
            numeric_df=numeric_df,
            categorical_df=categorical_df,
            norm_stats=norm_stats,
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
                num_nodes, vector_dim, dtype=torch.float32
            )

        combined = all_parts[0]
        for df in all_parts[1:]:
            combined = combined.unionAll(df)

        # Aggregate on executors: sum values at same (node_id, dim)
        combined = (
            combined
            .groupBy("node_id", "dim")
            .agg(F.sum("value").alias("value"))
            .select(
                F.col("node_id").cast("long"),
                F.col("dim").cast("int"),
                F.col("value").cast("float"),
            )
        )

        # ============================================
        # Collect to driver and scatter into dense array
        # ============================================
        tensor = np.zeros((num_nodes, vector_dim), dtype=np.float32)

        if num_nodes <= self._chunk_threshold:
            self._collect_and_scatter(combined, tensor, vector_dim)
        else:
            self._collect_and_scatter_chunked(
                combined, tensor, num_nodes, vector_dim
            )

        return torch.from_numpy(tensor).contiguous()

    def _collect_and_scatter(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        vector_dim: int,
    ) -> None:
        """
        Collect all sparse entries and scatter into dense array.
        Used for small-to-medium node types.
        """
        pdf = combined.toPandas()

        if not pdf.empty:
            node_ids = pdf["node_id"].values
            dims = pdf["dim"].values
            values = pdf["value"].values

            valid_mask = (dims >= 0) & (dims < vector_dim)
            tensor[
                node_ids[valid_mask], dims[valid_mask]
            ] = values[valid_mask]

        del pdf
        gc.collect()

    def _collect_and_scatter_chunked(
        self,
        combined: DataFrame,
        tensor: np.ndarray,
        num_nodes: int,
        vector_dim: int,
    ) -> None:
        """
        Collect sparse entries in chunks by node_id range and scatter
        into the pre-allocated dense array incrementally.
        """
        chunk_size = self._chunk_threshold

        combined = combined.cache()
        total_entries = combined.count()

        logger.info(
            f"      Chunked collection: {num_nodes:,} nodes, "
            f"{total_entries:,} sparse entries, "
            f"chunk size {chunk_size:,}"
        )

        num_chunks = (num_nodes + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            lo = chunk_idx * chunk_size
            hi = min(lo + chunk_size, num_nodes)

            chunk_df = combined.filter(
                (F.col("node_id") >= lo) & (F.col("node_id") < hi)
            )

            pdf = chunk_df.toPandas()

            if not pdf.empty:
                node_ids = pdf["node_id"].values
                dims = pdf["dim"].values
                values = pdf["value"].values

                valid_mask = (dims >= 0) & (dims < vector_dim)
                tensor[
                    node_ids[valid_mask], dims[valid_mask]
                ] = values[valid_mask]

            del pdf
            gc.collect()

            if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                logger.info(
                    f"      Chunk {chunk_idx + 1}/{num_chunks} complete "
                    f"(nodes {lo:,}–{hi:,})"
                )

        combined.unpersist()

    # ================================================================
    # Segment 1: Ontology Structure encoding
    # ================================================================

    def _encode_ontology_structure(
        self,
        node_type: str,
        type_nodes: DataFrame,
        type_type_uris: DataFrame,
        class_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 1: class identity, class hierarchy, and
        ontology/source membership.

        All dim indices are derived from self._layout.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Class Identity ---
        class_identity = (
            type_type_uris
            .select("node_id", "type_uri")
            .distinct()
        )

        ci_start = layout.seg1_class_identity_start
        ci_dim = layout.seg1_class_identity_dim

        for seed_offset in _HASH_SEEDS:
            ci_encoded = class_identity.select(
                F.col("node_id"),
                (
                    F.abs(F.hash(F.col("type_uri"), F.lit(seed_offset)))
                    % F.lit(ci_dim)
                    + F.lit(ci_start)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(ci_encoded)

        # --- Sub-segment 1b: Class Hierarchy ---
        ch_start = layout.seg1_class_hierarchy_start
        ch_dim = layout.seg1_class_hierarchy_dim

        if class_hierarchy_df.head(1):
            node_supers = (
                type_type_uris
                .select(
                    F.col("node_id"),
                    F.col("type_uri").alias("class_uri"),
                )
                .join(class_hierarchy_df, "class_uri", "inner")
                .select("node_id", "superclass_uri", "depth")
            )

            if node_supers.head(1):
                node_supers = node_supers.withColumn(
                    "weight",
                    F.lit(1.0) / F.col("depth").cast("double"),
                )

                for seed_offset in _HASH_SEEDS[:2]:
                    hier_encoded = node_supers.select(
                        F.col("node_id"),
                        (
                            F.abs(
                                F.hash(
                                    F.col("superclass_uri"),
                                    F.lit(seed_offset + 100),
                                )
                            )
                            % F.lit(ch_dim)
                            + F.lit(ch_start)
                        ).alias("dim"),
                        F.col("weight").alias("value"),
                    )
                    parts.append(hier_encoded)

        # --- Sub-segment 1c: Ontology/Source Membership ---
        os_start = layout.seg1_ontology_source_start
        os_dim = layout.seg1_ontology_source_dim

        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                type_type_uris
                .filter(F.col("type_uri").startswith(namespace))
                .select("node_id")
                .distinct()
                .withColumn(
                    "dim",
                    F.lit(os_start + (onto_idx % os_dim)),
                )
                .withColumn("value", F.lit(1.0))
            )
            parts.append(ns_match)

        # Also encode from node URI namespace
        node_ns_parts = self._encode_node_uri_namespace(
            type_nodes, os_start, os_dim
        )
        if node_ns_parts is not None:
            parts.append(node_ns_parts)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    def _encode_node_uri_namespace(
        self,
        type_nodes: DataFrame,
        os_start: int,
        os_dim: int,
    ) -> Optional[DataFrame]:
        """
        Encode ontology membership from the node's own URI namespace.
        """
        parts = []
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                type_nodes
                .filter(F.col("uri").startswith(namespace))
                .select("node_id")
                .distinct()
                .withColumn(
                    "dim",
                    F.lit(
                        os_start
                        + ((onto_idx + os_dim // 2) % os_dim)
                    ),
                )
                .withColumn("value", F.lit(0.5))
            )
            parts.append(ns_match)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result

    # ================================================================
    # Segment 2: Property Schema encoding
    # ================================================================

    def _encode_property_schema(
        self,
        node_type: str,
        type_nodes: DataFrame,
        type_type_uris: DataFrame,
        type_node_props: DataFrame,
        property_schema_df: DataFrame,
        property_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 2: property presence, domain/range signals,
        and property hierarchy.

        All dim indices are derived from self._layout.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 2a: Property Presence ---
        pp_start = layout.seg2_property_presence_start
        pp_dim = layout.seg2_property_presence_dim

        if type_node_props.head(1):
            for seed_offset in _HASH_SEEDS[:3]:
                pp_encoded = type_node_props.select(
                    F.col("node_id"),
                    (
                        F.abs(
                            F.hash(
                                F.col("predicate"),
                                F.lit(seed_offset + 200),
                            )
                        )
                        % F.lit(pp_dim)
                        + F.lit(pp_start)
                    ).alias("dim"),
                    F.lit(1.0).alias("value"),
                )
                parts.append(pp_encoded)

        # --- Sub-segment 2b: Domain/Range Signals ---
        dr_start = layout.seg2_domain_range_start
        dr_dim = layout.seg2_domain_range_dim
        dr_half = max(1, dr_dim // 2)

        if type_node_props.head(1) and property_schema_df.head(1):
            prop_with_schema = (
                type_node_props
                .select("node_id", "predicate")
                .join(
                    property_schema_df,
                    type_node_props["predicate"]
                    == property_schema_df["property_uri"],
                    "inner",
                )
                .drop("property_uri")
            )

            if prop_with_schema.head(1):
                domain_entries = (
                    prop_with_schema
                    .filter(F.col("domain_uri").isNotNull())
                    .select(
                        F.col("node_id"),
                        (
                            F.abs(
                                F.hash(F.col("domain_uri"), F.lit(300))
                            )
                            % F.lit(dr_half)
                            + F.lit(dr_start)
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                )
                parts.append(domain_entries)

                range_entries = (
                    prop_with_schema
                    .filter(F.col("range_uri").isNotNull())
                    .select(
                        F.col("node_id"),
                        (
                            F.abs(
                                F.hash(F.col("range_uri"), F.lit(301))
                            )
                            % F.lit(dr_dim - dr_half)
                            + F.lit(dr_start + dr_half)
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                )
                parts.append(range_entries)

        # --- Sub-segment 2c: Property Hierarchy ---
        ph_start = layout.seg2_property_hierarchy_start
        ph_dim = layout.seg2_property_hierarchy_dim

        if (type_node_props.head(1)
                and property_hierarchy_df.head(1)):
            prop_with_super = (
                type_node_props
                .select("node_id", "predicate")
                .join(
                    property_hierarchy_df,
                    type_node_props["predicate"]
                    == property_hierarchy_df["property_uri"],
                    "inner",
                )
                .drop("property_uri")
            )

            if prop_with_super.head(1):
                for seed_offset in _HASH_SEEDS[:2]:
                    ph_encoded = prop_with_super.select(
                        F.col("node_id"),
                        (
                            F.abs(
                                F.hash(
                                    F.col("super_property_uri"),
                                    F.lit(seed_offset + 400),
                                )
                            )
                            % F.lit(ph_dim)
                            + F.lit(ph_start)
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                    parts.append(ph_encoded)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Segment 3: Literal Values encoding
    # ================================================================

    def _encode_literal_values(
        self,
        node_type: str,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        norm_stats: Optional[DataFrame],
    ) -> Optional[DataFrame]:
        """
        Encode Segment 3: numeric values in hashed slots and categorical
        values as multi-hot hash encoding.

        All dim indices are derived from self._layout.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Numeric Values ---
        num_start = layout.seg3_numeric_start
        num_dim = layout.seg3_numeric_dim

        if numeric_df is not None:
            type_numeric = numeric_df.filter(
                F.col("node_type") == node_type
            )

            if type_numeric.head(1):
                if norm_stats is not None and self._normalize:
                    type_numeric = (
                        type_numeric
                        .join(
                            F.broadcast(norm_stats),
                            "predicate",
                            "left",
                        )
                        .withColumn(
                            "normalized_value",
                            (
                                F.col("numeric_value")
                                - F.coalesce(F.col("mu"), F.lit(0.0))
                            )
                            / F.coalesce(F.col("sigma"), F.lit(1.0)),
                        )
                        .drop("mu", "sigma")
                    )
                    value_col = "normalized_value"
                else:
                    type_numeric = type_numeric.withColumn(
                        "normalized_value", F.col("numeric_value")
                    )
                    value_col = "normalized_value"

                num_encoded = type_numeric.select(
                    F.col("node_id"),
                    (
                        F.abs(F.hash(F.col("predicate"), F.lit(500)))
                        % F.lit(num_dim)
                        + F.lit(num_start)
                    ).alias("dim"),
                    F.col(value_col).alias("value"),
                )
                parts.append(num_encoded)

        # --- Sub-segment 3b: Categorical Values ---
        cat_start = layout.seg3_categorical_start
        cat_dim = layout.seg3_categorical_dim

        if categorical_df is not None:
            type_cat = categorical_df.filter(
                F.col("node_type") == node_type
            )

            if type_cat.head(1):
                for seed_offset in _HASH_SEEDS:
                    cat_encoded = type_cat.select(
                        F.col("node_id"),
                        (
                            F.abs(
                                F.hash(
                                    F.concat(
                                        F.col("predicate"),
                                        F.lit("::"),
                                        F.col("cat_value"),
                                    ),
                                    F.lit(seed_offset + 600),
                                )
                            )
                            % F.lit(cat_dim)
                            + F.lit(cat_start)
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                    parts.append(cat_encoded)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )