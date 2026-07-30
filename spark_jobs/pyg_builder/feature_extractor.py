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

from spark_jobs.utils.rdf_utils import (
    NAMESPACE_PREFIXES,
    ONTOLOGY_NAMESPACE_INDICES,
)
from spark_jobs.utils.spark_rdf_utils import collect_sorted

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
#
# class_identity holds a multi-hot code per class, and a d-dim segment holds at
# most d linearly independent codes -- so this fraction, not the property count,
# is what bounds how many ontology CLASSES the encoding can represent. At the
# previous 0.25 it was 64 dims of a 1024-d vector: enough for the 44 classes of
# a two-source run, but not for the 118 of a full sec+noaa+market+BLS build,
# where identity stopped being recoverable while every code stayed distinct (so
# nothing looked wrong).
#
# The dims come from class_hierarchy rather than from vector_dim, which would
# have doubled driver memory. class_hierarchy is the cheapest source: it encodes
# rdfs:subClassOf chains, and with ontology mapping off -- the case in every
# build so far -- it is entirely zero. At 0.25 it still has 64 dims for 2 hashes
# per superclass once mapping is switched on.
_SEG1_CLASS_IDENTITY_FRAC = 0.50
_SEG1_CLASS_HIERARCHY_FRAC = 0.25
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

        # Metadata artifacts — populated during build_features()
        # by _collect_* methods. Small Python objects only.
        self._collected_norm_stats: Optional[List[Dict[str, Any]]] = None
        self._collected_zero_variance: List[str] = []
        self._collected_ontology_schema: Optional[Dict[str, Any]] = None
        self._collected_slot_mapping: Optional[Dict[str, Any]] = None

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

        # Ontology-wide existence flags, populated once per build_features()
        # run and reused across all node types (see the hoist in that method).
        # Conservative defaults so a direct encoder call can't AttributeError.
        self._has_class_hierarchy = True
        self._has_property_schema = True
        self._has_property_hierarchy = True

        # Metadata artifacts — populated by the _collect_* methods during
        # build_features(). Initialized here (not just inside the conditional
        # collect paths) so get_metadata_artifacts() is safe even when a build
        # has no numeric literals (normalization collection is then skipped).
        self._collected_norm_stats: Optional[List[Dict[str, Any]]] = None
        self._collected_zero_variance: List[str] = []
        self._collected_ontology_schema: Optional[Dict[str, Any]] = None
        self._collected_slot_mapping: Optional[Dict[str, Any]] = None

    def get_layout(self) -> "VectorLayout":
        """Return the VectorLayout instance for metadata registration."""
        return self._layout

    def get_encoding_config(self) -> Dict[str, Any]:
        """
        Return the complete encoding configuration needed to
        deterministically reproduce the hash-based encoding.

        All values here are the same constants used in the encoding
        methods below. If any of these change, the same ontology class
        or property hashes to different vector positions and the
        trained model breaks.

        Called by constructor.py after build_features() to register
        with MetadataCollector.
        """
        layout = self._layout
        return {
            "version": "1.0",
            "hash_algorithm": "spark_murmur3",
            "node_features": {
                "total_dim": layout.vector_dim,
                "segment_proportions": {
                    "ontology_structure": _SEG1_FRAC,
                    "property_schema": _SEG2_FRAC,
                    "literal_values": _SEG3_FRAC,
                },
                "class_identity": {
                    "dim": layout.seg1_class_identity_dim,
                    "num_hashes": len(_HASH_SEEDS),
                    "seeds": list(_HASH_SEEDS),
                },
                "class_hierarchy": {
                    "dim": layout.seg1_class_hierarchy_dim,
                    "num_hashes": 2,
                    "seeds": [
                        _HASH_SEEDS[0] + 100,
                        _HASH_SEEDS[1] + 100,
                    ],
                    "decay_function": "inverse_depth",
                    "max_depth": 10,
                },
                "ontology_source": {
                    "dim": layout.seg1_ontology_source_dim,
                    "method": "index_modulo",
                    "node_uri_weight": 0.5,
                },
                "property_presence": {
                    "dim": layout.seg2_property_presence_dim,
                    "num_hashes": 3,
                    "seeds": [s + 200 for s in _HASH_SEEDS[:3]],
                    "encoding_convention": {
                        "present": 1.0,
                        "absent": -1.0,
                        "not_in_schema": 0.0,
                    },
                },
                "domain_range": {
                    "dim": layout.seg2_domain_range_dim,
                    "seeds": [300, 301],
                },
                "property_hierarchy": {
                    "dim": layout.seg2_property_hierarchy_dim,
                    "num_hashes": 2,
                    "seeds": [s + 400 for s in _HASH_SEEDS[:2]],
                },
                "numeric_values": {
                    "dim": layout.seg3_numeric_dim,
                    "seed": 500,
                },
                "categorical_values": {
                    "dim": layout.seg3_categorical_dim,
                    "num_hashes": _NUM_CATEGORICAL_HASHES,
                    "seeds": [s + 600 for s in _HASH_SEEDS],
                },
            },
        }
        # NOTE: no "checksum" key here. The total dim it used to carry is
        # already recorded as node_features.total_dim, and a dimension is not a
        # checksum -- two builds with different seeds but the same vector width
        # produced identical values. The real contract digest is computed once
        # over the MERGED node+edge config in MetadataCollector, which is the
        # only place that sees the whole contract.

    def get_metadata_artifacts(self) -> Dict[str, Any]:
        """
        Return all metadata artifacts collected during build_features().

        Returns a dict with keys:
          - normalization_stats: list of per-predicate stat dicts
          - zero_variance_properties: list of predicate URIs
          - ontology_schema: frozen ontology structure dict
          - slot_mapping: dimension-to-meaning mapping dict

        All values are small Python objects — no tensors, no DataFrames.
        Populated by _collect_* methods called during build_features().
        """
        return {
            "normalization_stats": self._collected_norm_stats,
            "zero_variance_properties": self._collected_zero_variance,
            "ontology_schema": self._collected_ontology_schema,
            "slot_mapping": self._collected_slot_mapping,
        }

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

            # Collect stats for metadata — small collect, one row per
            # predicate (typically <200 predicates across all ontologies)
            self._collect_normalization_metadata(numeric_df)

        # ============================================
        # Get type URIs for ontology encoding
        # ============================================
        type_uri_df = self._get_type_uri_mapping(triples_df, node_id_df)

        # ============================================
        # Collect ontology schema and slot mapping for metadata.
        # All collect() calls here target small aggregated/distinct
        # DataFrames — never raw triples or per-node data.
        # ============================================
        self._collect_ontology_schema_metadata(
            triples_df, node_id_df, node_counts,
            class_hierarchy_df, property_schema_df,
        )
        self._collect_slot_mapping_metadata(
            numeric_df, categorical_df, type_uri_df,
            class_hierarchy_df,
        )

        # ============================================
        # Build ALL node-type vectors in ONE distributed pass.
        #
        # The encoders carry ``node_type`` through every projection, so every
        # type's sparse (node_type, node_id, dim, value) entries are produced
        # and aggregated together — a single Spark job-chain instead of one
        # per type (which was the dominant cost). node_id is 0-indexed within
        # a type, so the driver splits the collected frame back out by
        # node_type when scattering into per-type dense tensors.
        # ============================================
        logger.info("  Assembling feature vectors (single pass, all types)...")

        # Ontology-wide existence checks, evaluated ONCE (not once per type).
        self._has_class_hierarchy = bool(class_hierarchy_df.head(1))
        self._has_property_schema = bool(property_schema_df.head(1))
        self._has_property_hierarchy = bool(property_hierarchy_df.head(1))

        segment_names = [
            f"ontology_structure[{layout.seg1_start}:{layout.seg2_start}]",
            f"property_schema[{layout.seg2_start}:{layout.seg3_start}]",
            f"literal_values[{layout.seg3_start}:{layout.vector_dim}]",
        ]

        all_nodes = node_id_df.select("uri", "node_id", "node_type")

        seg1 = self._encode_ontology_structure(
            all_type_uris=type_uri_df,
            all_nodes=all_nodes,
            class_hierarchy_df=class_hierarchy_df,
        )
        seg2 = self._encode_property_schema(
            all_node_props=node_properties_df,
            property_schema_df=property_schema_df,
            property_hierarchy_df=property_hierarchy_df,
        )
        seg3 = self._encode_literal_values(
            numeric_df=numeric_df,
            categorical_df=categorical_df,
            norm_stats=norm_stats,
        )

        all_parts = [p for p in [seg1, seg2, seg3] if p is not None]

        feature_tensors: Dict[str, torch.Tensor] = {}
        feature_names: Dict[str, List[str]] = {}
        active_types = [
            (t, n) for t, n in node_counts.items() if n > 0
        ]

        if not all_parts:
            # No sparse entries at all — every type is a zero tensor.
            for node_type, num_nodes in active_types:
                feature_tensors[node_type] = torch.zeros(
                    num_nodes, vector_dim, dtype=torch.float32
                )
                feature_names[node_type] = segment_names
        else:
            combined = all_parts[0]
            for p in all_parts[1:]:
                combined = combined.unionAll(p)

            # Aggregate on executors: sum values at same (node_type, node_id,
            # dim). Cache the single result so the bounded per-batch collects
            # below read from cache instead of recomputing the encode plan.
            combined = (
                combined
                .groupBy("node_type", "node_id", "dim")
                .agg(F.sum("value").alias("value"))
                .select(
                    F.col("node_type"),
                    F.col("node_id").cast("long"),
                    F.col("dim").cast("int"),
                    F.col("value").cast("float"),
                )
                .cache()
            )

            self._scatter_all_types(
                combined, active_types, vector_dim,
                feature_tensors, feature_names, segment_names,
            )

            combined.unpersist()

        # Cleanup cached intermediates
        for df in [numeric_df, categorical_df, class_hierarchy_df,
                    property_schema_df]:
            if df is not None:
                try:
                    df.unpersist()
                except Exception:
                    pass

        return feature_tensors, feature_names

    def _scatter_all_types(
        self,
        combined: DataFrame,
        active_types: List[Tuple[str, int]],
        vector_dim: int,
        feature_tensors: Dict[str, "torch.Tensor"],
        feature_names: Dict[str, List[str]],
        segment_names: List[str],
    ) -> None:
        """
        Collect the single aggregated (node_type, node_id, dim, value) frame
        and scatter it into per-type dense tensors.

        Driver-memory discipline (preserves the #186 guarantee at cluster
        scale; a single batch on small data):
          - Large types (> chunk_threshold nodes) are collected one at a time
            via the chunked node_id-range path, so one big type can't blow the
            driver heap.
          - Small types are collected in node-count-bounded batches — one
            toPandas per batch instead of one per type.
        """
        import torch

        large = [
            (t, n) for t, n in active_types if n > self._chunk_threshold
        ]
        small = [
            (t, n) for t, n in active_types if n <= self._chunk_threshold
        ]

        for node_type, num_nodes in large:
            logger.info(
                f"    [{node_type}] {num_nodes:,} nodes (chunked collection)"
            )
            tensor = np.zeros((num_nodes, vector_dim), dtype=np.float32)
            type_combined = (
                combined
                .filter(F.col("node_type") == node_type)
                .select("node_id", "dim", "value")
            )
            self._collect_and_scatter_chunked(
                type_combined, tensor, num_nodes, vector_dim
            )
            feature_tensors[node_type] = torch.from_numpy(tensor).contiguous()
            feature_names[node_type] = segment_names
            gc.collect()

        budget = max(1, self._chunk_threshold)

        def flush(batch: List[Tuple[str, int]]) -> None:
            if not batch:
                return
            names = [t for t, _ in batch]
            pdf = (
                combined
                .filter(F.col("node_type").isin(names))
                .toPandas()
            )
            groups = (
                {nt: g for nt, g in pdf.groupby("node_type")}
                if not pdf.empty else {}
            )
            for node_type, num_nodes in batch:
                tensor = np.zeros(
                    (num_nodes, vector_dim), dtype=np.float32
                )
                g = groups.get(node_type)
                if g is not None and not g.empty:
                    node_ids = g["node_id"].values
                    dims = g["dim"].values
                    values = g["value"].values
                    valid_mask = (dims >= 0) & (dims < vector_dim)
                    tensor[
                        node_ids[valid_mask], dims[valid_mask]
                    ] = values[valid_mask]
                feature_tensors[node_type] = (
                    torch.from_numpy(tensor).contiguous()
                )
                feature_names[node_type] = segment_names
            del pdf
            gc.collect()

        batch: List[Tuple[str, int]] = []
        batch_nodes = 0
        for node_type, num_nodes in small:
            if batch and batch_nodes + num_nodes > budget:
                flush(batch)
                batch = []
                batch_nodes = 0
            batch.append((node_type, num_nodes))
            batch_nodes += num_nodes
        flush(batch)
        logger.info(
            f"    Collected {len(small)} small + {len(large)} large "
            f"node types"
        )

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
    # Metadata collection helpers
    # ================================================================
    # These methods collect small aggregated data to the driver for
    # the six metadata files. All collect() calls target DataFrames
    # with at most a few hundred rows (per-predicate stats, per-type
    # URIs, per-class hierarchy entries). No per-node or per-edge
    # data is ever collected here.

    def _collect_normalization_metadata(
        self, numeric_df: DataFrame
    ) -> None:
        """
        Collect normalization statistics to driver for metadata.

        Small collect — one row per predicate (typically <200
        predicates across all ontologies).
        """
        stats_with_counts = (
            numeric_df
            .groupBy("predicate")
            .agg(
                F.mean("numeric_value").alias("mu"),
                F.stddev("numeric_value").alias("sigma"),
                F.count("numeric_value").alias("count"),
            )
        )

        rows = collect_sorted(stats_with_counts)

        collected = []
        zero_variance = []
        for row in rows:
            mu = float(row.mu) if row.mu is not None else 0.0
            sigma = float(row.sigma) if row.sigma is not None else 0.0
            count = int(row["count"])

            if sigma == 0.0:
                zero_variance.append(row.predicate)
                sigma = 1.0

            collected.append({
                "predicate": row.predicate,
                "mu": mu,
                "sigma": sigma,
                "count": count,
            })

        self._collected_norm_stats = collected
        self._collected_zero_variance = zero_variance

        logger.info(
            f"    Collected normalization stats for "
            f"{len(collected)} predicates "
            f"({len(zero_variance)} zero-variance)"
        )

    def _collect_ontology_schema_metadata(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
        class_hierarchy_df: DataFrame,
        property_schema_df: DataFrame,
    ) -> None:
        """
        Collect ontology schema snapshot for metadata.

        All collect() calls here are on small distinct/aggregated
        DataFrames:
        - type_uri distinct: ~500 rows (one per type URI)
        - type_uri per node_type: ~1000 rows (types × multi-type)
        - class_hierarchy: ~5000 rows (classes × depth, transitive)
        - property_schema: ~500 rows (properties with domain/range)
        """
        # Collect type URI → PyG name mapping.
        # One row per distinct type URI — typically <500.
        type_mapping_rows = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(F.col("object").alias("type_uri"))
            .distinct()
        )
        type_mapping_rows = collect_sorted(type_mapping_rows)

        uri_to_pyg: Dict[str, str] = {}
        for row in type_mapping_rows:
            uri = row.type_uri
            for ns, prefix in NAMESPACE_PREFIXES:
                if uri.startswith(ns):
                    local = uri[len(ns):].strip("/#")
                    if local:
                        uri_to_pyg[uri] = f"{prefix}_{local}"
                    break

        # Collect per-node-type source URI.
        # One row per (node_type, type_uri) — typically <1000.
        type_uri_rows = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(
                F.col("subject").alias("_subj"),
                F.col("object").alias("type_uri"),
            )
            .join(
                node_id_df.select(
                    F.col("uri").alias("_subj"),
                    F.col("node_type"),
                ),
                "_subj",
                "inner",
            )
            .select("node_type", "type_uri")
            .distinct()
        )
        type_uri_rows = collect_sorted(type_uri_rows)

        type_uri_map: Dict[str, str] = {}
        for row in type_uri_rows:
            if row.node_type not in type_uri_map:
                type_uri_map[row.node_type] = row.type_uri

        # Collect class hierarchy — transitive closure.
        # Typically ~5000 rows (500 classes × avg depth ~10).
        hierarchy_rows = (
            collect_sorted(class_hierarchy_df)
            if class_hierarchy_df.head(1)
            else []
        )
        hierarchy_map: Dict[str, List[Tuple[str, int]]] = {}
        for row in hierarchy_rows:
            cls = row.class_uri
            if cls not in hierarchy_map:
                hierarchy_map[cls] = []
            hierarchy_map[cls].append(
                (row.superclass_uri, int(row.depth))
            )

        # Collect property schema — one row per property.
        # Typically <500 rows.
        prop_schema_rows = (
            collect_sorted(property_schema_df)
            if property_schema_df.head(1)
            else []
        )
        prop_schema_map: Dict[str, Dict[str, Optional[str]]] = {}
        for row in prop_schema_rows:
            prop_schema_map[row.property_uri] = {
                "domain": (
                    row.domain_uri if row.domain_uri else None
                ),
                "range": (
                    row.range_uri if row.range_uri else None
                ),
            }

        # Build per-node-type schema entries
        node_type_schemas: Dict[str, Dict[str, Any]] = {}
        for pyg_name in node_counts:
            source_uri = type_uri_map.get(pyg_name, "")
            namespace = ""
            for ns, prefix in NAMESPACE_PREFIXES:
                if source_uri.startswith(ns):
                    namespace = ns
                    break

            superclass_chain = []
            if source_uri in hierarchy_map:
                sorted_supers = sorted(
                    hierarchy_map[source_uri], key=lambda x: x[1]
                )
                superclass_chain = [
                    {"uri": uri, "depth": depth}
                    for uri, depth in sorted_supers
                ]

            defined_properties = []
            for prop_uri, schema in prop_schema_map.items():
                if schema.get("domain") == source_uri:
                    defined_properties.append({
                        "property_uri": prop_uri,
                        "range": schema.get("range"),
                    })

            node_type_schemas[pyg_name] = {
                "source_type_uri": source_uri,
                "superclass_chain": superclass_chain,
                "namespace": namespace,
                "defined_properties": defined_properties,
            }

        self._collected_ontology_schema = {
            "version": "1.0",
            "node_types": node_type_schemas,
            "uri_to_pyg_name": uri_to_pyg,
            "namespace_prefixes": {
                prefix: ns for ns, prefix in NAMESPACE_PREFIXES
            },
        }

        logger.info(
            f"    Collected ontology schema for "
            f"{len(node_type_schemas)} node types, "
            f"{len(uri_to_pyg)} URI mappings"
        )

    def _collect_slot_mapping_metadata(
        self,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        type_uri_df: DataFrame,
        class_hierarchy_df: DataFrame,
    ) -> None:
        """
        Collect slot mapping metadata — maps vector dimensions back
        to their semantic meaning.

        All collect() calls are on distinct single-column DataFrames:
        - numeric predicates: typically <100
        - categorical predicates: typically <100
        - type URIs: typically <500
        - superclass URIs: typically <200

        Hash computation is done on the driver using a Python
        approximation of Spark's murmur3. This is NOT used during
        encoding (which uses Spark's F.hash on executors). The
        slot_mapping file is for interpretability only — training
        and inference code never depends on it.
        """
        layout = self._layout

        # --- Numeric property slots ---
        numeric_slots = []
        if numeric_df is not None:
            pred_rows = (
                collect_sorted(
                    numeric_df.select("predicate").distinct()
                )
            )
            for row in pred_rows:
                pred = row.predicate
                local_name = (
                    pred.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                )
                slot = (
                    abs(_hash_approx(pred, 500))
                    % layout.seg3_numeric_dim
                )
                global_dim = slot + layout.seg3_numeric_start
                numeric_slots.append({
                    "predicate_uri": pred,
                    "local_name": local_name,
                    "hash_slot": slot,
                    "global_dim": global_dim,
                })

        # --- Categorical property slots ---
        categorical_slots = []
        if categorical_df is not None:
            cat_pred_rows = (
                collect_sorted(
                    categorical_df.select("predicate").distinct()
                )
            )
            for row in cat_pred_rows:
                pred = row.predicate
                local_name = (
                    pred.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                )
                slots = []
                global_dims = []
                for seed_offset in _HASH_SEEDS:
                    slot = (
                        abs(_hash_approx(
                            pred, seed_offset + 600
                        ))
                        % layout.seg3_categorical_dim
                    )
                    slots.append(slot)
                    global_dims.append(
                        slot + layout.seg3_categorical_start
                    )
                categorical_slots.append({
                    "predicate_uri": pred,
                    "local_name": local_name,
                    "hash_slots": slots,
                    "global_dims": global_dims,
                })

        # --- Class identity slots ---
        class_slots = []
        class_uri_rows = (
            collect_sorted(type_uri_df.select("type_uri").distinct())
        )
        for row in class_uri_rows:
            uri = row.type_uri
            pyg_name = ""
            for ns, prefix in NAMESPACE_PREFIXES:
                if uri.startswith(ns):
                    local = uri[len(ns):].strip("/#")
                    if local:
                        pyg_name = f"{prefix}_{local}"
                    break

            slots = []
            global_dims = []
            for seed_offset in _HASH_SEEDS:
                slot = (
                    abs(_hash_approx(uri, seed_offset))
                    % layout.seg1_class_identity_dim
                )
                slots.append(slot)
                global_dims.append(
                    slot + layout.seg1_class_identity_start
                )
            class_slots.append({
                "class_uri": uri,
                "pyg_name": pyg_name,
                "hash_slots": slots,
                "global_dims": global_dims,
            })

        # --- Namespace slots ---
        namespace_slots = []
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            prefix = ""
            for ns, p in NAMESPACE_PREFIXES:
                if ns == namespace:
                    prefix = p
                    break
            slot = onto_idx % layout.seg1_ontology_source_dim
            global_dim = slot + layout.seg1_ontology_source_start
            namespace_slots.append({
                "namespace": namespace,
                "prefix": prefix,
                "slot": slot,
                "global_dim": global_dim,
            })

        # --- Superclass hierarchy slots ---
        hierarchy_slots = []
        if class_hierarchy_df.head(1):
            super_rows = (
                collect_sorted(
                    class_hierarchy_df.select("superclass_uri").distinct()
                )
            )
            for row in super_rows:
                uri = row.superclass_uri
                slots = []
                global_dims = []
                for seed_offset in _HASH_SEEDS[:2]:
                    slot = (
                        abs(_hash_approx(uri, seed_offset + 100))
                        % layout.seg1_class_hierarchy_dim
                    )
                    slots.append(slot)
                    global_dims.append(
                        slot + layout.seg1_class_hierarchy_start
                    )
                hierarchy_slots.append({
                    "superclass_uri": uri,
                    "hash_slots": slots,
                    "global_dims": global_dims,
                })

        # --- Collision report ---
        collision_report = _compute_collision_report(
            numeric_slots, categorical_slots, class_slots,
            namespace_slots, hierarchy_slots,
            class_identity_dim=layout.seg1_class_identity_dim,
        )
        _warn_if_class_identity_is_saturated(collision_report)

        self._collected_slot_mapping = {
            # 1.1: class_identity in collision_report reports code
            # separability (distinct_codes, linearly_separable, headroom)
            # instead of slot occupancy. The 1.0 keys `collisions` /
            # `collision_rate` counted multi-hot slot reuse, which pigeonhole
            # forces high on a healthy code -- they are now `slot_reuse` /
            # `slot_reuse_rate` so the number cannot be read as lost identity.
            "version": "1.1",
            "numeric_properties": numeric_slots,
            "categorical_properties": categorical_slots,
            "classes": class_slots,
            "superclasses": hierarchy_slots,
            "namespaces": namespace_slots,
            "collision_report": collision_report,
        }

        logger.info(
            f"    Collected slot mappings: "
            f"{len(numeric_slots)} numeric, "
            f"{len(categorical_slots)} categorical, "
            f"{len(class_slots)} classes, "
            f"{len(hierarchy_slots)} superclasses, "
            f"{len(namespace_slots)} namespaces"
        )

    # ================================================================
    # Segment 1: Ontology Structure encoding
    # ================================================================

    def _encode_ontology_structure(
        self,
        all_type_uris: DataFrame,
        all_nodes: DataFrame,
        class_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 1: class identity, class hierarchy, and
        ontology/source membership — for ALL node types in one pass.

        Every projection carries ``node_type`` so the disjoint per-type
        vectors can be split apart on the driver (node_id is 0-indexed
        within a type). All dim indices are derived from self._layout.

        Inputs:
          all_type_uris: DataFrame(node_type, node_id, type_uri) — every type
          all_nodes:     DataFrame(uri, node_id, node_type) — every type

        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Class Identity ---
        class_identity = (
            all_type_uris
            .select("node_type", "node_id", "type_uri")
            .distinct()
        )

        ci_start = layout.seg1_class_identity_start
        ci_dim = layout.seg1_class_identity_dim

        for seed_offset in _HASH_SEEDS:
            ci_encoded = class_identity.select(
                F.col("node_type"),
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

        if self._has_class_hierarchy:
            node_supers = (
                all_type_uris
                .select(
                    F.col("node_type"),
                    F.col("node_id"),
                    F.col("type_uri").alias("class_uri"),
                )
                .join(class_hierarchy_df, "class_uri", "inner")
                .select("node_type", "node_id", "superclass_uri", "depth")
            )

            if node_supers.head(1):
                node_supers = node_supers.withColumn(
                    "weight",
                    F.lit(1.0) / F.col("depth").cast("double"),
                )

                for seed_offset in _HASH_SEEDS[:2]:
                    hier_encoded = node_supers.select(
                        F.col("node_type"),
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
                all_type_uris
                .filter(F.col("type_uri").startswith(namespace))
                .select("node_type", "node_id")
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
            all_nodes, os_start, os_dim
        )
        if node_ns_parts is not None:
            parts.append(node_ns_parts)

        if not parts:
            return None

        result = parts[0]
        for df in parts[1:]:
            result = result.unionAll(df)

        return result.select(
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    def _encode_node_uri_namespace(
        self,
        all_nodes: DataFrame,
        os_start: int,
        os_dim: int,
    ) -> Optional[DataFrame]:
        """
        Encode ontology membership from the node's own URI namespace, for
        all node types at once. Carries node_type through every projection.

        Input:  all_nodes: DataFrame(uri, node_id, node_type)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        parts = []
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                all_nodes
                .filter(F.col("uri").startswith(namespace))
                .select("node_type", "node_id")
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
        all_node_props: DataFrame,
        property_schema_df: DataFrame,
        property_hierarchy_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Encode Segment 2: property presence, domain/range signals, and
        property hierarchy — for ALL node types in one pass. Every
        projection carries ``node_type``.

        Input:  all_node_props: DataFrame(node_type, node_id, predicate)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # Existence check once across all types (not once per type).
        has_any_props = bool(all_node_props.head(1))

        # --- Sub-segment 2a: Property Presence ---
        pp_start = layout.seg2_property_presence_start
        pp_dim = layout.seg2_property_presence_dim

        if has_any_props:
            for seed_offset in _HASH_SEEDS[:3]:
                pp_encoded = all_node_props.select(
                    F.col("node_type"),
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

        if has_any_props and self._has_property_schema:
            prop_with_schema = (
                all_node_props
                .select("node_type", "node_id", "predicate")
                .join(
                    property_schema_df,
                    all_node_props["predicate"]
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
                        F.col("node_type"),
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
                        F.col("node_type"),
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

        if has_any_props and self._has_property_hierarchy:
            prop_with_super = (
                all_node_props
                .select("node_type", "node_id", "predicate")
                .join(
                    property_hierarchy_df,
                    all_node_props["predicate"]
                    == property_hierarchy_df["property_uri"],
                    "inner",
                )
                .drop("property_uri")
            )

            if prop_with_super.head(1):
                for seed_offset in _HASH_SEEDS[:2]:
                    ph_encoded = prop_with_super.select(
                        F.col("node_type"),
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
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

    # ================================================================
    # Segment 3: Literal Values encoding
    # ================================================================

    def _encode_literal_values(
        self,
        numeric_df: Optional[DataFrame],
        categorical_df: Optional[DataFrame],
        norm_stats: Optional[DataFrame],
    ) -> Optional[DataFrame]:
        """
        Encode Segment 3: numeric values in hashed slots and categorical
        values as multi-hot hash encoding — for ALL node types in one pass.
        numeric_df / categorical_df already carry node_type, so no per-type
        filtering is needed; node_type is projected through.

        Inputs:
          numeric_df:     DataFrame(node_type, node_id, predicate, numeric_value)
          categorical_df: DataFrame(node_type, node_id, predicate, cat_value)
        Returns DataFrame(node_type, node_id, dim, value) or None.
        """
        layout = self._layout
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Numeric Values ---
        num_start = layout.seg3_numeric_start
        num_dim = layout.seg3_numeric_dim

        if numeric_df is not None and numeric_df.head(1):
            all_numeric = numeric_df
            if norm_stats is not None and self._normalize:
                all_numeric = (
                    all_numeric
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
                all_numeric = all_numeric.withColumn(
                    "normalized_value", F.col("numeric_value")
                )
                value_col = "normalized_value"

            num_encoded = all_numeric.select(
                F.col("node_type"),
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

        if categorical_df is not None and categorical_df.head(1):
            for seed_offset in _HASH_SEEDS:
                cat_encoded = categorical_df.select(
                    F.col("node_type"),
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
            F.col("node_type"),
            F.col("node_id").cast("long"),
            F.col("dim").cast("int"),
            F.col("value").cast("float"),
        )

# ================================================================
# Module-level helpers for metadata collection
# ================================================================

def _hash_approx(value: str, seed: int) -> int:
    """
    Approximate Spark's murmur3 hash on the driver side for slot
    mapping metadata.

    This is NOT used during encoding (which uses Spark's F.hash on
    executors). It's used only for the slot_mapping metadata file to
    predict which slots each property/class occupies.

    Note: Spark's hash() uses a specific murmur3 variant. This Python
    approximation may not match exactly for all inputs. The slot_mapping
    file is for interpretability only — training and inference code
    never depends on it.
    """
    import hashlib

    combined = f"{value}:{seed}"
    h = hashlib.md5(combined.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _warn_if_class_identity_is_saturated(report: Dict[str, Any]) -> None:
    """
    Warn when the class_identity segment can no longer carry class identity.

    Two distinct failures, both silent: the graph builds, the vectors are the
    declared width, and every metadata file is internally consistent.

      * two classes share an identical code -- indistinguishable to any
        downstream model, whatever the segment width;
      * more classes than the segment has dimensions -- a d-dim segment holds
        at most d linearly independent codes, so beyond that a readout layer
        cannot recover class identity even though the codes stay distinct.
        Empirically the codes remain unique well past that point (4-hot into
        64 slots gives C(64,4) patterns), so distinctness alone will not warn
        anyone.

    Headroom is worth a nudge before it becomes a wall, because the class
    count grows with every source added and the segment does not.
    """
    ci = report.get("class_identity")
    if not ci:
        return

    shared = ci.get("classes_sharing_a_code") or []
    if shared:
        logger.warning(
            f"  {len(shared)} group(s) of classes share an identical "
            f"class_identity code and are indistinguishable downstream: "
            f"{shared[:3]}"
        )

    total, dim = ci.get("total_classes", 0), ci.get("segment_dim", 0)
    if not dim:
        return
    if total > dim:
        logger.warning(
            f"  class_identity segment is over-subscribed: {total} classes "
            f"into {dim} dims. At most {dim} codes can be linearly "
            f"independent, so class identity is no longer fully recoverable. "
            f"Raise feature_config.vector_dim, or reduce the class count."
        )
    elif total > 0.85 * dim:
        logger.warning(
            f"  class_identity segment is near capacity: {total} classes in "
            f"{dim} dims ({dim - total} left). Adding a source will "
            f"over-subscribe it."
        )


def _compute_collision_report(
    numeric_slots: List[Dict],
    categorical_slots: List[Dict],
    class_slots: List[Dict],
    namespace_slots: List[Dict],
    hierarchy_slots: List[Dict],
    class_identity_dim: int = 0,
) -> Dict[str, Any]:
    """
    Compute hash collision statistics across all slot assignments.

    Returns a report with collision counts and rates per sub-segment.

    Args:
        class_identity_dim: Width of the class_identity sub-segment. Needed
            because class identity is a multi-hot code whose health depends
            on the segment width, not on slot occupancy alone -- see the
            class_identity branch below.
    """
    dim = class_identity_dim
    report: Dict[str, Any] = {}

    if numeric_slots:
        dims_used = [s["global_dim"] for s in numeric_slots]
        unique_dims = len(set(dims_used))
        total = len(dims_used)
        collisions = total - unique_dims
        report["numeric_properties"] = {
            "total_properties": total,
            "unique_slots": unique_dims,
            "collisions": collisions,
            "collision_rate": (
                round(collisions / total, 4) if total > 0 else 0.0
            ),
        }

    if class_slots:
        all_dims: List[int] = []
        for s in class_slots:
            all_dims.extend(s["global_dims"])
        unique_dims = len(set(all_dims))
        total = len(all_dims)

        # Class identity is a MULTI-HOT code: each class occupies
        # num_hashes slots, and what identifies it is the set of slots, not
        # any single one. So slot reuse is not identity loss -- 44 classes x
        # 4 hashes into 64 slots reuses ~67% of slot entries while still
        # giving all 44 classes distinct codes, full rank, and a condition
        # number near 12. Reported as `collisions` / `collision_rate` (slot
        # mapping 1.0) that number read as "two thirds of class identity is
        # aliased", which is false and alarming: with total > dim, pigeonhole
        # forces a high value no matter how healthy the code is.
        #
        # What actually costs identity is measured instead:
        #   - two classes sharing an identical code (genuinely
        #     indistinguishable), and
        #   - the class count outgrowing the segment, past which no set of
        #     codes can be linearly separable, so a readout layer cannot
        #     recover class identity however distinct the codes look.
        codes = [tuple(sorted(set(s["global_dims"]))) for s in class_slots]
        by_code: Dict[Tuple[int, ...], List[str]] = {}
        for slot, code in zip(class_slots, codes):
            by_code.setdefault(code, []).append(
                slot.get("pyg_name") or slot.get("class_uri", "?")
            )
        shared = sorted(
            (sorted(names) for names in by_code.values() if len(names) > 1),
            key=lambda names: names[0],
        )
        max_overlap = 0
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                overlap = len(set(codes[i]) & set(codes[j]))
                if overlap > max_overlap:
                    max_overlap = overlap

        num_classes = len(class_slots)
        report["class_identity"] = {
            "total_classes": num_classes,
            "segment_dim": dim,
            # Raw occupancy, kept because it is a fact about the slots --
            # but named so it is not mistaken for lost identity.
            "total_hash_entries": total,
            "unique_slots": unique_dims,
            "slot_reuse": total - unique_dims,
            "slot_reuse_rate": (
                round((total - unique_dims) / total, 4) if total > 0 else 0.0
            ),
            # Identity, which is what a reader actually needs.
            "distinct_codes": len(by_code),
            "classes_sharing_a_code": shared,
            "max_pairwise_slot_overlap": max_overlap,
            # Capacity. A d-dim segment holds at most d linearly independent
            # codes, so this is a hard ceiling, not a heuristic.
            "capacity_classes": dim,
            "headroom_classes": (dim - num_classes) if dim else None,
            "linearly_separable": (
                bool(dim) and num_classes <= dim and not shared
            ),
        }

    if namespace_slots:
        dims_used = [s["global_dim"] for s in namespace_slots]
        unique_dims = len(set(dims_used))
        total = len(dims_used)
        collisions = total - unique_dims
        report["namespaces"] = {
            "total_namespaces": total,
            "unique_slots": unique_dims,
            "collisions": collisions,
            "collision_rate": (
                round(collisions / total, 4) if total > 0 else 0.0
            ),
        }

    return report