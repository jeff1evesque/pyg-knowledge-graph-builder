"""
Feature Extractor — Ontology-Aware Fixed-Width Node Feature Vectors

Constructs universal 1024-dimensional feature vectors that encode three
layers of information for every node:

  Segment 1 — Ontology Structure  (dims 0–255):   class identity,
              class hierarchy (rdfs:subClassOf chains), ontology/source
              membership

  Segment 2 — Property Schema     (dims 256–639):  property presence
              (which ontology-defined properties this node has),
              domain/range signals, property hierarchy

  Segment 3 — Literal Values      (dims 640–1023): numeric values in
              hashed slots (z-score normalized), categorical values as
              multi-hot hash encodings

All encoding runs on Spark executors using deterministic hash-based
functions expressed as pure Spark column expressions. Only the final
[num_nodes, 1024] float32 array per node type is collected to the driver.

Driver memory safety:
  The dense tensor for a node type is num_nodes × 1024 × 4 bytes.
  For large types (>500K nodes), this can exceed available driver memory.
  To prevent OOM:
  - Sparse (node_id, dim, value) entries are collected via toPandas()
    and scattered directly into a pre-allocated numpy array
  - The Pandas DataFrame is deleted immediately after scatter
  - For very large types, collection is chunked by node_id range
    so that at most ~500K nodes' sparse entries are in Pandas at once
  - The dense tensor itself is unavoidable (PyG requires it), but we
    ensure only ONE type's tensor + its Pandas intermediary coexist
"""
import logging
import gc
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from glue_jobs.utils.rdf_utils import ONTOLOGY_NAMESPACE_INDICES

logger = logging.getLogger(__name__)

# ============================================
# Vector layout constants
# ============================================
VECTOR_DIM = 1024

# Segment 1: Ontology Structure (dims 0–255)
SEG1_START = 0
SEG1_CLASS_IDENTITY_START = 0
SEG1_CLASS_IDENTITY_DIM = 64
SEG1_CLASS_HIERARCHY_START = 64
SEG1_CLASS_HIERARCHY_DIM = 128
SEG1_ONTOLOGY_SOURCE_START = 192
SEG1_ONTOLOGY_SOURCE_DIM = 64
SEG1_END = 256

# Segment 2: Property Schema (dims 256–639)
SEG2_START = 256
SEG2_PROPERTY_PRESENCE_START = 256
SEG2_PROPERTY_PRESENCE_DIM = 192
SEG2_DOMAIN_RANGE_START = 448
SEG2_DOMAIN_RANGE_DIM = 112
SEG2_PROPERTY_HIERARCHY_START = 560
SEG2_PROPERTY_HIERARCHY_DIM = 80
SEG2_END = 640

# Segment 3: Literal Values (dims 640–1023)
SEG3_START = 640
SEG3_NUMERIC_START = 640
SEG3_NUMERIC_DIM = 256
SEG3_CATEGORICAL_START = 896
SEG3_CATEGORICAL_DIM = 128
SEG3_END = 1024

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
# Maximum nodes per chunk when collecting sparse entries to driver.
# At ~15 sparse entries per node × 16 bytes/row = ~240 bytes/node,
# 500K nodes ≈ 120 MB Pandas overhead — safe alongside the dense tensor.
_CHUNK_NODE_THRESHOLD = 500_000

# Number of hash functions for multi-hot categorical encoding
_NUM_CATEGORICAL_HASHES = 4

# Seed offsets for independent hash functions
_HASH_SEEDS = [0, 7, 13, 31]


class FeatureExtractor:
    """
    Builds universal 1024-d ontology-aware feature vectors for all nodes.

    All heavy computation runs on Spark executors. Only compact float
    arrays are collected to the driver, one node type at a time.

    Driver memory safety:
      - Dense tensor is pre-allocated once per type (num_nodes × 1024 × 4B)
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
        vector_dim = self._vector_dim

        logger.info(
            f"  Building {vector_dim}-d ontology-aware feature vectors"
        )

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
                f"ontology_structure[0:{SEG1_END}]",
                f"property_schema[{SEG2_START}:{SEG2_END}]",
                f"literal_values[{SEG3_START}:{SEG3_END}]",
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

        Memory lifecycle on driver:
          1. Pre-allocate dense array: num_nodes × vector_dim × 4 bytes
          2. Collect sparse chunk via toPandas() (~120 MB per 500K nodes)
          3. Scatter into dense array (in-place, no copy)
          4. Delete Pandas DataFrame
          5. Repeat for next chunk (if chunked)
          6. Convert numpy → torch tensor (zero-copy via from_numpy)

        Peak driver memory for this type:
          dense_array + one_pandas_chunk
          = (num_nodes × vector_dim × 4) + (~240 bytes × chunk_size)
        """
        vector_dim = self._vector_dim

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
        # This handles hash collisions and multi-type/multi-property
        # contributions gracefully
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
        # Pre-allocate the dense tensor — this is the unavoidable cost
        tensor = np.zeros((num_nodes, vector_dim), dtype=np.float32)

        if num_nodes <= self._chunk_threshold:
            # Small type: single collect
            self._collect_and_scatter(combined, tensor, vector_dim)
        else:
            # Large type: chunked collect by node_id range to bound
            # peak Pandas memory on the driver
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

        Used for small-to-medium node types where the full Pandas
        DataFrame fits comfortably alongside the dense tensor.
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

        This bounds peak Pandas memory to ~chunk_threshold nodes' worth
        of sparse entries (~120 MB) regardless of total type size.

        The combined DataFrame is cached once on executors. Each chunk
        filters by node_id range, collects via toPandas(), scatters
        into the dense array, and frees the Pandas memory before the
        next chunk.
        """
        chunk_size = self._chunk_threshold

        # Cache the aggregated sparse entries on executors — each chunk
        # reads from this cached DataFrame with a simple range filter
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
        Encode Segment 1 (dims 0–255): class identity, class hierarchy,
        and ontology/source membership.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        parts: List[DataFrame] = []

        # --- Sub-segment 1a: Class Identity (dims 0–63) ---
        # Hash each rdf:type URI into multiple dims within [0, 64)
        # using deterministic hash functions. Multi-hot encoding.
        class_identity = (
            type_type_uris
            .select("node_id", "type_uri")
            .distinct()
        )

        for seed_offset in _HASH_SEEDS:
            ci_encoded = class_identity.select(
                F.col("node_id"),
                (
                    F.abs(F.hash(F.col("type_uri"), F.lit(seed_offset)))
                    % F.lit(SEG1_CLASS_IDENTITY_DIM)
                    + F.lit(SEG1_CLASS_IDENTITY_START)
                ).alias("dim"),
                F.lit(1.0).alias("value"),
            )
            parts.append(ci_encoded)

        # --- Sub-segment 1b: Class Hierarchy (dims 64–191) ---
        # For each node, find all superclasses via the hierarchy table
        # and hash them into dims [64, 192). Depth-weighted: closer
        # superclasses get higher values.
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
                            % F.lit(SEG1_CLASS_HIERARCHY_DIM)
                            + F.lit(SEG1_CLASS_HIERARCHY_START)
                        ).alias("dim"),
                        F.col("weight").alias("value"),
                    )
                    parts.append(hier_encoded)

        # --- Sub-segment 1c: Ontology/Source Membership (dims 192–255) ---
        # Multi-hot encoding of which ontology namespace(s) the node's
        # type URI belongs to. Uses canonical list from rdf_utils.
        for namespace, onto_idx in ONTOLOGY_NAMESPACE_INDICES:
            ns_match = (
                type_type_uris
                .filter(F.col("type_uri").startswith(namespace))
                .select("node_id")
                .distinct()
                .withColumn(
                    "dim",
                    F.lit(
                        SEG1_ONTOLOGY_SOURCE_START
                        + (onto_idx % SEG1_ONTOLOGY_SOURCE_DIM)
                    ),
                )
                .withColumn("value", F.lit(1.0))
            )
            parts.append(ns_match)

        # Also encode ontology membership from the node URI itself
        node_ns_parts = self._encode_node_uri_namespace(type_nodes)
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
        self, type_nodes: DataFrame
    ) -> Optional[DataFrame]:
        """
        Encode ontology membership from the node's own URI namespace.
        Adds a second signal beyond the type URI namespace.
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
                        SEG1_ONTOLOGY_SOURCE_START
                        + (
                            (onto_idx + SEG1_ONTOLOGY_SOURCE_DIM // 2)
                            % SEG1_ONTOLOGY_SOURCE_DIM
                        )
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
        Encode Segment 2 (dims 256–639): property presence, domain/range
        signals, and property hierarchy.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        parts: List[DataFrame] = []

        # --- Sub-segment 2a: Property Presence (dims 256–447) ---
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
                        % F.lit(SEG2_PROPERTY_PRESENCE_DIM)
                        + F.lit(SEG2_PROPERTY_PRESENCE_START)
                    ).alias("dim"),
                    F.lit(1.0).alias("value"),
                )
                parts.append(pp_encoded)

        # --- Sub-segment 2b: Domain/Range Signals (dims 448–559) ---
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
                            % F.lit(SEG2_DOMAIN_RANGE_DIM // 2)
                            + F.lit(SEG2_DOMAIN_RANGE_START)
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
                            % F.lit(SEG2_DOMAIN_RANGE_DIM // 2)
                            + F.lit(
                                SEG2_DOMAIN_RANGE_START
                                + SEG2_DOMAIN_RANGE_DIM // 2
                            )
                        ).alias("dim"),
                        F.lit(1.0).alias("value"),
                    )
                )
                parts.append(range_entries)

        # --- Sub-segment 2c: Property Hierarchy (dims 560–639) ---
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
                            % F.lit(SEG2_PROPERTY_HIERARCHY_DIM)
                            + F.lit(SEG2_PROPERTY_HIERARCHY_START)
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
        Encode Segment 3 (dims 640–1023): numeric values in hashed slots
        and categorical values as multi-hot hash encoding.

        Returns DataFrame(node_id: long, dim: int, value: float) or None.
        """
        parts: List[DataFrame] = []

        # --- Sub-segment 3a: Numeric Values (dims 640–895) ---
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
                        % F.lit(SEG3_NUMERIC_DIM)
                        + F.lit(SEG3_NUMERIC_START)
                    ).alias("dim"),
                    F.col(value_col).alias("value"),
                )
                parts.append(num_encoded)

        # --- Sub-segment 3b: Categorical Values (dims 896–1023) ---
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
                            % F.lit(SEG3_CATEGORICAL_DIM)
                            + F.lit(SEG3_CATEGORICAL_START)
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

    # ================================================================
    # Backward-compatible public API (deprecated)
    # ================================================================

    def build_features_legacy(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """
        Legacy per-type variable-width feature extraction.

        Deprecated — use build_features() for ontology-aware vectors.
        Kept for A/B comparison during migration.
        """
        logger.warning(
            "Using legacy per-type feature extraction. "
            "Consider switching to ontology-aware vectors."
        )
        from glue_jobs.pyg_builder._legacy_feature_extractor import (
            LegacyFeatureExtractor,
        )
        legacy = LegacyFeatureExtractor(self.spark, self.config)
        return legacy.build_features(triples_df, node_id_df, node_counts)