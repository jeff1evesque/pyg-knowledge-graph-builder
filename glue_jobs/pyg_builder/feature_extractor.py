"""
Feature Extractor — Triples to PyG Feature Tensors

Extracts numeric and categorical properties from RDF triples and
constructs per-node-type feature tensors for PyG HeteroData.

Responsibilities:
1. Identify literal-valued triples (properties where object is not a node URI)
2. Classify properties as numeric or categorical
3. Pivot from long format to wide format per node type — on Spark executors
4. Optionally z-score normalize numeric features (single-pass on executors)
5. Collect per-type [num_nodes, num_features] FloatTensors to the driver

Feature isolation by node type:
    Each node type gets its own feature tensor containing only the
    properties that its entities actually use. A CPI Index node carries
    CPI-specific features (~5-10 columns), not all 200+ properties
    from every ontology.

Categorical encoding:
    Categorical properties are encoded as integer indices via dense_rank()
    on executors. 1-indexed (0 reserved for missing after fillna).
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_NON_FEATURE_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2002/07/owl#imports",
}


class FeatureExtractor:
    """
    Extracts features from RDF triples and builds per-node-type tensors.

    All heavy work runs on Spark executors. Only compact float arrays
    are collected to the driver, one node type at a time.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        feat_config = config.get("feature_config", {})
        self._numeric_properties = feat_config.get(
            "numeric_properties", None
        )
        self._categorical_properties = feat_config.get(
            "categorical_properties", None
        )
        self._normalize = feat_config.get("normalize", False)

    def build_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """
        Build per-node-type feature tensors.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate, object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
            node_counts: Dict[str, int] of node type counts

        Returns:
            Tuple of:
            - Dict[node_type -> FloatTensor[num_nodes, num_features]]
            - Dict[node_type -> List[str]] feature name lists
        """
        # ============================================
        # Step 1: Identify literal triples (not in node_id_df)
        # ============================================
        # Anti-join: keep triples whose object is NOT a known entity URI.
        # This filters out edge triples, leaving only literal properties.
        # Much more efficient than running a UDF on every triple.
        literal_triples = triples_df.join(
            node_id_df.select(F.col("uri").alias("_obj_uri")),
            triples_df["object"] == F.col("_obj_uri"),
            "left_anti",
        )

        # Exclude structural predicates
        excluded_list = list(_NON_FEATURE_PREDICATES)
        literal_triples = literal_triples.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        # ============================================
        # Step 2: Extract numeric features (pure Spark, no UDF)
        # ============================================
        numeric_features_df = self._extract_numeric_features(
            literal_triples, node_id_df
        )

        # ============================================
        # Step 3: Extract categorical features
        # ============================================
        categorical_features_df = self._extract_categorical_features(
            triples_df, node_id_df
        )

        # ============================================
        # Step 4: Combine into long-format table
        # ============================================
        feature_parts: List[DataFrame] = []
        if numeric_features_df is not None:
            feature_parts.append(numeric_features_df)
        if categorical_features_df is not None:
            feature_parts.append(categorical_features_df)

        if not feature_parts:
            logger.info("  No features found")
            return {}, {}

        all_features_long = feature_parts[0]
        for df in feature_parts[1:]:
            all_features_long = all_features_long.unionAll(df)

        all_features_long = all_features_long.cache()

        # ============================================
        # Step 5: Discover feature columns per node type (small collect)
        # ============================================
        type_feature_rows = (
            all_features_long
            .select("node_type", "feature_name")
            .distinct()
            .collect()
        )

        type_features: Dict[str, List[str]] = {}
        for row in type_feature_rows:
            ntype = row.node_type
            fname = row.feature_name
            if ntype not in type_features:
                type_features[ntype] = []
            type_features[ntype].append(fname)

        for ntype in type_features:
            type_features[ntype] = sorted(type_features[ntype])

        logger.info("  Feature columns per node type:")
        for ntype, fnames in sorted(type_features.items()):
            logger.info(f"    {ntype}: {len(fnames)} features")

        # ============================================
        # Step 6: Pivot and collect per node type
        # ============================================
        feature_tensors: Dict[str, torch.Tensor] = {}
        feature_names: Dict[str, List[str]] = {}

        for node_type, fnames in type_features.items():
            num_nodes = node_counts.get(node_type, 0)
            if num_nodes == 0:
                continue

            tensor = self._pivot_and_collect(
                all_features_long, node_type, fnames, num_nodes
            )

            if tensor is not None:
                feature_tensors[node_type] = tensor
                feature_names[node_type] = fnames

        all_features_long.unpersist()

        return feature_tensors, feature_names

    # ================================================================
    # Numeric feature extraction (pure Spark — no Python UDF)
    # ================================================================

    def _extract_numeric_features(
        self,
        literal_triples: DataFrame,
        node_id_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Extract numeric literal properties and join with node IDs.

        Uses pure Spark cast("double") instead of a Python UDF.
        Cast returns null for non-numeric strings, which we filter out.

        Args:
            literal_triples: Triples with literal objects only
                             (already anti-joined against node_id_df)
            node_id_df: Node ID table (uri, node_id, node_type)

        Returns:
            DataFrame(node_type, node_id, feature_name, feature_value)
            or None if no numeric features found.
        """
        # Strip XSD datatype suffix if present, then cast to double.
        # Pure Spark — no UDF.
        #
        # Handles:
        #   "295.8"                                    -> 295.8
        #   "295.8^^http://...XMLSchema#decimal"       -> 295.8
        #   "-0.3"                                     -> -0.3
        #   "not a number"                             -> null (filtered out)
        #
        # Note: build_graph.py's N-Triples parser already strips datatype
        # suffixes, so most objects arrive as plain strings like "295.8".
        # The split("^^") handles any that slip through.
        candidates = literal_triples.withColumn(
            "numeric_value",
            F.split(F.col("object"), r"\^\^").getItem(0).cast("double"),
        ).filter(F.col("numeric_value").isNotNull())

        # Apply config whitelist if specified
        if self._numeric_properties:
            prop_conditions = F.lit(False)
            for prop in self._numeric_properties:
                prop_conditions = prop_conditions | F.col(
                    "predicate"
                ).endswith(prop)
            candidates = candidates.filter(prop_conditions)

        # Check if any candidates remain (bounded — reads 1 partition)
        if not candidates.head(1):
            logger.info("  No numeric literals found")
            return None

        # Join with node_id_df to get node_type and node_id
        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        numeric_df = candidates.join(
            node_lookup,
            candidates["subject"] == node_lookup["_node_uri"],
            "inner",
        ).drop("_node_uri")

        # Extract feature name from predicate URI (local name)
        numeric_df = numeric_df.withColumn(
            "feature_name",
            F.coalesce(
                F.regexp_extract(
                    F.col("predicate"), r"[#/]([^#/]+)$", 1
                ),
                F.col("predicate"),
            ),
        )

        # Handle duplicate properties per node (take mean)
        result = (
            numeric_df
            .groupBy("node_type", "node_id", "feature_name")
            .agg(F.mean("numeric_value").alias("feature_value"))
        )

        logger.info("  Numeric features extracted (lazy)")
        return result

    # ================================================================
    # Categorical feature extraction
    # ================================================================

    def _extract_categorical_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Extract categorical properties and encode as integer indices.

        Only runs if categorical_properties are explicitly listed in config.

        Returns:
            DataFrame(node_type, node_id, feature_name, feature_value)
            or None.
        """
        if not self._categorical_properties:
            return None

        prop_conditions = F.lit(False)
        for prop in self._categorical_properties:
            prop_conditions = prop_conditions | F.col(
                "predicate"
            ).endswith(prop)

        candidates = triples_df.filter(prop_conditions)

        if not candidates.head(1):
            logger.info("  No matching categorical properties found")
            return None

        node_lookup = node_id_df.select(
            F.col("uri").alias("_node_uri"),
            F.col("node_id"),
            F.col("node_type"),
        )

        cat_df = candidates.join(
            node_lookup,
            candidates["subject"] == node_lookup["_node_uri"],
            "inner",
        ).drop("_node_uri")

        cat_df = cat_df.withColumn(
            "feature_name",
            F.concat(
                F.lit("cat_"),
                F.coalesce(
                    F.regexp_extract(
                        F.col("predicate"), r"[#/]([^#/]+)$", 1
                    ),
                    F.col("predicate"),
                ),
            ),
        )

        # Integer-encode per (node_type, feature_name) via dense_rank
        cat_df = cat_df.withColumn(
            "feature_value",
            F.dense_rank()
            .over(
                Window.partitionBy("node_type", "feature_name").orderBy(
                    "object"
                )
            )
            .cast("double"),
        )

        # Multiple values per node+property: take min rank
        result = (
            cat_df
            .groupBy("node_type", "node_id", "feature_name")
            .agg(F.min("feature_value").alias("feature_value"))
        )

        logger.info("  Categorical features extracted (lazy)")
        return result

    # ================================================================
    # Pivot and collect
    # ================================================================

    def _pivot_and_collect(
        self,
        all_features_long: DataFrame,
        node_type: str,
        feature_names: List[str],
        num_nodes: int,
    ) -> Optional[torch.Tensor]:
        """
        Pivot long-format features to wide format for one node type,
        optionally normalize, and collect as a torch tensor.

        All pivoting and normalization runs on Spark executors.
        Only the final [num_nodes, num_features] float array is collected.

        Args:
            all_features_long: DataFrame(node_type, node_id, feature_name,
                               feature_value)
            node_type: The node type to pivot
            feature_names: Sorted list of feature column names
            num_nodes: Total nodes of this type (for zero-padding)

        Returns:
            FloatTensor[num_nodes, num_features] or None
        """
        if not feature_names:
            return None

        # Filter to this node type
        type_features = all_features_long.filter(
            F.col("node_type") == node_type
        )

        # Pivot: rows=node_id, columns=feature_name, values=feature_value
        pivoted = (
            type_features
            .groupBy("node_id")
            .pivot("feature_name", feature_names)
            .agg(F.first("feature_value"))
        )

        # Fill missing values
        pivoted = pivoted.fillna(0.0)

        # Optional z-score normalization — single pass for all columns
        if self._normalize and feature_names:
            # Compute mean and stddev for all feature columns in one action
            agg_exprs = []
            for fname in feature_names:
                col = F.col(f"`{fname}`")
                agg_exprs.append(F.mean(col).alias(f"_mu_{fname}"))
                agg_exprs.append(F.stddev(col).alias(f"_sigma_{fname}"))

            stats_row = pivoted.agg(*agg_exprs).head(1)[0]

            for fname in feature_names:
                mu = stats_row[f"_mu_{fname}"] or 0.0
                sigma = stats_row[f"_sigma_{fname}"] or 1.0
                if sigma == 0.0:
                    sigma = 1.0

                pivoted = pivoted.withColumn(
                    f"`{fname}`",
                    (F.col(f"`{fname}`") - F.lit(mu)) / F.lit(sigma),
                )

        # Select feature columns in deterministic order + node_id
        select_cols = [F.col("node_id").cast("long")] + [
            F.col(f"`{fname}`").cast("float") for fname in feature_names
        ]
        pivoted = pivoted.select(*select_cols)

        # Collect via Arrow-optimized toPandas — only numeric columns
        pdf = pivoted.toPandas()

        if pdf.empty:
            return None

        # Build dense tensor with all node IDs 0..num_nodes-1
        num_features = len(feature_names)
        tensor = np.zeros((num_nodes, num_features), dtype=np.float32)

        node_ids = pdf["node_id"].values
        feature_values = pdf[feature_names].values.astype(np.float32)
        tensor[node_ids] = feature_values

        result = torch.from_numpy(tensor).contiguous()

        populated_pct = node_ids.shape[0] / max(num_nodes, 1) * 100
        logger.info(
            f"    [{node_type}] {num_nodes:,} nodes × "
            f"{num_features} features, "
            f"{populated_pct:.1f}% populated"
        )

        del pdf  # release Pandas memory immediately

        return result