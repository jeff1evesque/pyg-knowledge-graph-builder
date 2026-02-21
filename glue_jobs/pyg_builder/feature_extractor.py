"""
Feature Extractor — Triples to PyG Feature Tensors

Extracts numeric and categorical properties from RDF triples and
constructs per-node-type feature tensors for PyG HeteroData.

Responsibilities:
1. Identify literal-valued triples (properties where object is not a node URI)
2. Classify properties as numeric or categorical
3. Pivot from long format (node_id, property, value) to wide format
   (node_id, feat_1, feat_2, ...) per node type — all on Spark executors
4. Optionally normalize numeric features
5. Collect per-type [num_nodes, num_features] FloatTensors to the driver

Feature isolation by node type:
    Each node type gets its own feature tensor containing only the
    properties that its entities actually use. A CPI Index node carries
    CPI-specific features (indexValue, relativeImportance, ~5-10 columns),
    not all 200+ properties from every ontology. This keeps per-type
    tensors compact even with 100+ ontologies.

Handling missing values:
    Not every entity has every property. After pivoting, missing values
    are filled with 0.0. This is standard for GNN input — the model
    learns to ignore zero-padded features.

Categorical encoding:
    Categorical properties (e.g., hasCategory → cpi:Food_Entity) are
    encoded as integer indices per property. The mapping is built on
    executors via dense_rank(). One-hot encoding is NOT used to avoid
    wide sparse matrices — integer encoding is more memory-efficient
    and works with PyG's embedding layers.

Expected config:
    {
        "feature_config": {
            "numeric_properties": ["indexValue", "changeValue", ...],
            "categorical_properties": ["hasCategory", "hasIndustry", ...],
            "normalize": true
        }
    }

If feature_config is absent, numeric properties are auto-discovered
by checking which literal objects parse as floats. Categorical properties
are not included unless explicitly listed (to avoid blowing up feature
dimensionality with arbitrary URI-valued properties).
"""
import logging
from typing import Dict, Any, List, Optional

import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Properties that are never features — structural or identity predicates
_NON_FEATURE_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2002/07/owl#imports",
}

# XSD datatype suffixes that indicate numeric literals
# Objects may appear as "295.8" or "295.8^^http://www.w3.org/2001/XMLSchema#decimal"
_NUMERIC_XSD_SUFFIXES = (
    "XMLSchema#decimal",
    "XMLSchema#float",
    "XMLSchema#double",
    "XMLSchema#integer",
    "XMLSchema#int",
    "XMLSchema#long",
    "XMLSchema#short",
    "XMLSchema#nonNegativeInteger",
    "XMLSchema#positiveInteger",
)


def _extract_numeric_value(raw_object: str) -> Optional[float]:
    """
    Extract a numeric value from an RDF literal object string.

    Handles:
        "295.8"                                          -> 295.8
        "295.8^^http://www.w3.org/2001/XMLSchema#decimal" -> 295.8
        "-0.3"                                           -> -0.3
        "not a number"                                   -> None
    """
    if raw_object is None:
        return None

    # Strip XSD datatype suffix if present
    value_str = raw_object.split("^^")[0].strip().strip('"')

    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None


_extract_numeric_udf = F.udf(_extract_numeric_value, "double")


class FeatureExtractor:
    """
    Extracts features from RDF triples and builds per-node-type tensors.

    All heavy work (filtering, joining, pivoting, normalization) runs on
    Spark executors. Only compact [num_nodes, num_features] float arrays
    are collected to the driver, one node type at a time.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        # Parse feature config
        feat_config = config.get("feature_config", {})
        self._numeric_properties = feat_config.get("numeric_properties", None)
        self._categorical_properties = feat_config.get("categorical_properties", None)
        self._normalize = feat_config.get("normalize", False)

    def build_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Build per-node-type feature tensors.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate, object)
            node_id_df: Node ID table DataFrame (uri, node_id, node_type)
                        — cached on executors, produced by NodeMapper
            node_counts: Dict[str, int] of node type counts

        Returns:
            Dict mapping node_type -> FloatTensor[num_nodes, num_features]
            Only includes node types that have at least one feature.
        """
        # ============================================
        # Step 1: Build numeric features table (on executors)
        # ============================================
        numeric_features_df = self._extract_numeric_features(
            triples_df, node_id_df
        )

        # ============================================
        # Step 2: Build categorical features table (on executors)
        # ============================================
        categorical_features_df = self._extract_categorical_features(
            triples_df, node_id_df
        )

        # ============================================
        # Step 3: Discover feature columns per node type (small collect)
        # ============================================
        # Combine numeric and categorical into one long-format table:
        #   (node_type, node_id, feature_name, feature_value)
        feature_parts: List[DataFrame] = []
        if numeric_features_df is not None:
            feature_parts.append(numeric_features_df)
        if categorical_features_df is not None:
            feature_parts.append(categorical_features_df)

        if not feature_parts:
            logger.info("  No features found")
            return {}

        if len(feature_parts) == 1:
            all_features_long = feature_parts[0]
        else:
            all_features_long = feature_parts[0].unionAll(feature_parts[1])

        all_features_long = all_features_long.cache()

        # Discover which feature columns exist per node type
        # This is a small collect — bounded by (num_node_types × num_properties)
        type_feature_rows = (
            all_features_long
            .select("node_type", "feature_name")
            .distinct()
            .collect()
        )

        # Group: node_type -> sorted list of feature names
        type_features: Dict[str, List[str]] = {}
        for row in type_feature_rows:
            ntype = row.node_type
            fname = row.feature_name
            if ntype not in type_features:
                type_features[ntype] = []
            type_features[ntype].append(fname)

        for ntype in type_features:
            type_features[ntype] = sorted(type_features[ntype])

        logger.info(f"  Feature columns per node type:")
        for ntype, fnames in sorted(type_features.items()):
            logger.info(f"    {ntype}: {len(fnames)} features")

        # ============================================
        # Step 4: Pivot and collect per node type
        # ============================================
        feature_tensors: Dict[str, torch.Tensor] = {}

        for node_type, feature_names in type_features.items():
            num_nodes = node_counts.get(node_type, 0)
            if num_nodes == 0:
                continue

            tensor = self._pivot_and_collect(
                all_features_long, node_type, feature_names, num_nodes
            )

            if tensor is not None:
                feature_tensors[node_type] = tensor

        all_features_long.unpersist()

        return feature_tensors

    # ================================================================
    # Numeric feature extraction
    # ================================================================

    def _extract_numeric_features(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> Optional[DataFrame]:
        """
        Extract numeric literal properties and join with node IDs.

        Returns DataFrame: (node_type, node_id, feature_name, feature_value)
        or None if no numeric features found.
        """
        # Filter out structural predicates
        excluded_list = list(_NON_FEATURE_PREDICATES)
        candidates = triples_df.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        # Try to parse object as numeric
        candidates = candidates.withColumn(
            "numeric_value", _extract_numeric_udf(F.col("object"))
        ).filter(F.col("numeric_value").isNotNull())

        if candidates.head(1) == []:
            logger.info("  No numeric literals found")
            return None

        # If config specifies numeric properties, filter to those
        if self._numeric_properties:
            # Match by local name — predicate URI ends with the property name
            prop_conditions = F.lit(False)
            for prop in self._numeric_properties:
                prop_conditions = prop_conditions | F.col("predicate").endswith(prop)
            candidates = candidates.filter(prop_conditions)

            if candidates.head(1) == []:
                logger.info("  No matching numeric properties found")
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

        # Extract feature name from predicate URI (local name after last / or #)
        numeric_df = numeric_df.withColumn(
            "feature_name",
            F.coalesce(
                F.regexp_extract(F.col("predicate"), r"[#/]([^#/]+)$", 1),
                F.col("predicate"),
            ),
        )

        # Select final columns
        result = numeric_df.select(
            "node_type",
            "node_id",
            "feature_name",
            F.col("numeric_value").alias("feature_value"),
        )

        # Handle duplicate properties per node (take mean)
        result = (
            result
            .groupBy("node_type", "node_id", "feature_name")
            .agg(F.mean("feature_value").alias("feature_value"))
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
        Each categorical property gets its own integer encoding (0-indexed).

        Returns DataFrame: (node_type, node_id, feature_name, feature_value)
        or None if no categorical features configured/found.
        """
        if not self._categorical_properties:
            return None

        # Filter to configured categorical predicates (match by local name)
        prop_conditions = F.lit(False)
        for prop in self._categorical_properties:
            prop_conditions = prop_conditions | F.col("predicate").endswith(prop)

        candidates = triples_df.filter(prop_conditions)

        if candidates.head(1) == []:
            logger.info("  No matching categorical properties found")
            return None

        # Join with node_id_df
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

        # Extract feature name from predicate URI
        cat_df = cat_df.withColumn(
            "feature_name",
            F.concat(
                F.lit("cat_"),
                F.coalesce(
                    F.regexp_extract(F.col("predicate"), r"[#/]([^#/]+)$", 1),
                    F.col("predicate"),
                ),
            ),
        )

        # Encode categorical values as integers per (node_type, feature_name)
        # Using dense_rank so values are 1-indexed (0 reserved for missing)
        cat_df = cat_df.withColumn(
            "feature_value",
            F.dense_rank().over(
                Window.partitionBy("node_type", "feature_name")
                .orderBy("object")
            ).cast("double"),
        )

        # Handle multiple values per node+property (take first/min rank)
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
            all_features_long: DataFrame(node_type, node_id, feature_name, feature_value)
            node_type: The node type to pivot
            feature_names: Sorted list of feature column names for this type
            num_nodes: Total number of nodes of this type (for zero-padding)

        Returns:
            FloatTensor[num_nodes, num_features] or None
        """
        if not feature_names:
            return None

        # Filter to this node type
        type_features = all_features_long.filter(
            F.col("node_type") == node_type
        )

        # Pivot: rows = node_id, columns = feature_name, values = feature_value
        # Passing explicit feature_names list to pivot() avoids an extra
        # distinct() scan — Spark knows the column set upfront.
        pivoted = (
            type_features
            .groupBy("node_id")
            .pivot("feature_name", feature_names)
            .agg(F.first("feature_value"))
        )

        # Fill missing values with 0.0
        pivoted = pivoted.fillna(0.0)

        # Optional normalization (z-score per column, on executors)
        if self._normalize:
            for fname in feature_names:
                col = F.col(f"`{fname}`")
                stats = pivoted.select(
                    F.mean(col).alias("mu"),
                    F.stddev(col).alias("sigma"),
                ).head(1)[0]

                mu = stats.mu or 0.0
                sigma = stats.sigma or 1.0
                if sigma == 0.0:
                    sigma = 1.0

                pivoted = pivoted.withColumn(
                    f"`{fname}`",
                    (col - F.lit(mu)) / F.lit(sigma),
                )

        # Select feature columns in deterministic order
        select_cols = [F.col("node_id").cast("long")] + [
            F.col(f"`{fname}`").cast("float") for fname in feature_names
        ]
        pivoted = pivoted.select(*select_cols)

        # Collect via Arrow-optimized toPandas()
        pdf = pivoted.toPandas()

        if pdf.empty:
            return None

        # Build dense tensor — ensure all node_ids 0..num_nodes-1 are present
        # Nodes without any features get all-zero rows
        import numpy as np

        num_features = len(feature_names)
        tensor = np.zeros((num_nodes, num_features), dtype=np.float32)

        # Fill in rows that have data
        node_ids = pdf["node_id"].values
        feature_values = pdf[feature_names].values.astype(np.float32)
        tensor[node_ids] = feature_values

        result = torch.from_numpy(tensor)

        logger.info(
            f"    [{node_type}] {num_nodes:,} nodes × "
            f"{num_features} features, "
            f"{(node_ids.shape[0] / num_nodes * 100):.1f}% populated"
        )

        return result