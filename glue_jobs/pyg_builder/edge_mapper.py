"""
Edge Mapper — Triples to PyG Edge Index Tensors

Resolves RDF triples into PyG edge_index tensors by joining with the
node ID table on Spark executors. Only compact integer arrays are
collected to the driver.

Responsibilities:
1. Identify edge triples (predicates linking two typed entities)
2. Join subject/object URIs against node_id_df to resolve integer IDs
3. Derive PyG edge types as (src_node_type, relation_name, dst_node_type)
4. Collect per-edge-type [2, num_edges] LongTensors to the driver
5. Return the cached resolved edges DataFrame for downstream consumers
   (EdgeFeatureExtractor) to avoid replaying the expensive double-join

Edge type naming:
    Predicate URI: https://www.bls.gov/enrichment/precedes
    Relation name: bls_enrichment_precedes

    Full PyG edge type: ("cpi_Index", "bls_enrichment_precedes", "cpi_Index")

Filtering:
- rdf:type and structural predicates are excluded
- Triples where subject or object is not in node_id_df are excluded
  (literal-valued properties are handled by feature_extractor)
- Config edge_types whitelist filters by relation name

Ordering contract:
- Within each edge type, edges are collected in deterministic order
  (src_id ASC, dst_id ASC). EdgeFeatureExtractor assigns edge_idx
  using the same Window ordering on executors, ensuring feature tensor
  rows align with edge_index tensor columns without any driver
  round-trip.
"""
import logging
from typing import Dict, Any, Tuple

import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from glue_jobs.utils.rdf_utils import NAMESPACE_PREFIXES

logger = logging.getLogger(__name__)

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


def _build_predicate_to_relation_expr(
    pred_col: str = "predicate",
) -> F.Column:
    """
    Build a pure-Spark Column expression that converts a predicate URI
    to a PyG-compatible relation name. No Python UDF.

    Same strategy as node_mapper's URI-to-name: chain of WHEN clauses
    for known namespace prefixes, with a fallback for unknown namespaces.
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

    # Fallback
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


class EdgeMapper:
    """
    Maps RDF triples to PyG edge_index tensors.

    All heavy work (joining URIs to integer IDs, grouping by edge type)
    runs on Spark executors. Only compact [2, num_edges] integer arrays
    are collected to the driver, one edge type at a time.

    The resolved edges DataFrame is returned alongside the tensors so
    that EdgeFeatureExtractor can reuse it without replaying the
    expensive double-join. The caller (constructor.py) is responsible
    for unpersisting it after all consumers are done.

    Ordering contract: within each edge type, edges are collected in
    deterministic order (src_id ASC, dst_id ASC). This allows
    EdgeFeatureExtractor to assign matching edge_idx values on
    executors without any driver round-trip.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self._requested_edge_types = config.get("edge_types", None)

    def build_edge_indices(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Tuple[
        Dict[Tuple[str, str, str], torch.Tensor],
        DataFrame,
    ]:
        """
        Build edge_index tensors for all edge types.

        Joins triples with node_id_df on executors to resolve URIs to
        integer IDs, then collects per-edge-type [2, num_edges] tensors
        to the driver one type at a time.

        The resolved edges DataFrame (edges_final) is returned cached
        on executors for reuse by EdgeFeatureExtractor. The caller
        must unpersist it when all downstream consumers are done.

        Ordering: within each edge type, edges are sorted by
        (src_id ASC, dst_id ASC) before collection. This deterministic
        ordering is the contract that EdgeFeatureExtractor relies on
        to align feature tensor rows with edge_index columns.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate,
                        object)
            node_id_df: Node ID table (uri, node_id, node_type) — cached
            node_counts: Dict[str, int] for validation

        Returns:
            Tuple of:
            - Dict mapping (src_type, relation, dst_type) ->
              LongTensor[2, N]
            - edges_final DataFrame (cached on executors) with columns
              (src_type, src_id, relation, dst_type, dst_id)
        """
        # ============================================
        # Step 1: Filter to edge-candidate triples
        # ============================================
        excluded_list = list(_EXCLUDED_PREDICATES)
        edge_triples = triples_df.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        # ============================================
        # Step 2: Double-join to resolve URIs → integer IDs
        # ============================================
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

        # ============================================
        # Step 3: Derive relation names (pure Spark, no UDF)
        # ============================================
        edges_resolved = edges_resolved.withColumn(
            "relation", _build_predicate_to_relation_expr("predicate")
        )

        # ============================================
        # Step 4: Apply config filter
        # ============================================
        if self._requested_edge_types:
            edges_resolved = edges_resolved.filter(
                F.col("relation").isin(self._requested_edge_types)
            )

        # Select only needed columns and deduplicate
        edges_final = (
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

        edges_final = edges_final.cache()

        # ============================================
        # Step 5: Discover distinct edge types (small collect)
        # ============================================
        edge_type_rows = (
            edges_final
            .select("src_type", "relation", "dst_type")
            .distinct()
            .collect()
        )

        logger.info(
            f"  Discovered {len(edge_type_rows)} distinct edge types"
        )

        # ============================================
        # Step 6: Collect each edge type's indices to driver
        #         in deterministic order (src_id, dst_id)
        # ============================================
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for row in edge_type_rows:
            src_type = row.src_type
            relation = row.relation
            dst_type = row.dst_type
            edge_type_key = (src_type, relation, dst_type)

            # Filter to this edge type, select only integer IDs,
            # sort deterministically for alignment with edge features
            type_edges = (
                edges_final
                .filter(
                    (F.col("src_type") == src_type)
                    & (F.col("relation") == relation)
                    & (F.col("dst_type") == dst_type)
                )
                .select("src_id", "dst_id")
                .orderBy("src_id", "dst_id")
            )

            # Collect via Arrow-optimized toPandas — only int64 pairs
            pdf = type_edges.toPandas()

            if pdf.empty:
                continue

            # Convert to tensor — from_numpy shares memory with numpy,
            # .contiguous() ensures a clean tensor for PyG
            import numpy as np

            src_ids = torch.from_numpy(
                pdf["src_id"].values.astype(np.int64)
            )
            dst_ids = torch.from_numpy(
                pdf["dst_id"].values.astype(np.int64)
            )
            edge_index = torch.stack(
                [src_ids, dst_ids], dim=0
            ).contiguous()

            edge_indices[edge_type_key] = edge_index

            logger.info(
                f"    {edge_type_key}: {edge_index.shape[1]:,} edges"
            )

            del pdf  # release Pandas memory immediately

        # NOTE: edges_final is NOT unpersisted here — it is returned
        # for reuse by EdgeFeatureExtractor. The caller (constructor.py)
        # is responsible for unpersisting it after all consumers finish.

        return edge_indices, edges_final