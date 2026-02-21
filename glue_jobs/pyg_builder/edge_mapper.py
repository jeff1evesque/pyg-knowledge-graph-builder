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

Edge type naming:
    Predicate URI: https://www.bls.gov/enrichment/precedes
    Relation name: bls_enrichment_precedes

    Predicate URI: http://www.w3.org/2002/07/owl#sameAs
    Relation name: owl_sameAs

    Full PyG edge type: ("cpi_Index", "bls_enrichment_precedes", "cpi_Index")

Filtering:
- rdf:type triples are excluded (used for node typing, not edges)
- Triples where subject or object is not in node_id_df are excluded
  (literal-valued properties are handled by feature_extractor instead)
- Config edge_types whitelist filters by relation name
"""
import logging
from typing import Dict, Any, Tuple, List

import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Predicates to always exclude from edges — these are structural, not relational
_EXCLUDED_PREDICATES = {
    RDF_TYPE,
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
    "http://www.w3.org/2002/07/owl#imports",
}

# ============================================
# Known namespace → prefix mappings (same as node_mapper)
# ============================================
_NAMESPACE_PREFIXES: List[Tuple[str, str]] = [
    # BLS data sources
    ("https://www.bls.gov/cpi/", "cpi"),
    ("https://www.bls.gov/ppi/", "ppi"),
    ("https://www.bls.gov/eci/", "eci"),
    ("https://www.bls.gov/empsit/", "empsit"),
    ("https://www.bls.gov/jolts/", "jolts"),
    ("https://www.bls.gov/laus/", "laus"),
    ("https://www.bls.gov/metro/", "metro"),
    ("https://www.bls.gov/realer/", "realer"),
    ("https://www.bls.gov/wkyeng/", "wkyeng"),
    ("https://www.bls.gov/ximpim/", "ximpim"),
    # BLS enrichment
    ("https://www.bls.gov/enrichment/", "bls_enrichment"),
    # SEC data sources
    ("http://www.sec.gov/filings#", "filings"),
    ("https://www.sec.gov/ontology/administrative-proceedings#", "sec_admin"),
    ("https://www.sec.gov/ontology/litigation#", "sec_lit"),
    ("https://www.sec.gov/ontology/trading-suspensions#", "sec_susp"),
    # SEC enrichment
    ("https://www.sec.gov/enrichment/", "sec_enrichment"),
    # Market data
    ("https://financial-data.org/options/", "market_options"),
    ("https://financial-data.org/", "market"),
    # Market enrichment
    ("https://financial-data.org/enrichment/", "market_enrichment"),
    # NOAA / weather
    ("http://www.oasis-open.org/committees/emergency/cap/1.2/", "cap"),
    ("http://api.weather.gov/ontology/", "nws"),
    ("http://api.weather.gov/alerts/", "alert"),
    ("https://www.noaa.gov/", "noaa"),
    # NOAA enrichment
    ("https://www.noaa.gov/enrichment/", "noaa_enrichment"),
    # Unified / cross-source
    ("https://example.org/unified/", "unified"),
    # Standard ontologies
    ("http://www.w3.org/2002/07/owl#", "owl"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs"),
]


def _predicate_uri_to_relation_name(pred_uri: str) -> str:
    """
    Convert a predicate URI to a PyG-compatible relation name.

    Examples:
        https://www.bls.gov/enrichment/precedes -> bls_enrichment_precedes
        http://www.w3.org/2002/07/owl#sameAs    -> owl_sameAs
        https://www.bls.gov/cpi/hasCategory     -> cpi_hasCategory
    """
    for namespace, prefix in _NAMESPACE_PREFIXES:
        if pred_uri.startswith(namespace):
            local_name = pred_uri[len(namespace):]
            local_name = local_name.strip("/").strip("#")
            if local_name:
                return f"{prefix}_{local_name}"

    # Fallback: extract last segment after / or #
    for sep in ("#", "/"):
        if sep in pred_uri:
            local_name = pred_uri.rsplit(sep, 1)[-1]
            if local_name:
                return f"unknown_{local_name}"

    return f"unknown_{abs(hash(pred_uri)) % 100000}"


_predicate_to_relation_udf = F.udf(_predicate_uri_to_relation_name)


class EdgeMapper:
    """
    Maps RDF triples to PyG edge_index tensors.

    All heavy work (joining URIs to integer IDs, grouping by edge type)
    runs on Spark executors. Only compact [2, num_edges] integer arrays
    are collected to the driver, one edge type at a time.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        # Config options
        self._requested_edge_types = config.get("edge_types", None)

    def build_edge_indices(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
        node_counts: Dict[str, int],
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build edge_index tensors for all edge types.

        Joins triples with node_id_df on executors to resolve URIs to
        integer IDs, then collects per-edge-type [2, num_edges] tensors
        to the driver one type at a time.

        Args:
            triples_df: Enriched triples DataFrame (subject, predicate, object)
            node_id_df: Node ID table DataFrame (uri, node_id, node_type)
                        — cached on executors, produced by NodeMapper
            node_counts: Dict[str, int] of node type counts (for validation)

        Returns:
            Dict mapping (src_type, relation, dst_type) -> LongTensor[2, num_edges]
        """
        # ============================================
        # Step 1: Filter to edge-candidate triples
        # ============================================
        # Exclude rdf:type and other structural predicates
        excluded_list = list(_EXCLUDED_PREDICATES)
        edge_triples = triples_df.filter(
            ~F.col("predicate").isin(excluded_list)
        )

        # ============================================
        # Step 2: Join subject → node_id_df (resolve src URI → src_id + src_type)
        # ============================================
        src_lookup = node_id_df.select(
            F.col("uri").alias("_src_uri"),
            F.col("node_id").alias("src_id"),
            F.col("node_type").alias("src_type"),
        )

        edges_with_src = edge_triples.join(
            src_lookup,
            edge_triples["subject"] == src_lookup["_src_uri"],
            "inner",
        ).drop("_src_uri")

        # ============================================
        # Step 3: Join object → node_id_df (resolve dst URI → dst_id + dst_type)
        # ============================================
        # The inner join naturally filters out triples where the object is
        # a literal (not in node_id_df) — those are features, not edges.
        dst_lookup = node_id_df.select(
            F.col("uri").alias("_dst_uri"),
            F.col("node_id").alias("dst_id"),
            F.col("node_type").alias("dst_type"),
        )

        edges_resolved = edges_with_src.join(
            dst_lookup,
            edges_with_src["object"] == dst_lookup["_dst_uri"],
            "inner",
        ).drop("_dst_uri")

        # ============================================
        # Step 4: Derive relation names from predicate URIs
        # ============================================
        edges_resolved = edges_resolved.withColumn(
            "relation", _predicate_to_relation_udf(F.col("predicate"))
        )

        # ============================================
        # Step 5: Apply config filter on relation names
        # ============================================
        if self._requested_edge_types:
            edges_resolved = edges_resolved.filter(
                F.col("relation").isin(self._requested_edge_types)
            )

        # Select only the columns needed for edge indices
        edges_final = (
            edges_resolved
            .select("src_type", "src_id", "relation", "dst_type", "dst_id")
            .dropDuplicates(["src_type", "src_id", "relation", "dst_type", "dst_id"])
        )

        # Cache for the per-type collection loop
        edges_final = edges_final.cache()

        # ============================================
        # Step 6: Discover distinct edge types (small collect)
        # ============================================
        edge_type_rows = (
            edges_final
            .select("src_type", "relation", "dst_type")
            .distinct()
            .collect()
        )

        logger.info(f"  Discovered {len(edge_type_rows)} distinct edge types")

        # ============================================
        # Step 7: Collect each edge type's indices to driver
        # ============================================
        # Each iteration filters to one (src_type, relation, dst_type),
        # selects [src_id, dst_id], and converts to a torch tensor.
        # This keeps per-collection memory bounded by the size of one
        # edge type rather than all edges at once.
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for row in edge_type_rows:
            src_type = row.src_type
            relation = row.relation
            dst_type = row.dst_type

            edge_type_key = (src_type, relation, dst_type)

            # Filter to this edge type on executors
            type_edges = (
                edges_final
                .filter(
                    (F.col("src_type") == src_type)
                    & (F.col("relation") == relation)
                    & (F.col("dst_type") == dst_type)
                )
                .select(
                    F.col("src_id").cast("long"),
                    F.col("dst_id").cast("long"),
                )
            )

            # Collect via Arrow-optimized toPandas — transfers only int64 pairs
            pdf = type_edges.toPandas()

            if pdf.empty:
                continue

            src_ids = torch.from_numpy(pdf["src_id"].values).long()
            dst_ids = torch.from_numpy(pdf["dst_id"].values).long()
            edge_index = torch.stack([src_ids, dst_ids], dim=0)

            edge_indices[edge_type_key] = edge_index

            logger.info(
                f"    {edge_type_key}: {edge_index.shape[1]:,} edges"
            )

        edges_final.unpersist()

        return edge_indices