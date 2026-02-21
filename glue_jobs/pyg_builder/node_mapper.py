"""
Node Mapper — Triples to PyG Node ID Tables

Assigns integer IDs to RDF entities for PyG HeteroData construction.
All work runs on Spark executors; only a small counts dict is collected.

Responsibilities:
1. Discover node types from rdf:type triples (or use config whitelist)
2. Assign each entity a canonical node type (handles multi-type entities)
3. Assign per-type integer IDs (0-indexed within each type) via Spark
4. Return a cached DataFrame on executors + a small counts dict

Node type naming:
    Full URI: https://www.bls.gov/cpi/Index
    PyG name: cpi_Index

    Full URI: https://financial-data.org/PriceObservation
    PyG name: market_PriceObservation

    The namespace prefix is extracted from known namespace mappings.
    Unknown namespaces fall back to a hash-based short prefix.

Multi-type entities:
    An entity may have multiple rdf:type triples (e.g., a CPI entity
    that is both cpi:Index and owl:NamedIndividual). We assign each
    entity to exactly one canonical type using priority:
    1. If config specifies node_types, first match wins
    2. Otherwise, most specific type (fewest instances) wins
    This avoids double-counting nodes across types.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Types to always exclude — these are meta-ontology types, not data nodes
_EXCLUDED_TYPE_PREFIXES = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
)

_EXCLUDED_TYPES = {
    "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#Thing",
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2000/01/rdf-schema#Resource",
    "http://www.w3.org/2000/01/rdf-schema#Class",
}

# ============================================
# Known namespace → prefix mappings
# ============================================
# Order matters — longer prefixes checked first to avoid partial matches
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
]

# Temporal and sector type URI fragments for config-based filtering
_TEMPORAL_TYPE_FRAGMENTS = ("Month", "Year", "Quarter", "UnifiedMonth",
                            "UnifiedYear", "UnifiedQuarter", "TimePeriod")
_SECTOR_TYPE_FRAGMENTS = ("EconomicSector", "Sector")


def _uri_to_pyg_name(type_uri: str) -> str:
    """
    Convert a full type URI to a PyG-compatible node type name.

    Examples:
        https://www.bls.gov/cpi/Index           -> cpi_Index
        https://financial-data.org/PriceObservation -> market_PriceObservation
        https://www.sec.gov/ontology/litigation#LitigationRelease -> sec_lit_LitigationRelease
        https://example.org/unified/UnifiedMonth -> unified_UnifiedMonth

    Unknown namespaces produce: unknown_<last_segment>
    """
    for namespace, prefix in _NAMESPACE_PREFIXES:
        if type_uri.startswith(namespace):
            local_name = type_uri[len(namespace):]
            # Clean up any remaining URI artifacts
            local_name = local_name.strip("/").strip("#")
            if local_name:
                return f"{prefix}_{local_name}"

    # Fallback: extract last segment after / or #
    for sep in ("#", "/"):
        if sep in type_uri:
            local_name = type_uri.rsplit(sep, 1)[-1]
            if local_name:
                return f"unknown_{local_name}"

    return f"unknown_{abs(hash(type_uri)) % 100000}"


# Register as UDF for Spark
_uri_to_pyg_name_udf = F.udf(_uri_to_pyg_name)


class NodeMapper:
    """
    Maps RDF entities to PyG integer node IDs.

    All heavy work (type discovery, deduplication, ID assignment) runs
    on Spark executors. Only a small Dict[str, int] of node counts
    is collected to the driver.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config

        # Config options
        self._requested_node_types = config.get("node_types", None)
        self._include_temporal = config.get("include_temporal_nodes", True)
        self._include_sector = config.get("include_sector_nodes", True)

    def build_node_id_table(
        self, triples_df: DataFrame
    ) -> Tuple[DataFrame, Dict[str, int]]:
        """
        Build a node ID table from rdf:type triples.

        Returns:
            Tuple of:
            - node_id_df: DataFrame(uri: string, node_id: long, node_type: string)
              Cached on executors. node_id is 0-indexed within each node_type.
            - node_counts: Dict[str, int] mapping node_type -> count.
              Small dict collected to driver (bounded by number of ontology classes).
        """
        # ============================================
        # Step 1: Extract all (entity, type_uri) pairs from rdf:type triples
        # ============================================
        type_triples = (
            triples_df
            .filter(F.col("predicate") == RDF_TYPE)
            .select(
                F.col("subject").alias("uri"),
                F.col("object").alias("type_uri"),
            )
        )

        # ============================================
        # Step 2: Filter out meta-ontology types (OWL, RDFS, RDF)
        # ============================================
        excluded_set = F.lit(False)
        for excluded_type in _EXCLUDED_TYPES:
            excluded_set = excluded_set | (F.col("type_uri") == excluded_type)

        excluded_prefix = F.lit(False)
        for prefix in _EXCLUDED_TYPE_PREFIXES:
            excluded_prefix = excluded_prefix | F.col("type_uri").startswith(prefix)

        type_triples = type_triples.filter(~(excluded_set | excluded_prefix))

        # ============================================
        # Step 3: Convert type URIs to PyG names
        # ============================================
        type_triples = type_triples.withColumn(
            "node_type", _uri_to_pyg_name_udf(F.col("type_uri"))
        )

        # ============================================
        # Step 4: Apply config filters
        # ============================================
        type_triples = self._apply_config_filters(type_triples)

        # ============================================
        # Step 5: Assign canonical type per entity
        # ============================================
        # An entity may have multiple rdf:type triples. We pick the most
        # specific type (fewest total instances) to avoid assigning
        # everything to a broad superclass.
        #
        # Count instances per type
        type_counts = (
            type_triples
            .groupBy("node_type")
            .agg(F.count("*").alias("type_count"))
        )

        # Join counts back and pick the type with fewest instances per entity
        # (most specific). Ties broken alphabetically for determinism.
        ranked = (
            type_triples
            .join(F.broadcast(type_counts), "node_type", "inner")
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("uri")
                    .orderBy(F.col("type_count").asc(), F.col("node_type").asc())
                ),
            )
            .filter(F.col("rank") == 1)
            .select("uri", "node_type")
        )

        # ============================================
        # Step 6: Assign per-type integer IDs (0-indexed)
        # ============================================
        node_id_df = ranked.withColumn(
            "node_id",
            (F.row_number().over(
                Window.partitionBy("node_type").orderBy("uri")
            ) - 1).cast("long"),
        )

        # Cache — this is read by edge_mapper and feature_extractor
        node_id_df = node_id_df.cache()

        # ============================================
        # Step 7: Collect node counts (small — bounded by ontology size)
        # ============================================
        count_rows = (
            node_id_df
            .groupBy("node_type")
            .agg(F.count("*").alias("cnt"))
            .collect()
        )

        node_counts = {row.node_type: row.cnt for row in count_rows}

        total = sum(node_counts.values())
        logger.info(f"  Assigned IDs to {total:,} nodes across {len(node_counts)} types")

        # Log top 10 types by count
        sorted_types = sorted(node_counts.items(), key=lambda x: -x[1])
        for ntype, cnt in sorted_types[:10]:
            logger.info(f"    {ntype}: {cnt:,}")
        if len(sorted_types) > 10:
            logger.info(f"    ... and {len(sorted_types) - 10} more types")

        return node_id_df, node_counts

    def _apply_config_filters(self, type_triples: DataFrame) -> DataFrame:
        """
        Apply configuration-based filters to the type triples.

        Handles:
        - Explicit node_types whitelist
        - include_temporal_nodes toggle
        - include_sector_nodes toggle
        """
        # Whitelist filter: only keep types in the config list
        if self._requested_node_types:
            requested_set = set(self._requested_node_types)
            # Broadcast the whitelist for efficient filtering
            requested_df = self.spark.createDataFrame(
                [(t,) for t in requested_set], ["node_type"]
            )
            type_triples = type_triples.join(
                F.broadcast(requested_df), "node_type", "inner"
            )
            return type_triples

        # Temporal node filter
        if not self._include_temporal:
            for fragment in _TEMPORAL_TYPE_FRAGMENTS:
                type_triples = type_triples.filter(
                    ~F.col("node_type").contains(fragment)
                )

        # Sector node filter
        if not self._include_sector:
            for fragment in _SECTOR_TYPE_FRAGMENTS:
                type_triples = type_triples.filter(
                    ~F.col("node_type").contains(fragment)
                )

        return type_triples