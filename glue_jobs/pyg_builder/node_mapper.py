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

Multi-type entities:
    An entity may have multiple rdf:type triples. We assign each entity
    to exactly one canonical type using priority:
    1. If config specifies node_types, first match wins
    2. Otherwise, most specific type (fewest instances) wins
"""
import logging
from typing import Dict, Any, Tuple, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ============================================
# URI constants
# ============================================
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_EXCLUDED_TYPES = {
    "http://www.w3.org/2002/07/owl#NamedIndividual",
    "http://www.w3.org/2002/07/owl#Thing",
    "http://www.w3.org/2002/07/owl#Class",
    "http://www.w3.org/2000/01/rdf-schema#Resource",
    "http://www.w3.org/2000/01/rdf-schema#Class",
}

_EXCLUDED_TYPE_PREFIXES = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
)

# ============================================
# Namespace → prefix mappings (longest prefix first)
# ============================================
_NAMESPACE_PREFIXES: List[Tuple[str, str]] = [
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
    ("https://www.bls.gov/enrichment/", "bls_enrichment"),
    ("http://www.sec.gov/filings#", "filings"),
    ("https://www.sec.gov/ontology/administrative-proceedings#", "sec_admin"),
    ("https://www.sec.gov/ontology/litigation#", "sec_lit"),
    ("https://www.sec.gov/ontology/trading-suspensions#", "sec_susp"),
    ("https://www.sec.gov/enrichment/", "sec_enrichment"),
    ("https://financial-data.org/options/", "market_options"),
    ("https://financial-data.org/enrichment/", "market_enrichment"),
    ("https://financial-data.org/", "market"),
    ("http://www.oasis-open.org/committees/emergency/cap/1.2/", "cap"),
    ("http://api.weather.gov/ontology/", "nws"),
    ("http://api.weather.gov/alerts/", "alert"),
    ("https://www.noaa.gov/enrichment/", "noaa_enrichment"),
    ("https://www.noaa.gov/", "noaa"),
    ("https://example.org/unified/", "unified"),
]

_TEMPORAL_TYPE_FRAGMENTS = (
    "Month", "Year", "Quarter", "UnifiedMonth",
    "UnifiedYear", "UnifiedQuarter", "TimePeriod",
)
_SECTOR_TYPE_FRAGMENTS = ("EconomicSector", "Sector")


def _build_uri_to_pyg_name_expr(uri_col: str = "type_uri") -> F.Column:
    """
    Build a pure-Spark Column expression that converts a type URI to a
    PyG-compatible node type name. No Python UDF — runs natively on the
    JVM across all executors.

    Strategy: chain of WHEN clauses checking startsWith for each known
    namespace prefix, extracting the local name via substring.
    Falls back to extracting the last segment after / or #.
    """
    col = F.col(uri_col)
    expr = None

    for namespace, prefix in _NAMESPACE_PREFIXES:
        ns_len = len(namespace)
        # local_name = everything after the namespace prefix
        local_name = F.substring(col, ns_len + 1, 1000)
        # Clean trailing / and #
        local_name = F.regexp_replace(local_name, r"^[/#]+|[/#]+$", "")
        pyg_name = F.concat(F.lit(f"{prefix}_"), local_name)

        condition = col.startswith(namespace) & (F.length(local_name) > 0)

        if expr is None:
            expr = F.when(condition, pyg_name)
        else:
            expr = expr.when(condition, pyg_name)

    # Fallback: extract last segment after # or /
    fallback_local = F.regexp_extract(col, r"[#/]([^#/]+)$", 1)
    fallback_name = F.concat(F.lit("unknown_"), fallback_local)

    expr = expr.otherwise(
        F.when(
            F.length(fallback_local) > 0, fallback_name
        ).otherwise(
            F.concat(F.lit("unknown_"), F.abs(F.hash(col)).cast("string"))
        )
    )

    return expr


class NodeMapper:
    """
    Maps RDF entities to PyG integer node IDs.

    All heavy work runs on Spark executors. Only a small Dict[str, int]
    of node counts is collected to the driver.
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
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
            - node_id_df: DataFrame(uri, node_id, node_type) cached on executors.
              node_id is 0-indexed within each node_type.
            - node_counts: Dict[str, int] — small, collected to driver.
        """
        # ============================================
        # Step 1: Extract (entity, type_uri) from rdf:type triples
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
        # Step 2: Filter out meta-ontology types
        # ============================================
        excluded_set = F.col("type_uri").isin(list(_EXCLUDED_TYPES))

        excluded_prefix = F.lit(False)
        for prefix in _EXCLUDED_TYPE_PREFIXES:
            excluded_prefix = excluded_prefix | F.col("type_uri").startswith(
                prefix
            )

        type_triples = type_triples.filter(
            ~(excluded_set | excluded_prefix)
        )

        # ============================================
        # Step 3: Convert type URIs to PyG names (pure Spark, no UDF)
        # ============================================
        type_triples = type_triples.withColumn(
            "node_type", _build_uri_to_pyg_name_expr("type_uri")
        )

        # ============================================
        # Step 4: Apply config filters
        # ============================================
        type_triples = self._apply_config_filters(type_triples)

        # ============================================
        # Step 5: Assign canonical type per entity (most specific wins)
        # ============================================
        type_counts = (
            type_triples
            .groupBy("node_type")
            .agg(F.count("*").alias("type_count"))
        )

        ranked = (
            type_triples
            .join(F.broadcast(type_counts), "node_type", "inner")
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("uri").orderBy(
                        F.col("type_count").asc(),
                        F.col("node_type").asc(),
                    )
                ),
            )
            .filter(F.col("rank") == 1)
            .select("uri", "node_type")
        )

        # ============================================
        # Step 6: Assign per-type 0-indexed integer IDs
        # ============================================
        node_id_df = ranked.withColumn(
            "node_id",
            (
                F.row_number().over(
                    Window.partitionBy("node_type").orderBy("uri")
                )
                - 1
            ).cast("long"),
        )

        # Cache and materialize — downstream modules (EdgeMapper,
        # FeatureExtractor) both join against this DataFrame.
        node_id_df = node_id_df.cache()
        node_id_df.count()  # force full materialization into cache

        # ============================================
        # Step 7: Collect node counts (small)
        # ============================================
        count_rows = (
            node_id_df
            .groupBy("node_type")
            .agg(F.count("*").alias("cnt"))
            .collect()
        )

        node_counts = {row.node_type: row.cnt for row in count_rows}

        total = sum(node_counts.values())
        logger.info(
            f"  Assigned IDs to {total:,} nodes "
            f"across {len(node_counts)} types"
        )

        sorted_types = sorted(node_counts.items(), key=lambda x: -x[1])
        for ntype, cnt in sorted_types[:10]:
            logger.info(f"    {ntype}: {cnt:,}")
        if len(sorted_types) > 10:
            logger.info(f"    ... and {len(sorted_types) - 10} more types")

        return node_id_df, node_counts

    def _apply_config_filters(self, type_triples: DataFrame) -> DataFrame:
        """Apply configuration-based filters to type triples."""
        if self._requested_node_types:
            requested_df = self.spark.createDataFrame(
                [(t,) for t in self._requested_node_types],
                ["node_type"],
            )
            return type_triples.join(
                F.broadcast(requested_df), "node_type", "inner"
            )

        # Temporal filter: single regex instead of per-fragment loop
        if not self._include_temporal:
            temporal_pattern = "|".join(_TEMPORAL_TYPE_FRAGMENTS)
            type_triples = type_triples.filter(
                ~F.col("node_type").rlike(temporal_pattern)
            )

        # Sector filter: single regex
        if not self._include_sector:
            sector_pattern = "|".join(_SECTOR_TYPE_FRAGMENTS)
            type_triples = type_triples.filter(
                ~F.col("node_type").rlike(sector_pattern)
            )

        return type_triples