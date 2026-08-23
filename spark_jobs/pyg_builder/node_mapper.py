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
    Full URI: https://jefflevesque.com/ontology/cpi/Index
    PyG name: cpi_Index

    Full URI: https://jefflevesque.com/ontology/market-feeds/PriceObservation
    PyG name: market_feeds_PriceObservation

Multi-type entities:
    An entity may have multiple rdf:type triples. We assign each entity
    to exactly one canonical type using priority:
    1. If config specifies node_types, first match wins
    2. Otherwise, a type in _CANONICAL_TYPE_PRIORITY wins (temporal types, which
       must not shard across per-namespace source types -- see that constant)
    3. Otherwise, a type that SUBSUMES another of the entity's types loses to
       nothing and wins -- see _subsumed_node_types()
    4. Otherwise, most specific type (fewest instances) wins
"""
import logging
from typing import Dict, Any, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from spark_jobs.utils.rdf_utils import NAMESPACE_PREFIXES
from spark_jobs.utils.spark_rdf_utils import collect_sorted

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

_TEMPORAL_TYPE_FRAGMENTS = (
    "Month", "Year", "Quarter", "Day", "UnifiedMonth",
    "UnifiedYear", "UnifiedQuarter", "UnifiedDay", "TimePeriod",
)

# Node types that must win the canonical-type contest outright, ahead of the
# "fewest instances" heuristic.
#
# TemporalUnifier types EVERY source period it collects as temporal_Source*,
# and many of those URIs already carry a source type (cpi:2024 is both cpi:Year
# and temporal_SourceYear). Left to instance counts the source type always wins,
# because it is per-namespace and therefore rarer: cpi:2024 -> cpi_Year (4),
# eci:2024 -> eci_Year (4), and so on. That splits one semantic concept across
# as many node types as there are namespaces -- measured on the e2e fixtures,
# 37 months over 5 types and 14 years over 5 types, leaving temporal_SourceYear
# holding a single node. The owl:sameAs edges from Unified{Month,Year} then land
# on whichever shard each period fell into, and a GNN treats those shards as
# unrelated types with no path between them.
#
# Pinning the temporal types collapses the shards back into one node type per
# granularity, which is the whole point of the unifier. The source type is not
# lost -- it stays an rdf:type triple on the entity and is still available as a
# feature; only the *canonical* type used for graph structure is overridden.
#
# SourceDay is here for the same reason as the other three, and matters more:
# the day grain exists to take load off the month hub, and a day that sharded
# across temporal_sec_Day / temporal_market_Day / temporal_noaa_Day would put
# the three sources back on separate nodes -- undoing the bridge instead of
# narrowing it.
_CANONICAL_TYPE_PRIORITY = (
    "temporal_SourceMonth",
    "temporal_SourceYear",
    "temporal_SourceQuarter",
    "temporal_SourceDay",
)
_SECTOR_TYPE_FRAGMENTS = ("EconomicSector", "Sector")


def _subsumed_node_types(
    type_triples: DataFrame,
    type_counts: DataFrame,
) -> List[str]:
    """Node types that are a SUBCLASS of another type the same entities carry.

    The canonical-type contest used to be settled by instance count, "most
    specific wins". A subclass is by definition rarer than its parent, so that
    rule always picks the subclass -- and shards one semantic concept across as
    many node types as the source happens to sub-classify:

        40 SEC filings became five node types (filings_SECFiling 33, Form4 3,
        Form8K 2, Form10K 1, Form10Q 1) because a Form 4 filing is typed BOTH
        sec-filings:Form4 and sec-filings:SECFiling. filings_hasIssuer then
        became five edge types pointing at the same filings_Issuer, and a GNN
        allocates five weight matrices where the relation is one.

    Subsumption is read off the INSTANCE DATA rather than any declared axiom,
    because no source declares rdfs:subClassOf (ontology_mapper.py:306) -- the
    hierarchy exists only as co-typing. A is a subclass of B when every instance
    of A is also a B and B is strictly larger:

        cooccur(A, B) == count(A)  and  count(B) > count(A)

    Extension containment is exactly what subsumption means for the purpose this
    serves, and it is the only evidence available. The strict `>` matters: two
    types with IDENTICAL extensions subsume each other, and returning both would
    make the winner depend on join order. Excluding them leaves the existing
    count/name tie-break to settle it deterministically.

    Chains need no transitive closure. If A ⊑ B ⊑ C then A's instances are all
    C's too, so (A, C) is derived directly and both A and B come back subsumed,
    leaving C -- the most general -- to win.

    Because containment is global, membership depends only on the type: if A ⊑ B
    then EVERY entity carrying A also carries B, so there is no entity for which
    A is the most general choice. That is what lets this return a flat set of
    type names rather than a per-entity judgement.

    This generalizes _CANONICAL_TYPE_PRIORITY, which pins the same outcome for
    temporal types by hand. That table is kept: it also covers periods the
    unifier typed only partially, where containment does not hold and the
    inference correctly declines to fire.

    COLLECTED to the driver rather than returned as a DataFrame, and this is a
    performance contract, not a style choice. Returned lazily, the self-join
    below becomes part of node_id_df's lineage, and build_features references
    that frame across ~29 collects and ~73 unions -- every one of which drags
    the join subtree along. Measured on the encode path, the lazy form cost
    1.95x (10.0s -> 19.6s per build) with an IDENTICAL number of Spark calls:
    the same operations, each executing a bigger plan. The result is one row
    per subsumed type -- bounded by the node-type count, ~80 on the e2e
    fixtures -- so materializing it is cheap and it collapses to a literal
    predicate at the call site.

    Returns:
        Sorted list of node type names that lost. May be empty. Sorted so the
        emitted plan is identical across runs on the same data.
    """
    sub_side = type_triples.select(
        F.col("uri"), F.col("node_type").alias("sub")
    ).distinct()
    sup_side = type_triples.select(
        F.col("uri"), F.col("node_type").alias("sup")
    ).distinct()

    co_occurrence = (
        sub_side
        .join(sup_side, "uri", "inner")
        .filter(F.col("sub") != F.col("sup"))
        .groupBy("sub", "sup")
        .agg(F.count("*").alias("cooccur"))
    )

    sub_counts = type_counts.select(
        F.col("node_type").alias("sub"),
        F.col("type_count").alias("sub_count"),
    )
    sup_counts = type_counts.select(
        F.col("node_type").alias("sup"),
        F.col("type_count").alias("sup_count"),
    )

    rows = (
        co_occurrence
        .join(F.broadcast(sub_counts), "sub", "inner")
        .join(F.broadcast(sup_counts), "sup", "inner")
        .filter(
            (F.col("cooccur") == F.col("sub_count"))
            & (F.col("sup_count") > F.col("sub_count"))
        )
        .select("sub")
        .distinct()
        .collect()
    )

    return sorted(row["sub"] for row in rows)


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

    for namespace, prefix in NAMESPACE_PREFIXES:
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


def build_type_uri_mapping(
    triples_df: DataFrame,
    node_id_df: DataFrame,
) -> Dict[str, str]:
    """
    Map each PyG node type name to the source type URI it was named after.

    A multi-type entity carries several rdf:type triples, so a node type has
    several candidate URIs and one has to be chosen. Choosing by sort order --
    which this did, keeping the alphabetically smallest -- picks a URI with no
    relationship to the node type at all whenever a supertype sorts earlier:

        jolts_HiresRate, jolts_JobOpeningsRate and jolts_QuitsRate every one
        reported bls_enrichment/RateMeasurement, because ".../enrichment/..."
        sorts before ".../jolts/...". Three node types, one URI.
        cpi_SpecialAggregateIndex reported cpi/ExpenditureCategory for the same
        reason.

    That is not cosmetic. graph_schema.json publishes it as `source_type_uri`,
    and ontology_schema.json additionally looks up `superclass_chain` and
    `defined_properties` BY it -- so a wrong URI silently attributes another
    class's hierarchy and property schema to this node type.

    The URI is instead chosen by inverting the naming rule: keep the candidate
    whose derived PyG name equals the node type. That is exact by construction
    (the name came from a URI through this very expression) and needs no
    priority table.

    A node type whose name came from somewhere other than a type URI -- the
    canonical-type overrides in _CANONICAL_TYPE_PRIORITY, or the sector
    override -- has no such candidate. Those fall back to the smallest URI,
    which is the previous behaviour: still deterministic, still stable across
    runs, and no worse than before for the cases the inversion cannot reach.

    Returns:
        Dict[pyg_name -> source type URI].
    """
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
        # The same expression NodeMapper named the node types with, so the
        # comparison below cannot drift from the naming rule.
        .withColumn("derived_name", _build_uri_to_pyg_name_expr("type_uri"))
    )
    # Sorted, NOT a bare collect(): Spark returns rows in task-completion
    # order, so the fallback below would pick a different URI on different
    # runs for any node type carrying multiple rdf:type URIs -- a content
    # non-determinism in the emitted metadata.
    type_uri_rows = collect_sorted(type_uri_rows)

    exact: Dict[str, str] = {}
    fallback: Dict[str, str] = {}
    for row in type_uri_rows:
        if row.derived_name == row.node_type:
            # First in sorted order still, so a type URI that somehow appears
            # twice resolves deterministically.
            exact.setdefault(row.node_type, row.type_uri)
        fallback.setdefault(row.node_type, row.type_uri)

    return {
        node_type: exact.get(node_type, uri)
        for node_type, uri in fallback.items()
    }


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
        # Step 5: Assign canonical type per entity
        #         (pinned > superclass > most specific)
        # ============================================
        type_counts = (
            type_triples
            .groupBy("node_type")
            .agg(F.count("*").alias("type_count"))
        )

        # Pinned types sort ahead of everything else; ties within the pinned set
        # follow declaration order, so a multi-granularity entity is still
        # deterministic. Everything unpinned keeps the existing behaviour.
        priority_expr = F.lit(len(_CANONICAL_TYPE_PRIORITY))
        for position, pinned_type in enumerate(_CANONICAL_TYPE_PRIORITY):
            priority_expr = F.when(
                F.col("node_type") == pinned_type, F.lit(position)
            ).otherwise(priority_expr)

        # A type that is a subclass of another type on the same entity sorts
        # last, so the most general of an entity's types wins. Ordered AHEAD of
        # type_count, which would otherwise pick the subclass every time (a
        # subclass is always rarer than its parent). Entities whose types stand
        # in no subsumption relation have every flag at 0 and fall through to
        # the count/name tie-break unchanged.
        subsumed = _subsumed_node_types(type_triples, type_counts)

        # A literal predicate rather than a join: `subsumed` is already a
        # driver-side list, so this adds one IN test to the plan instead of a
        # second scan of type_triples. The empty case is spelled out because
        # isin() over an empty list is a degenerate expression -- nothing is
        # subsumed, so every row scores 0 and the count/name tie-break decides,
        # exactly as it did before subsumption existed.
        is_subsumed_expr = (
            F.col("node_type").isin(subsumed).cast("int")
            if subsumed
            else F.lit(0)
        )

        ranked = (
            type_triples
            .join(F.broadcast(type_counts), "node_type", "inner")
            .withColumn("is_subsumed", is_subsumed_expr)
            .withColumn("type_priority", priority_expr)
            .withColumn(
                "rank",
                F.row_number().over(
                    Window.partitionBy("uri").orderBy(
                        F.col("type_priority").asc(),
                        F.col("is_subsumed").asc(),
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

    # Add this method to the existing NodeMapper class,
    # after build_node_id_table

    def get_type_uri_mapping(
        self,
        triples_df: DataFrame,
        node_id_df: DataFrame,
    ) -> Dict[str, str]:
        """
        Return a mapping from PyG node type name to source type URI.

        Small collect — one row per (node_type, type_uri) pair,
        deduplicated to one URI per type. Typically <500 rows.

        Called by constructor.py for metadata registration.
        """
        return build_type_uri_mapping(triples_df, node_id_df)

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