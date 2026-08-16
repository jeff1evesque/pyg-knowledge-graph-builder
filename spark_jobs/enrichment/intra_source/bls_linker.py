"""
BLS Intra-Source Enrichment Orchestrator — PySpark Implementation

Migrated from rdflib SPARQL-based enricher to fully distributed PySpark.
All operations run on executors — no rdflib, no driver data operations.

Coordinates enrichment across all BLS datasets:
- CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER, WKYENG

Enrichment steps:
1. Link temporal sequences (delegated to per-dataset sub-enrichers)
2. Apply sector patterns (keyword matching against category URIs)
3. Apply known correlations (cross-dataset economic relationships)
4. Link category hierarchies (parent-child from URI path structure)

Note: Temporal unification (Step 1 of the old rdflib orchestrator) is now
handled by the standalone TemporalUnifier in pipeline.py Phase 2.
It no longer runs inside the BLS intra-source enricher.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from functools import reduce
from typing import Dict, List, Optional, Set

from rdflib.namespace import RDF, RDFS

from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER, WKYENG,
    identifier_namespace,
)
from spark_jobs.enrichment.intra_source.bls.patterns import (
    BLS_SECTOR_CORRELATION, BLS_SECTOR_PATTERNS,
)
from spark_jobs.enrichment.intra_source.bls.correlations import KNOWN_CORRELATIONS
from spark_jobs.enrichment.intra_source.bls.base_enricher import (
    BLSDatasetEnricher, DATASET_ENRICHERS,
)
from spark_jobs.utils.spark_rdf_utils import deduplicate_against_existing

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

_RDF_TYPE = str(RDF.type)
_RDFS_LABEL = str(RDFS.label)
_BELONGS_TO_SECTOR = str(BLS_ENRICHMENT.belongsToSector)
_SECTOR_CORRELATION = str(BLS_SECTOR_CORRELATION)
_HAS_PARENT = str(BLS_ENRICHMENT.hasParent)

# IDENTIFIER prefixes for dataset detection and filtering.
#
# Every use of this map tests an ENTITY uri -- _detect_datasets samples
# subjects, _filter_bls_triples filters on subject, and the sector/correlation
# builders match category entities by namespace + keyword. Entities are
# individuals, so these must be the id/ namespaces. Keyed on the term
# namespaces they matched nothing: _detect_datasets returned an empty set and
# the linker logged "No BLS data detected, skipping enrichment" on input that
# was entirely BLS.
DATASET_NS_MAP: Dict[str, str] = {
    name: identifier_namespace(str(ns))
    for name, ns in (
        ('cpi', CPI),
        ('ppi', PPI),
        ('eci', ECI),
        ('jolts', JOLTS),
        ('empsit', EMPSIT),
        ('ximpim', XIMPIM),
        ('laus', LAUS),
        ('metro', METRO),
        ('realer', REALER),
        ('wkyeng', WKYENG),
    )
}

# All BLS identifier prefixes for filtering
_ALL_BLS_PREFIXES = list(DATASET_NS_MAP.values())


def normalize_keyword_for_uri_matching(keyword: str) -> str:
    """
    Normalize a keyword for matching against BLS URIs.

    BLS URIs use underscores and remove special characters:
        "Food at home" → "Food_at_home"
        "Owners' equivalent rent" → "Owners_equivalent_rent"
        "All items less food, shelter, and energy" → "All_items_less_food_shelter_and_energy"
    """
    return (
        keyword
        .replace(' ', '_')
        .replace("'", '')
        .replace(',', '')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')
    )


class BLSIntraSourceLinker:
    """
    PySpark-based BLS intra-source enrichment orchestrator.

    Handles orchestrator-level enrichment (sectors, correlations, hierarchies)
    and delegates temporal sequence linking to per-dataset sub-enrichers.

    All methods:
    1. Read from self._bls_triples (cached BLS subset)
    2. Return new triples DataFrames
    3. Never call .collect() except for bounded dataset detection
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._sub_enrichers: Dict[str, BLSDatasetEnricher] = {}

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run all BLS intra-source enrichment steps.

        Args:
            triples_df: DataFrame with (subject, predicate, object)

        Returns:
            DataFrame of NEW triples only
        """
        self.triples_df = triples_df
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        # Detect which BLS datasets are present
        self.available_datasets = self._detect_datasets()
        if not self.available_datasets:
            logger.info("No BLS data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting BLS Intra-Source Enrichment (PySpark)")
        logger.info(f"Detected BLS datasets: {', '.join(sorted(self.available_datasets))}")
        logger.info("=" * 60)

        # Cache the BLS-relevant subset for performance
        self._bls_triples = self._filter_bls_triples().cache()

        # Initialize per-dataset sub-enrichers
        self._initialize_sub_enrichers()

        new_dfs: List[DataFrame] = []

        # Step 1: Link temporal sequences (delegated to sub-enrichers)
        logger.info("\n[Step 1/4] Linking temporal sequences...")
        self._append(new_dfs, self._link_temporal_sequences())

        # Step 2: Apply sector patterns
        logger.info("\n[Step 2/4] Applying sector patterns...")
        self._append(new_dfs, self._apply_sector_patterns())

        # Step 3: Apply known correlations
        logger.info("\n[Step 3/4] Applying known correlations...")
        self._append(new_dfs, self._apply_known_correlations())

        # Step 4: Link category hierarchies
        logger.info("\n[Step 4/4] Linking category hierarchies...")
        self._append(new_dfs, self._link_category_hierarchies())

        # Unpersist cached subset
        self._bls_triples.unpersist()

        if not new_dfs:
            logger.info("BLS enrichment produced no new triples")
            return empty

        all_new = reduce(DataFrame.unionAll, new_dfs)
        all_new = all_new.dropDuplicates(["subject", "predicate", "object"])
        all_new = deduplicate_against_existing(all_new, triples_df)

        all_new = all_new.cache()
        total = all_new.count()

        logger.info("\n" + "=" * 60)
        logger.info(f"BLS Intra-Source Enrichment Complete: {total} new triples")
        logger.info("=" * 60)

        return all_new

    # ================================================================
    # Detection and filtering
    # ================================================================

    def _detect_datasets(self) -> Set[str]:
        """Detect which BLS datasets are present. Bounded collect."""
        datasets = set()

        # Sample subjects to detect which BLS namespaces are present
        sample = (
            self.triples_df
            .select("subject")
            .limit(200000)
            .distinct()
            .collect()
        )

        for row in sample:
            s = row.subject
            for dataset_name, ns in DATASET_NS_MAP.items():
                if s.startswith(ns):
                    datasets.add(dataset_name)
            if len(datasets) == len(DATASET_NS_MAP):
                break  # Found all possible datasets

        return datasets

    def _filter_bls_triples(self) -> DataFrame:
        """Filter triples to only BLS-relevant subjects."""
        bls_filter = F.col("subject").startswith(_ALL_BLS_PREFIXES[0])
        for p in _ALL_BLS_PREFIXES[1:]:
            bls_filter = bls_filter | F.col("subject").startswith(p)
        return self.triples_df.filter(bls_filter)

    def _initialize_sub_enrichers(self):
        """
        Initialize per-dataset enrichers from the registry.

        Each enricher is a BLSDatasetEnricher (or subclass) that
        implements link_temporal_sequences() for its measurement types.
        """
        for dataset_name in sorted(self.available_datasets):
            enricher_class = DATASET_ENRICHERS.get(dataset_name)
            if enricher_class:
                self._sub_enrichers[dataset_name] = enricher_class(
                    self.spark, dataset_name
                )
            else:
                logger.warning(f"No enricher registered for dataset: {dataset_name}")

        logger.info(
            f"  Initialized {len(self._sub_enrichers)} sub-enrichers: "
            f"{', '.join(sorted(self._sub_enrichers.keys()))}"
        )

    @staticmethod
    def _append(lst: list, item: Optional[DataFrame]):
        if item is not None:
            lst.append(item)

    # ================================================================
    # Step 1: Temporal sequences (delegated to sub-enrichers)
    # ================================================================

    def _link_temporal_sequences(self) -> Optional[DataFrame]:
        """
        Delegate temporal sequence linking to per-dataset sub-enrichers.

        Each sub-enricher reads from self._bls_triples and returns
        new precedes triples for its measurement types.
        """
        new_dfs: List[DataFrame] = []

        for dataset_name, enricher in sorted(self._sub_enrichers.items()):
            logger.info(f"  Processing {dataset_name}...")
            try:
                result = enricher.link_temporal_sequences(self._bls_triples)
                if result is not None:
                    new_dfs.append(result)
            except Exception as e:
                logger.error(f"  {dataset_name} temporal linking failed: {e}", exc_info=True)

        if not new_dfs:
            return None

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Temporal sequence triples prepared (lazy)")
        return result

    # ================================================================
    # Step 2: Sector patterns
    # ================================================================

    def _apply_sector_patterns(self) -> Optional[DataFrame]:
        """
        Apply sector-based linking patterns.

        Links category entities (URIs ending with _Entity) to economic
        sectors based on keyword matching against URI substrings.

        Example:
            cpi:All_items_Food_Food_at_home_Entity → belongsToSector → bls:FoodSector
        """
        # Build keyword lookup: (keyword_normalized, sector_uri, namespace)
        sector_rows = []
        for sector_name, pattern in BLS_SECTOR_PATTERNS.items():
            sector_uri = str(pattern['sector_uri'])
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                ns = DATASET_NS_MAP.get(dataset_name, '')
                if not ns or not keywords:
                    continue
                for keyword in keywords:
                    normalized = normalize_keyword_for_uri_matching(keyword)
                    sector_rows.append((normalized, sector_uri, ns))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows,
            schema=["keyword_normalized", "sector_uri", "namespace"],
        ).dropDuplicates(["keyword_normalized", "sector_uri", "namespace"])

        # Find all category entities (URIs ending with _Entity)
        category_entities = (
            self._bls_triples
            .select("subject")
            .distinct()
            .filter(F.col("subject").endswith("_Entity"))
        )

        # Join: category URI starts with namespace AND contains normalized keyword
        matched = (
            category_entities
            .join(
                F.broadcast(sector_df),
                on=(
                    F.col("subject").startswith(F.col("namespace")) &
                    F.col("subject").contains(F.col("keyword_normalized"))
                ),
                how="inner",
            )
            .dropDuplicates(["subject", "sector_uri"])
        )

        if matched.head(1) == []:
            return None

        # Sector type triples
        sector_type_triples = (
            matched.select("sector_uri").distinct()
            .select(
                F.col("sector_uri").alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(str(BLS_ENRICHMENT.EconomicSector)).alias("object"),
            )
        )

        # Sector label triples (derive label from URI local name)
        sector_label_triples = (
            matched.select("sector_uri").distinct()
            .withColumn(
                "label",
                F.regexp_replace(
                    F.regexp_extract(F.col("sector_uri"), r"[/#]([^/#]+)$", 1),
                    "([A-Z])", r" $1"
                )
            )
            .withColumn("label", F.trim(F.col("label")))
            .select(
                F.col("sector_uri").alias("subject"),
                F.lit(_RDFS_LABEL).alias("predicate"),
                F.col("label").alias("object"),
            )
        )

        # belongsToSector triples
        belongs_triples = matched.select(
            F.col("subject"),
            F.lit(_BELONGS_TO_SECTOR).alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        # Sector correlation triples — one predicate for every sector, the
        # sector itself carried by the object (see BLS_SECTOR_CORRELATION).
        rel_triples = matched.select(
            F.col("subject"),
            F.lit(_SECTOR_CORRELATION).alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        result = (
            sector_type_triples
            .unionAll(sector_label_triples)
            .unionAll(belongs_triples)
            .unionAll(rel_triples)
        )

        logger.info("  Sector pattern triples prepared (lazy)")
        return result

    # ================================================================
    # Step 3: Known correlations
    # ================================================================

    def _apply_known_correlations(self) -> Optional[DataFrame]:
        """
        Apply known correlations from domain expertise.

        Links category entities across datasets based on economic
        relationships using pattern matching on category URIs.

        Example:
            ppi:Food_Manufacturing_Entity → producerConsumerLink → cpi:Food_Entity
        """
        new_dfs: List[DataFrame] = []

        for correlation in KNOWN_CORRELATIONS:
            source_dataset = correlation['source_dataset']
            target_dataset = correlation['target_dataset']

            if source_dataset not in self.available_datasets:
                continue
            if target_dataset not in self.available_datasets:
                continue

            source_ns = DATASET_NS_MAP.get(source_dataset, '')
            target_ns = DATASET_NS_MAP.get(target_dataset, '')
            if not source_ns or not target_ns:
                continue

            source_pattern = normalize_keyword_for_uri_matching(correlation['source_pattern'])
            target_pattern = normalize_keyword_for_uri_matching(correlation['target_pattern'])
            relationship = str(correlation['relationship'])

            # Find source category entities matching pattern
            source_entities = (
                self._bls_triples
                .select("subject").distinct()
                .filter(
                    F.col("subject").startswith(source_ns) &
                    F.col("subject").endswith("_Entity") &
                    F.col("subject").contains(source_pattern)
                )
                .select(F.col("subject").alias("source_entity"))
            )

            # Find target category entities matching pattern
            target_entities = (
                self._bls_triples
                .select("subject").distinct()
                .filter(
                    F.col("subject").startswith(target_ns) &
                    F.col("subject").endswith("_Entity") &
                    F.col("subject").contains(target_pattern)
                )
                .select(F.col("subject").alias("target_entity"))
            )

            # Cross join source × target (both should be small for category entities)
            linked = source_entities.crossJoin(target_entities)

            if linked.head(1) != []:
                correlation_triples = linked.select(
                    F.col("source_entity").alias("subject"),
                    F.lit(relationship).alias("predicate"),
                    F.col("target_entity").alias("object"),
                )
                new_dfs.append(correlation_triples)

        if not new_dfs:
            return None

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Correlation triples prepared (lazy)")
        return result

    # ================================================================
    # Step 4: Category hierarchies
    # ================================================================

    def _link_category_hierarchies(self) -> Optional[DataFrame]:
        """
        Link category hierarchies based on URI path structure.

        BLS category URIs encode hierarchy in their path:
            cpi:All_items_Food_Food_at_home_Entity
                → hasParent → cpi:All_items_Food_Entity
            cpi:All_items_Food_Entity
                → hasParent → cpi:All_items_Entity

        Strategy:
        1. Collect all _Entity URIs
        2. For each, extract the path (remove namespace + _Entity suffix)
        3. Generate candidate parent paths by progressively removing trailing segments
        4. Join candidates against existing entities to find actual parents
        5. Keep only the immediate parent (longest matching parent path)
        """
        # Get all category entities across all BLS datasets
        all_categories = (
            self._bls_triples
            .select("subject")
            .distinct()
            .filter(F.col("subject").endswith("_Entity"))
        )

        if all_categories.head(1) == []:
            return None

        # Extract namespace and local path
        # URI format: {namespace}{path}_Entity
        # We need to split into namespace + path
        all_categories = all_categories.withColumn(
            "local_name",
            F.regexp_extract(F.col("subject"), r"[/]([^/]+)$", 1)
        ).withColumn(
            "namespace",
            F.regexp_extract(F.col("subject"), r"^(.+/)", 1)
        ).withColumn(
            "path",
            F.regexp_replace(F.col("local_name"), "_Entity$", "")
        ).filter(
            F.length(F.col("path")) > 0
        )

        # Count path segments to determine hierarchy depth
        all_categories = all_categories.withColumn(
            "segment_count",
            F.size(F.split(F.col("path"), "_"))
        )

        # Only entities with >1 segment can have parents
        children = all_categories.filter(F.col("segment_count") > 1)

        if children.head(1) == []:
            return None

        # Generate candidate parent paths by removing trailing segments
        # For path "A_B_C_D", candidates are: "A_B_C", "A_B", "A"
        # We want the longest match (immediate parent)
        #
        # Strategy: explode each child into multiple candidate rows,
        # one per possible parent depth, then join against existing entities.

        # First, build a lookup of all existing (namespace, path) pairs
        existing_paths = (
            all_categories
            .select(
                F.col("namespace").alias("parent_ns"),
                F.col("path").alias("parent_path"),
                F.col("subject").alias("parent_uri"),
            )
        )

        # For children, generate the immediate parent candidate (remove last segment)
        # This handles the most common case. For deeper gaps, we'd need iteration,
        # but the rdflib version also only linked to the immediate parent
        # (break after first match).
        children_with_parent_path = children.withColumn(
            "parent_path_candidate",
            F.regexp_replace(F.col("path"), r"_[^_]+$", "")
        ).filter(
            # Ensure we actually removed something
            F.col("parent_path_candidate") != F.col("path")
        )

        # Join against existing entities to find actual parents
        # Try immediate parent first (one segment removed)
        depth1_matched = (
            children_with_parent_path
            .join(
                existing_paths,
                on=(
                    (children_with_parent_path.namespace == existing_paths.parent_ns) &
                    (children_with_parent_path.parent_path_candidate == existing_paths.parent_path)
                ),
                how="inner",
            )
            .select(
                F.col("subject"),
                F.col("parent_uri"),
                F.lit(1).alias("depth"),
            )
        )

        # Entities that didn't match at depth 1
        unmatched_at_d1 = (
            children_with_parent_path
            .join(
                depth1_matched.select("subject").distinct(),
                on="subject",
                how="left_anti",
            )
        )

        # Try removing two segments for entities where immediate parent doesn't exist
        depth2_candidates = unmatched_at_d1.withColumn(
            "parent_path_d2",
            F.regexp_replace(F.col("parent_path_candidate"), r"_[^_]+$", "")
        ).filter(
            F.col("parent_path_d2") != F.col("parent_path_candidate")
        )

        depth2_matched = (
            depth2_candidates
            .join(
                existing_paths,
                on=(
                    (depth2_candidates.namespace == existing_paths.parent_ns) &
                    (depth2_candidates.parent_path_d2 == existing_paths.parent_path)
                ),
                how="inner",
            )
            .select(
                F.col("subject"),
                F.col("parent_uri"),
                F.lit(2).alias("depth"),
            )
        )

        # Union both depths, keeping the closest parent (depth-1 preferred)
        all_matched = depth1_matched.unionAll(depth2_matched)

        hierarchy_triples = all_matched.select(
            F.col("subject"),
            F.lit(_HAS_PARENT).alias("predicate"),
            F.col("parent_uri").alias("object"),
        )

        logger.info("  Hierarchy triples prepared (lazy)")
        return hierarchy_triples