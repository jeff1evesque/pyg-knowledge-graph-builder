"""
SEC Intra-Source Enrichment — PySpark Implementation

Migrated from rdflib SPARQL-based enricher to fully distributed PySpark.
All operations run on executors — no rdflib, no driver data operations.

Handles the filings feed (Forms 3, 4, 5, 10-K, 10-Q, 8-K) -- the only SEC feed
collected. Administrative proceedings, litigation releases and trading
suspensions were removed: upstream publishes no such feeds, so every step keyed
on them matched nothing and returned nothing, which is indistinguishable from
working correctly.

Enrichment steps:
1. Unify company entities (by CIK across datasets)
2. Unify person entities (by CIK across datasets)
3. Link temporal sequences (chronological ordering within each dataset)
4. Apply sector patterns (keyword matching against URIs and labels)
5. Apply violation patterns (keyword matching + cross-dataset linking)

Cross-dataset correlations were removed with the datasets they correlated: all
26 in KNOWN_CORRELATIONS paired filings with administrative proceedings,
litigation or trading suspensions, and none paired filings with itself.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from functools import reduce
from typing import List, Optional, Set

from rdflib.namespace import RDF, RDFS, OWL

from spark_jobs.utils.rdf_utils import (
    SEC_ENRICHMENT, SEC_FILINGS, SEC_COMMON,
    UNIFIED,
    identifier_namespace,
)
from spark_jobs.enrichment.intra_source.sec.patterns import (
    SEC_SECTOR_PATTERNS, SEC_VIOLATION_PATTERNS,
)
from spark_jobs.utils.spark_rdf_utils import (
    deduplicate_against_existing,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

_RDF_TYPE = str(RDF.type)
_OWL_SAME_AS = str(OWL.sameAs)
_RDFS_LABEL = str(RDFS.label)

# Namespace prefix strings for dataset detection and filtering
_SEC_FILINGS_NS = str(SEC_FILINGS)
_UNIFIED_NS = str(UNIFIED)
_SEC_ENRICHMENT_NS = str(SEC_ENRICHMENT)

# SEC Base namespace for date intermediate nodes
_SEC_BASE_HAS_DATE_VALUE = str(SEC_COMMON.hasDateValue)

# Enrichment predicates
_BELONGS_TO_SECTOR = str(SEC_ENRICHMENT.belongsToSector)
_HAS_VIOLATION_TYPE = str(SEC_ENRICHMENT.hasViolationType)
_HAS_CIK = str(SEC_ENRICHMENT.hasCik)
_PRECEDES = str(SEC_ENRICHMENT.precedes)

# Dataset-specific predicates (filings)
_FILINGS_HAS_ISSUER_CIK = str(SEC_FILINGS.hasIssuerCik)
_FILINGS_HAS_REPORTING_OWNER_CIK = str(SEC_FILINGS.hasReportingOwnerCik)
_FILINGS_HAS_REPORTING_OWNER = str(SEC_FILINGS.hasReportingOwner)
# The filing's subject company. Upstream remodelled this from hasCompany --
# which no filing states any more -- to hasIssuer pointing at a filings:Issuer
# node (itself subClassOf sec-common:Company), so it is now an object property
# rather than a literal company name.
#
# The grouping below is unaffected by that shape change: it partitions on the
# object either way. What DOES change is that the groups now form at all --
# hasCompany matched nothing, so every periodic filing fell out of the join and
# no 10-K/10-Q/8-K sequence was ever built. Issuer URIs are keyed by CIK
# (Issuer_0000842657) and shared across that issuer's filings, so grouping on
# them is the same partition the literal name was meant to produce, minus the
# name-spelling collisions.
_FILINGS_HAS_ISSUER = str(SEC_FILINGS.hasIssuer)
_FILINGS_HAS_PERIOD_OF_REPORT = str(SEC_FILINGS.hasPeriodOfReport)
_FILINGS_HAS_TRANSACTION_DATE = str(SEC_FILINGS.hasTransactionDate)
_FILINGS_HAS_ISSUER_TRADING_SYMBOL = str(SEC_FILINGS.hasIssuerTradingSymbol)
_FILINGS_HAS_ISSUER_NAME = str(SEC_FILINGS.hasIssuerName)




# Form type URIs
_FORM3_TYPE = str(SEC_FILINGS.Form3)
_FORM4_TYPE = str(SEC_FILINGS.Form4)
_FORM5_TYPE = str(SEC_FILINGS.Form5)
_FORM10K_TYPE = str(SEC_FILINGS.Form10K)
_FORM10Q_TYPE = str(SEC_FILINGS.Form10Q)
_FORM8K_TYPE = str(SEC_FILINGS.Form8K)
_OWNERSHIP_DOC_TYPE = str(SEC_FILINGS.OwnershipDocument)




# Enrichment type URIs
_UNIFIED_COMPANY_TYPE = str(SEC_ENRICHMENT.UnifiedCompany)
_UNIFIED_PERSON_TYPE = str(SEC_ENRICHMENT.UnifiedPerson)
_ECONOMIC_SECTOR_TYPE = str(SEC_ENRICHMENT.EconomicSector)
_VIOLATION_TYPE_TYPE = str(SEC_ENRICHMENT.ViolationType)

# Dataset IDENTIFIER map. Every consumer matches an ENTITY uri -- the sector
# and correlation builders find entities by namespace + keyword -- so these are
# the id/ namespaces, not the term ones. See identifier_namespace() in
# rdf_utils for why the distinction is load-bearing rather than cosmetic.
#
# _detect_datasets does NOT use this: it keys on rdf:type objects, which are
# terms, and is correct as written.
DATASET_NS_MAP = {
    'filings': identifier_namespace(_SEC_FILINGS_NS),
}


def normalize_keyword_for_uri_matching(keyword: str) -> str:
    """Normalize a keyword for matching against SEC URIs."""
    normalized = keyword.replace(' ', '')
    normalized = normalized.replace('(', '').replace(')', '')
    normalized = normalized.replace('-', '')
    return normalized


class SECIntraSourceLinker:
    """
    PySpark-based SEC intra-source enrichment.

    All methods:
    1. Read from self.triples_df
    2. Return new triples DataFrames
    3. Never call .collect(), .toPandas(), or .toLocalIterator()
       (except for bounded dataset detection)
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run all SEC intra-source enrichment steps.

        Args:
            triples_df: DataFrame with (subject, predicate, object)

        Returns:
            DataFrame of NEW triples only (to be unioned with triples_df by caller)
        """
        self.triples_df = triples_df
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        # Detect which SEC datasets are present
        self.available_datasets = self._detect_datasets()
        if not self.available_datasets:
            logger.info("No SEC data detected, skipping enrichment")
            return empty

        logger.info("=" * 60)
        logger.info("Starting SEC Intra-Source Enrichment (PySpark)")
        logger.info(f"Detected SEC datasets: {', '.join(self.available_datasets)}")
        logger.info("=" * 60)

        # Cache the SEC-relevant subset for performance
        self._sec_triples = self._filter_sec_triples().cache()

        new_dfs: List[DataFrame] = []

        # Step 1: Unify company entities
        logger.info("\n[Step 1/6] Unifying company entities...")
        self._append(new_dfs, self._unify_company_entities())

        # Step 2: Unify person entities
        logger.info("\n[Step 2/6] Unifying person entities...")
        self._append(new_dfs, self._unify_person_entities())

        # Step 3: Link temporal sequences
        logger.info("\n[Step 3/6] Linking temporal sequences...")
        self._append(new_dfs, self._link_temporal_sequences())

        # Step 4: Apply sector patterns
        logger.info("\n[Step 4/6] Applying sector patterns...")
        self._append(new_dfs, self._apply_sector_patterns())

        # Step 5: Apply violation patterns
        logger.info("\n[Step 5/6] Applying violation patterns...")
        self._append(new_dfs, self._apply_violation_patterns())

        # Step 6: Apply known correlations
        logger.info("\n[Step 6/6] Applying known correlations...")

        # Unpersist cached subset
        self._sec_triples.unpersist()

        if not new_dfs:
            logger.info("SEC enrichment produced no new triples")
            return empty

        all_new = reduce(DataFrame.unionAll, new_dfs)
        all_new = all_new.dropDuplicates(["subject", "predicate", "object"])
        all_new = deduplicate_against_existing(all_new, triples_df)

        all_new = all_new.cache()
        total = all_new.count()

        logger.info("\n" + "=" * 60)
        logger.info(f"SEC Intra-Source Enrichment Complete: {total} new triples")
        logger.info("=" * 60)

        return all_new

    # ================================================================
    # Detection and filtering
    # ================================================================

    def _detect_datasets(self) -> Set[str]:
        """Detect which SEC datasets are present. Bounded collect."""
        datasets = set()

        # Check for filings types
        filings_types = [
            _FORM3_TYPE, _FORM4_TYPE, _FORM5_TYPE,
            _FORM10K_TYPE, _FORM10Q_TYPE, _FORM8K_TYPE,
            _OWNERSHIP_DOC_TYPE,
        ]

        type_triples = (
            self.triples_df
            .filter(F.col("predicate") == _RDF_TYPE)
            .select("object")
            .distinct()
        )

        # Collect just the distinct type URIs — bounded by ontology size
        existing_types = {row.object for row in type_triples.collect()}

        if any(t in existing_types for t in filings_types):
            datasets.add('filings')

        return datasets

    def _filter_sec_triples(self) -> DataFrame:
        """Filter triples to only SEC-relevant subjects.

        Matches BOTH sides of the term/individual split. This is the broad
        "keep what SEC enrichment might need" gate, and the two carry different
        things: entities and the date intermediate nodes they hang off live
        under id/, while rdf:type objects are terms under ontology/. Dropping
        either half silently starves a later step of its input.
        """
        sec_prefixes = [
            *DATASET_NS_MAP.values(),
            _SEC_FILINGS_NS,
            # Date intermediate nodes -- individuals under id/, and the
            # predicates that reach them, under ontology/.
            str(SEC_COMMON), identifier_namespace(str(SEC_COMMON)),
        ]

        def _any_startswith(col):
            cond = F.col(col).startswith(sec_prefixes[0])
            for p in sec_prefixes[1:]:
                cond = cond | F.col(col).startswith(p)
            return cond

        return self.triples_df.filter(
            _any_startswith("subject") | _any_startswith("object")
        )

    @staticmethod
    def _append(lst: list, item: Optional[DataFrame]):
        if item is not None:
            lst.append(item)

    # ================================================================
    # Helper: resolve date through intermediate node
    # ================================================================

    def _resolve_dates(
        self,
        entities_df: DataFrame,
        entity_col: str,
        date_predicate: str,
    ) -> DataFrame:
        """
        Resolve dates for entities, handling the SEC filings intermediate
        node pattern:
            entity → date_predicate → dateNode → hasDateValue → "2024-01-15"

        Also handles direct literal dates:
            entity → date_predicate → "2024-01-15"

        Returns DataFrame with columns: (entity_col, date_value)
        """
        # Get the object of the date predicate
        date_links = (
            self._sec_triples
            .filter(F.col("predicate") == date_predicate)
            .select(
                F.col("subject").alias(entity_col),
                F.col("object").alias("_date_obj"),
            )
        )

        # Join to entities
        date_links = entities_df.join(date_links, entity_col, "inner")

        # Case 1: object is a direct date literal (contains digits and dashes)
        direct_dates = (
            date_links
            .filter(
                F.col("_date_obj").rlike(r"^\d{4}-\d{2}")
            )
            .select(
                F.col(entity_col),
                F.col("_date_obj").alias("date_value"),
            )
        )

        # Case 2: object is an intermediate URI node → look up hasDateValue
        intermediate_nodes = (
            date_links
            .filter(
                ~F.col("_date_obj").rlike(r"^\d{4}-\d{2}")
            )
            .select(
                F.col(entity_col),
                F.col("_date_obj").alias("_date_node"),
            )
        )

        date_values = (
            self._sec_triples
            .filter(F.col("predicate") == _SEC_BASE_HAS_DATE_VALUE)
            .select(
                F.col("subject").alias("_date_node"),
                F.col("object").alias("date_value"),
            )
        )

        resolved_dates = (
            intermediate_nodes
            .join(date_values, "_date_node", "inner")
            .select(entity_col, "date_value")
        )

        # Union both paths
        all_dates = direct_dates.unionAll(resolved_dates)

        # Extract just the date portion (YYYY-MM-DD) for sorting
        all_dates = all_dates.withColumn(
            "date_value",
            F.regexp_extract(F.col("date_value"), r"(\d{4}-\d{2}-\d{2})", 1)
        ).filter(F.length(F.col("date_value")) > 0)

        return all_dates

    # ================================================================
    # Step 1: Unify company entities
    # ================================================================

    def _unify_company_entities(self) -> Optional[DataFrame]:
        """
        Unify company entities sharing a CIK.

        Creates unified:Company_{CIK} with owl:sameAs links to every entity
        stating that CIK, whenever more than one does.

        This said "across SEC datasets ... appear in multiple datasets", which
        was never what it keys on -- the grouping counts distinct ENTITIES per
        CIK, not distinct datasets. The distinction became load-bearing when
        filings became the only collected SEC feed: read as written, the method
        would now be dead, when in fact it still unifies the repeated filings a
        single issuer makes.
        """
        # Collect CIK → entity mappings from each dataset
        cik_sources: List[DataFrame] = []

        # Filings: issuer CIK
        if 'filings' in self.available_datasets:
            filings_cik = (
                self._sec_triples
                .filter(F.col("predicate") == _FILINGS_HAS_ISSUER_CIK)
                .select(
                    F.col("subject").alias("entity"),
                    F.trim(F.col("object")).alias("cik"),
                )
                .filter(F.length(F.col("cik")) > 0)
            )
            cik_sources.append(filings_cik)



        if not cik_sources:
            return None

        # Union all CIK → entity mappings
        all_cik_entities = reduce(DataFrame.unionAll, cik_sources).dropDuplicates()

        # Find CIKs that appear with more than one distinct entity
        cik_counts = (
            all_cik_entities
            .groupBy("cik")
            .agg(F.countDistinct("entity").alias("entity_count"))
            .filter(F.col("entity_count") > 1)
        )

        # Join back to get the entities for CIKs stated by more than one entity
        multi_dataset = all_cik_entities.join(cik_counts.select("cik"), "cik", "inner")

        # Build unified URI
        unified_uri = F.concat(F.lit(_UNIFIED_NS), F.lit("Company_"), F.col("cik"))

        # Type triples: unified:Company_{CIK} rdf:type sec_enrichment:UnifiedCompany
        type_triples = (
            multi_dataset.select("cik").distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(_UNIFIED_COMPANY_TYPE).alias("object"),
            )
        )

        # CIK triples: unified:Company_{CIK} sec_enrichment:hasCik "{CIK}"
        cik_triples = (
            multi_dataset.select("cik").distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_HAS_CIK).alias("predicate"),
                F.col("cik").alias("object"),
            )
        )

        # sameAs triples: unified:Company_{CIK} owl:sameAs <entity_uri>
        sameas_triples = multi_dataset.select(
            F.concat(F.lit(_UNIFIED_NS), F.lit("Company_"), F.col("cik")).alias("subject"),
            F.lit(_OWL_SAME_AS).alias("predicate"),
            F.col("entity").alias("object"),
        )

        result = type_triples.unionAll(cik_triples).unionAll(sameas_triples)
        logger.info("  Company unification triples prepared (lazy)")
        return result

    # ================================================================
    # Step 2: Unify person entities
    # ================================================================

    def _unify_person_entities(self) -> Optional[DataFrame]:
        """
        Unify person entities across SEC datasets using CIK numbers.

        Creates unified:Person_{CIK} entities with owl:sameAs links.
        """
        cik_sources: List[DataFrame] = []

        # Filings: reporting owner CIK
        if 'filings' in self.available_datasets:
            owner_cik = (
                self._sec_triples
                .filter(F.col("predicate") == _FILINGS_HAS_REPORTING_OWNER_CIK)
                .select(
                    F.col("subject").alias("entity"),
                    F.trim(F.col("object")).alias("cik"),
                )
                .filter(F.length(F.col("cik")) > 0)
            )
            cik_sources.append(owner_cik)


        if not cik_sources:
            return None

        all_cik_entities = reduce(DataFrame.unionAll, cik_sources).dropDuplicates()

        cik_counts = (
            all_cik_entities
            .groupBy("cik")
            .agg(F.countDistinct("entity").alias("entity_count"))
            .filter(F.col("entity_count") > 1)
        )

        multi_dataset = all_cik_entities.join(cik_counts.select("cik"), "cik", "inner")

        unified_uri = F.concat(F.lit(_UNIFIED_NS), F.lit("Person_"), F.col("cik"))

        type_triples = (
            multi_dataset.select("cik").distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(_UNIFIED_PERSON_TYPE).alias("object"),
            )
        )

        cik_triples = (
            multi_dataset.select("cik").distinct()
            .select(
                unified_uri.alias("subject"),
                F.lit(_HAS_CIK).alias("predicate"),
                F.col("cik").alias("object"),
            )
        )

        sameas_triples = multi_dataset.select(
            F.concat(F.lit(_UNIFIED_NS), F.lit("Person_"), F.col("cik")).alias("subject"),
            F.lit(_OWL_SAME_AS).alias("predicate"),
            F.col("entity").alias("object"),
        )

        result = type_triples.unionAll(cik_triples).unionAll(sameas_triples)
        logger.info("  Person unification triples prepared (lazy)")
        return result

    # ================================================================
    # Step 3: Temporal sequences
    # ================================================================

    def _link_temporal_sequences(self) -> Optional[DataFrame]:
        """Link temporal sequences across all SEC datasets."""
        new_dfs: List[DataFrame] = []

        self._append(new_dfs, self._link_ownership_filing_sequences())
        self._append(new_dfs, self._link_periodic_reporting_sequences())

        if not new_dfs:
            return None

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Temporal sequence triples prepared (lazy)")
        return result

    def _build_precedes_links(
        self,
        entities_with_dates: DataFrame,
        entity_col: str,
        group_col: str,
    ) -> Optional[DataFrame]:
        """
        Generic helper: given a DataFrame with (entity, group_key, date_value),
        produce precedes triples linking consecutive entities within each group
        ordered by date.

        Args:
            entities_with_dates: DataFrame with columns (entity_col, group_col, date_value)
            entity_col: name of the entity URI column
            group_col: name of the grouping column (e.g., owner, company, respondent)

        Returns:
            DataFrame of (subject, predicate, object) precedes triples
        """
        if entities_with_dates is None:
            return None

        # Window: partition by group, order by date
        w = Window.partitionBy(group_col).orderBy("date_value")

        with_next = (
            entities_with_dates
            .withColumn("next_entity", F.lead(entity_col).over(w))
            .filter(F.col("next_entity").isNotNull())
        )

        precedes_triples = with_next.select(
            F.col(entity_col).alias("subject"),
            F.lit(_PRECEDES).alias("predicate"),
            F.col("next_entity").alias("object"),
        )

        return precedes_triples

    def _link_ownership_filing_sequences(self) -> Optional[DataFrame]:
        """Link Form 3 → Form 4 → Form 5 sequences for same reporting owner."""
        if 'filings' not in self.available_datasets:
            return None

        ownership_types = [_FORM3_TYPE, _FORM4_TYPE, _FORM5_TYPE]

        # Get all ownership filings
        ownership_filings = (
            self._sec_triples
            .filter(F.col("predicate") == _RDF_TYPE)
            .filter(F.col("object").isin(ownership_types))
            .select(F.col("subject").alias("filing"))
            .distinct()
        )

        if ownership_filings.head(1) == []:
            return None

        # Get reporting owner for each filing
        filing_owners = (
            self._sec_triples
            .filter(F.col("predicate") == _FILINGS_HAS_REPORTING_OWNER)
            .select(
                F.col("subject").alias("filing"),
                F.col("object").alias("owner"),
            )
        )

        filings_with_owner = ownership_filings.join(filing_owners, "filing", "inner")

        # Resolve dates
        dates = self._resolve_dates(
            filings_with_owner.select(F.col("filing").alias("entity")).distinct(),
            "entity",
            _FILINGS_HAS_PERIOD_OF_REPORT,
        ).withColumnRenamed("entity", "filing")

        filings_dated = filings_with_owner.join(dates, "filing", "inner")

        return self._build_precedes_links(filings_dated, "filing", "owner")

    def _link_periodic_reporting_sequences(self) -> Optional[DataFrame]:
        """Link 10-K, 10-Q, 8-K sequences for same company."""
        if 'filings' not in self.available_datasets:
            return None

        periodic_types = [_FORM10K_TYPE, _FORM10Q_TYPE, _FORM8K_TYPE]

        periodic_filings = (
            self._sec_triples
            .filter(F.col("predicate") == _RDF_TYPE)
            .filter(F.col("object").isin(periodic_types))
            .select(F.col("subject").alias("filing"))
            .distinct()
        )

        if periodic_filings.head(1) == []:
            return None

        filing_companies = (
            self._sec_triples
            .filter(F.col("predicate") == _FILINGS_HAS_ISSUER)
            .select(
                F.col("subject").alias("filing"),
                F.col("object").alias("company"),
            )
        )

        filings_with_company = periodic_filings.join(filing_companies, "filing", "inner")

        dates = self._resolve_dates(
            filings_with_company.select(F.col("filing").alias("entity")).distinct(),
            "entity",
            _FILINGS_HAS_PERIOD_OF_REPORT,
        ).withColumnRenamed("entity", "filing")

        filings_dated = filings_with_company.join(dates, "filing", "inner")

        return self._build_precedes_links(filings_dated, "filing", "company")

    def _apply_sector_patterns(self) -> Optional[DataFrame]:
        """
        Apply sector-based linking patterns.

        Links entities to industry sectors based on keyword matching
        against entity URIs and rdfs:labels.
        """
        # Build keyword lookup: (keyword_normalized, keyword_lower, sector_uri, relationship, namespace)
        sector_rows = []
        for sector_name, pattern in SEC_SECTOR_PATTERNS.items():
            sector_uri = str(pattern['sector_uri'])
            relationship = str(pattern['relationship'])
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                ns = DATASET_NS_MAP.get(dataset_name, '')
                if not ns or not keywords:
                    continue
                for keyword in keywords:
                    normalized = normalize_keyword_for_uri_matching(keyword)
                    sector_rows.append((normalized, keyword.lower(), sector_uri, relationship, ns))

        if not sector_rows:
            return None

        sector_df = self.spark.createDataFrame(
            sector_rows,
            schema=["keyword_normalized", "keyword_lower", "sector_uri", "relationship", "namespace"],
        )

        # Get all SEC entities with their namespace
        sec_entities = (
            self._sec_triples
            .select("subject")
            .distinct()
            .withColumn("subject_str", F.col("subject"))
        )

        # Match by URI: entity URI contains normalized keyword and starts with namespace
        uri_matched = (
            sec_entities
            .join(F.broadcast(sector_df), how="inner",
                  on=(
                      F.col("subject").startswith(F.col("namespace")) &
                      F.col("subject").contains(F.col("keyword_normalized"))
                  ))
        )

        # Match by label: entity has rdfs:label containing keyword
        labels = (
            self._sec_triples
            .filter(F.col("predicate") == _RDFS_LABEL)
            .select(
                F.col("subject"),
                F.lower(F.col("object")).alias("label_lower"),
            )
        )

        label_matched = (
            labels
            .join(F.broadcast(sector_df), how="inner",
                  on=(
                      F.col("subject").startswith(F.col("namespace")) &
                      F.col("label_lower").contains(F.col("keyword_lower"))
                  ))
        )

        # Union URI and label matches
        all_matched = (
            uri_matched.select("subject", "sector_uri", "relationship")
            .unionAll(label_matched.select("subject", "sector_uri", "relationship"))
            .dropDuplicates(["subject", "sector_uri"])
        )

        if all_matched.head(1) == []:
            return None

        # Sector type triples
        sector_type_triples = (
            all_matched.select("sector_uri").distinct()
            .select(
                F.col("sector_uri").alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(_ECONOMIC_SECTOR_TYPE).alias("object"),
            )
        )

        # Sector label triples
        sector_label_triples = (
            all_matched.select("sector_uri").distinct()
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
        belongs_triples = all_matched.select(
            F.col("subject"),
            F.lit(_BELONGS_TO_SECTOR).alias("predicate"),
            F.col("sector_uri").alias("object"),
        )

        # Relationship triples
        rel_triples = all_matched.select(
            F.col("subject"),
            F.col("relationship").alias("predicate"),
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
    # Step 5: Violation patterns
    # ================================================================

    def _apply_violation_patterns(self) -> Optional[DataFrame]:
        """
        Apply violation type patterns.

        Links entities to violation types based on keyword matching,
        and cross-links entities of the same violation type across datasets.
        """
        # Build keyword lookup
        violation_rows = []
        for violation_name, pattern in SEC_VIOLATION_PATTERNS.items():
            violation_uri = str(pattern['violation_uri'])
            relationship = str(pattern['relationship'])
            for dataset_name in self.available_datasets:
                keywords = pattern['keywords'].get(dataset_name, [])
                ns = DATASET_NS_MAP.get(dataset_name, '')
                if not ns or not keywords:
                    continue
                for keyword in keywords:
                    normalized = normalize_keyword_for_uri_matching(keyword)
                    violation_rows.append((normalized, keyword.lower(), violation_uri, relationship, ns))

        if not violation_rows:
            return None

        violation_df = self.spark.createDataFrame(
            violation_rows,
            schema=["keyword_normalized", "keyword_lower", "violation_uri", "relationship", "namespace"],
        )

        # Match by URI
        sec_entities = self._sec_triples.select("subject").distinct()

        uri_matched = (
            sec_entities
            .join(F.broadcast(violation_df), how="inner",
                  on=(
                      F.col("subject").startswith(F.col("namespace")) &
                      F.col("subject").contains(F.col("keyword_normalized"))
                  ))
        )

        # Match by label
        labels = (
            self._sec_triples
            .filter(F.col("predicate") == _RDFS_LABEL)
            .select(F.col("subject"), F.lower(F.col("object")).alias("label_lower"))
        )

        label_matched = (
            labels
            .join(F.broadcast(violation_df), how="inner",
                  on=(
                      F.col("subject").startswith(F.col("namespace")) &
                      F.col("label_lower").contains(F.col("keyword_lower"))
                  ))
        )

        all_matched = (
            uri_matched.select("subject", "violation_uri", "relationship", "namespace")
            .unionAll(label_matched.select("subject", "violation_uri", "relationship", "namespace"))
            .dropDuplicates(["subject", "violation_uri"])
        )

        if all_matched.head(1) == []:
            return None

        new_dfs: List[DataFrame] = []

        # Violation type entity triples
        viol_type_triples = (
            all_matched.select("violation_uri").distinct()
            .select(
                F.col("violation_uri").alias("subject"),
                F.lit(_RDF_TYPE).alias("predicate"),
                F.lit(_VIOLATION_TYPE_TYPE).alias("object"),
            )
        )
        new_dfs.append(viol_type_triples)

        # Violation label triples
        viol_label_triples = (
            all_matched.select("violation_uri").distinct()
            .withColumn(
                "label",
                F.regexp_replace(
                    F.regexp_extract(F.col("violation_uri"), r"[/#]([^/#]+)$", 1),
                    "([A-Z])", r" $1"
                )
            )
            .withColumn("label", F.trim(F.col("label")))
            .select(
                F.col("violation_uri").alias("subject"),
                F.lit(_RDFS_LABEL).alias("predicate"),
                F.col("label").alias("object"),
            )
        )
        new_dfs.append(viol_label_triples)

        # hasViolationType triples
        has_viol_triples = all_matched.select(
            F.col("subject"),
            F.lit(_HAS_VIOLATION_TYPE).alias("predicate"),
            F.col("violation_uri").alias("object"),
        )
        new_dfs.append(has_viol_triples)

        # Cross-link entities of same violation type across different datasets
        # Self-join on violation_uri where namespaces differ
        left = all_matched.select(
            F.col("subject").alias("entity_a"),
            F.col("violation_uri"),
            F.col("relationship"),
            F.col("namespace").alias("ns_a"),
        )
        right = all_matched.select(
            F.col("subject").alias("entity_b"),
            F.col("violation_uri").alias("violation_uri_r"),
            F.col("namespace").alias("ns_b"),
        )

        cross_links = (
            left.join(
                right,
                (left.violation_uri == right.violation_uri_r) &
                (left.ns_a != right.ns_b) &
                (left.entity_a < right.entity_b),  # avoid duplicates
                "inner",
            )
            .select(
                F.col("entity_a").alias("subject"),
                F.col("relationship").alias("predicate"),
                F.col("entity_b").alias("object"),
            )
        )

        new_dfs.append(cross_links)

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Violation pattern triples prepared (lazy)")
        return result

    # ================================================================
    # Step 6: Known correlations
    # ================================================================
