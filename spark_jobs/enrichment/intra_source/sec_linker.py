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
3. Link temporal sequences (chronological ordering of filings by owner and by
   issuer, and of the transactions an ownership filing reports)
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
from typing import List, Optional, Sequence, Set, Union

from rdflib.namespace import RDF, RDFS, OWL

from spark_jobs.utils.rdf_utils import (
    SEC_ENRICHMENT, SEC_FILINGS, SEC_COMMON,
    UNIFIED,
    identifier_namespace,
)
from spark_jobs.enrichment.intra_source.sec.patterns import (
    SEC_VIOLATION_PATTERNS,
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

# SEC Base namespace for date intermediate nodes.
#
# This is the LIVE path, not a fallback -- see _resolve_dates. Every SEC date
# predicate that this linker sequences on points at an intermediate node, never
# at a literal: measured over 23,983 filing rows, hasFilingDate resolves through
# a node on 23,983 of them and hasPeriodOfReport on all 10,628 that state it,
# with zero literal objects between them. hasDateValue is then stated by 100% of
# rows. Delete this constant and both temporal sequence steps resolve no date
# and emit no precedes edge at all.
#
# Recorded because a census run reported it ABSENT at 0.00% and proposed
# removing it. That reading came from resolving the term under the sec-filings
# namespace when it is declared in sec-common; the drift checker, which resolves
# qnames through each document's own @prefix, has always reported it correctly
# as emitted. See the note in bin/census_sec_terms.py's report().
#
# Upstream guarantees the pairing rather than merely happening to emit it: the
# date individual's subject URI is built from the date column, and a null column
# makes the whole mapping emit nothing. So a Date or ReportingDate node cannot
# exist without its hasDateValue -- the 100% is structural, not a sample
# artifact. Those two are also the only date classes declared, and hasDateValue
# has sec-common:Date as its rdfs:domain with ReportingDate a subclass of it.
_SEC_BASE_HAS_DATE_VALUE = str(SEC_COMMON.hasDateValue)

# Enrichment predicates
# sec:belongsToSector is no longer emitted. Its only writer was the keyword
# sector classifier deleted below; filings now reach a sector through
# filings:hasSic, which CrossSourceLinker attaches to the COMPANY.
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

# The two pointers from an ownership filing to the transactions it reports.
# They are the only path from a transaction back to the document it was
# reported in, and through it to the reporting owner -- the transaction states
# neither.
_FILINGS_HAS_NON_DERIVATIVE_TRANSACTION = str(
    SEC_FILINGS.hasNonDerivativeTransaction
)
_FILINGS_HAS_DERIVATIVE_TRANSACTION = str(SEC_FILINGS.hasDerivativeTransaction)




# Form type URIs
#
# Upstream declares exactly four form classes -- Form4, Form8K, Form10K,
# Form10Q. Form3, Form5 and OwnershipDocument were referenced here and are not
# declared under any name, so every filter naming them matched nothing.
#
# Forms 3 and 5 ARE collected upstream (both are in its seed form list); they
# simply carry no class that distinguishes them, so they cannot be selected by
# rdf:type at all. That is an upstream gap, not something this linker can work
# around -- selecting them would need root_form, which the RDF does not carry.
_FORM4_TYPE = str(SEC_FILINGS.Form4)
_FORM10K_TYPE = str(SEC_FILINGS.Form10K)
_FORM10Q_TYPE = str(SEC_FILINGS.Form10Q)
_FORM8K_TYPE = str(SEC_FILINGS.Form8K)

# The two transaction classes an ownership filing reports. Unlike Form3/Form5
# above, these ARE declared and emitted: measured over 2026-08-06,
# NonDerivativeTransaction appears in 2,226 of 12,223 filing documents and
# DerivativeTransaction in 805 -- which is the large majority of the 2,679
# Form 4s, the only documents that can carry either.
#
# Those figures come from the 2026-08-03..08-07 objects, which upstream wrote
# as a backfill on 08-11. Every DAILY object written since is a tenth the size
# and carries almost none of this -- 08-19 enriched 132 filings of 2,894 -- and
# the reason is COVERAGE, not a mapping change. Upstream fetches the filing
# DOCUMENT for at most 150 accessions per run, selected as the head of EDGAR's
# accession-descending order with no cursor, so the same few hundred filings
# are re-read and the rest are never fetched. An ownership filing whose
# document was fetched has its full contents: 606 of 606 on 08-07, 33 of 33 on
# 08-10, 23 of 23 on 08-19. Reporting owner present is exactly equivalent to
# "the document was fetched".
#
# Two consequences here. The filing-level ownership sequence above is starved
# on recent daily data for the same reason -- it has almost nothing to link,
# rather than being broken. And an e2e fixture must be sampled from a day
# upstream backfilled, or it encodes 5% document coverage as though that were
# the shape of the data.
_NON_DERIVATIVE_TRANSACTION_TYPE = str(SEC_FILINGS.NonDerivativeTransaction)
_DERIVATIVE_TRANSACTION_TYPE = str(SEC_FILINGS.DerivativeTransaction)




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

        # Four steps, not six. The old numbering announced "[Step 4/6]
        # Applying sector patterns" and "[Step 6/6] Applying known
        # correlations" and then called nothing -- there are no such methods
        # here. Two headers over no work is the same reporting problem as a
        # step that returns nothing quietly (#350).
        #
        # Filings reach a sector through CrossSourceLinker._link_filings_by_sic
        # instead, on the SIC code upstream states on the filing.

        # Step 1: Unify company entities
        logger.info("\n[Step 1/4] Unifying company entities...")
        self._append(new_dfs, self._unify_company_entities(), "[Step 1/4]")

        # Step 2: Unify person entities
        logger.info("\n[Step 2/4] Unifying person entities...")
        self._append(new_dfs, self._unify_person_entities(), "[Step 2/4]")

        # Step 3: Link temporal sequences
        logger.info("\n[Step 3/4] Linking temporal sequences...")
        self._append(new_dfs, self._link_temporal_sequences(), "[Step 3/4]")

        # Step 4: Apply violation patterns
        logger.info("\n[Step 4/4] Applying violation patterns...")
        self._append(new_dfs, self._apply_violation_patterns(), "[Step 4/4]")

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
            _FORM4_TYPE, _FORM10K_TYPE, _FORM10Q_TYPE, _FORM8K_TYPE,
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
    def _append(lst: list, item: Optional[DataFrame], step: str):
        """Keep a step's triples, and say so when it made none.

        Same reason as CrossSourceLinker._append: a step that returns None
        logged only its opening line, so a missing link family read like a
        healthy graph (#350).
        """
        if item is not None:
            lst.append(item)
            return
        logger.warning(f"  {step} produced no triples")

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

        Both branches are live and neither is the fallback, but which one fires
        depends entirely on the predicate, so they are not interchangeable:

          hasFilingDate       node    23,983 of 23,983 rows,  0 literals
          hasPeriodOfReport   node    10,628 of 10,628 stating it, 0 literals
          hasTransactionDate  literal 11,651 occurrences, no node

        Both filing-level sequence steps below key on hasPeriodOfReport and so
        take the node branch; _link_transaction_sequences keys on
        hasTransactionDate and is the caller the literal branch carries. Neither
        branch is dead and neither can serve the other's predicate.

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

        self._append(
            new_dfs, self._link_ownership_filing_sequences(), "Ownership filings"
        )
        self._append(
            new_dfs, self._link_periodic_reporting_sequences(), "Periodic reports"
        )
        self._append(
            new_dfs, self._link_transaction_sequences(), "Transactions"
        )

        if not new_dfs:
            return None

        result = reduce(DataFrame.unionAll, new_dfs)
        logger.info("  Temporal sequence triples prepared (lazy)")
        return result

    def _build_precedes_links(
        self,
        entities_with_dates: DataFrame,
        entity_col: str,
        group_col: Union[str, Sequence[str]],
        order_cols: Sequence[str] = (),
    ) -> Optional[DataFrame]:
        """
        Generic helper: given a DataFrame with (entity, group_key, date_value),
        produce precedes triples linking consecutive entities within each group
        ordered by date.

        The window is ordered on date_value, then on any order_cols the caller
        supplies, then ALWAYS on the entity URI. That last key is what makes the
        chain reproducible: entities sharing a date are otherwise ordered by
        whatever row order the shuffle happened to produce, which differs
        between runs of the same input. Ties are not hypothetical -- see
        _link_transaction_sequences, where most groups contain one -- and the
        e2e suite pins reproducibility across twin runs.

        Args:
            entities_with_dates: DataFrame with columns (entity_col, date_value,
                and every column named by group_col and order_cols)
            entity_col: name of the entity URI column
            group_col: name of the grouping column, or several names to
                partition on jointly (e.g. owner AND transaction class)
            order_cols: tie-break columns applied after date_value and before
                the entity URI, most significant first

        Returns:
            DataFrame of (subject, predicate, object) precedes triples
        """
        if entities_with_dates is None:
            return None

        group_cols = [group_col] if isinstance(group_col, str) else list(group_col)

        # Window: partition by group, order by date and then to a total order
        w = Window.partitionBy(*group_cols).orderBy(
            "date_value", *order_cols, entity_col
        )

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
        """Link an owner's ownership filings into a dated sequence.

        This said "Form 3 → Form 4 → Form 5", which upstream cannot express:
        Form4 is the only ownership class it declares, so the 3 and 5 legs
        selected nothing and the sequence was always Form 4 to Form 4. That is
        still the useful edge -- a reporting owner's transaction history is
        their run of Form 4s -- but it is a sequence within one form, not
        across three.
        """
        if 'filings' not in self.available_datasets:
            return None

        ownership_types = [_FORM4_TYPE]

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

    def _link_transaction_sequences(self) -> Optional[DataFrame]:
        """Chain a reporting owner's ownership transactions into a sequence.

        The two steps above sequence whole FILINGS and nothing inside them. The
        transactions a Form 4 reports are the signal the filing exists to carry,
        and they had no precedes edge at all: a director who bought on the 3rd
        and sold on the 11th produced two nodes with no relative order, so a
        GNN aggregated them as an unordered bag instead of along a path.

        GROUPING KEY -- (reporting owner, transaction class).

        Owner rather than filing, because the owner's transaction history is
        what the ordering is ABOUT and it does not stop at a document boundary.
        Within one day's data the two keys give the identical partition
        (measured on 2026-08-19: all 21 transaction-bearing filings state
        exactly one reporting owner, and no owner filed twice that day), so
        choosing owner costs nothing there and chains an owner's filings
        together once a run spans several days -- which production runs do, as
        they read the whole feed prefix rather than one partition. That works
        because upstream mints ReportingOwner_{cik} as ONE global node per
        person across every filing they appear on, deliberately and not
        accession-scoped. Filing-grouping would throw that away.

        The owner IS reachable, which had to be confirmed rather than assumed:
        a transaction states neither its filing nor its owner, so the join runs
        the other way, filing -> hasNonDerivativeTransaction/
        hasDerivativeTransaction -> transaction and filing ->
        hasReportingOwner -> owner. Both hops are already in _sec_triples.

        Joint filings are real -- 15 of the 606 ownership filings on 2026-08-07
        name more than one reporting owner, one of them ten -- so a
        transaction there joins EACH of those owners' chains. That fan-out is
        faithful rather than lossy: upstream attaches transactions to the
        filing and never to an owner, because the form itself does not say
        which owner a row belongs to. A consumer counting edges per transaction
        has to expect it.

        Class is in the key for two reasons. The instruments differ -- an
        option grant and an open-market purchase are reported in different
        tables and are not consecutive events in one series, and merging them
        interleaves two streams into a single path (25 edges rather than 19 on
        2026-08-19), answering "what happened next" wrongly rather than not at
        all. And the tie-break below depends on it: upstream's index counter
        runs per CLASS, so _NonDerivativeTransaction_0 and
        _DerivativeTransaction_0 are different rows of different tables and the
        index only identifies a row once the class is fixed.

        TIE-BREAK -- (date, filing, document index, transaction URI).

        Required, not defensive: 7 of the 11 multi-transaction groups on
        2026-08-19 contain a repeated transaction date, so ordering on date
        alone leaves most chains to the row order a shuffle happens to produce,
        which is not stable between runs -- and the e2e suite pins
        reproducibility across twin runs. Filing comes first so one document's
        transactions stay contiguous; the trailing _N of the transaction URI is
        the row's position in the form's transaction table, which upstream
        assigns in document order and which re-maps byte-identically from the
        same filing (contiguous 0..n-1 across all 606 ownership filings on
        2026-08-07). It is cast to an int so _10 sorts after _2, and it agrees
        with date order wherever both are defined -- 0 inversions over a day.
        The URI closes the order so it is total even for a shape carrying no
        index.

        WHAT IS DELIBERATELY LEFT OUT

        Undated transactions. 28 of 1,097 on 2026-08-07 (2.55%) carry no
        hasTransactionDate, so the date join drops them and they are chained to
        nothing. That is the right outcome -- an undated event has no place in
        a dated sequence -- but it is a live case rather than a rounding error,
        and it is upstream-recoverable: a FOOTNOTED <transactionDate> defeats
        the parser's unwrapping and the date is emitted under hasMarketValue
        instead. The count will drop if that is fixed; nothing here changes.

        The filing's form type. Transactions are keyed on their own classes and
        on the pointers to them, never on Form4, because Forms 3 and 5 report
        transactions identically and upstream declares no class for either. A
        form-based filter would silently see Form 4 only.

        hasExpirationDate, the other date a DerivativeTransaction carries: it
        dates when the instrument expires, not when the reported event
        happened.
        """
        if 'filings' not in self.available_datasets:
            return None

        transaction_types = [
            _NON_DERIVATIVE_TRANSACTION_TYPE, _DERIVATIVE_TRANSACTION_TYPE,
        ]

        transactions = (
            self._sec_triples
            .filter(F.col("predicate") == _RDF_TYPE)
            .filter(F.col("object").isin(transaction_types))
            .select(
                F.col("subject").alias("transaction"),
                F.col("object").alias("transaction_class"),
            )
            .distinct()
        )

        if transactions.head(1) == []:
            return None

        # Which document reported each transaction. Both pointers in one frame:
        # the class is read off rdf:type above, so all this has to establish is
        # the filing, and through it the owner.
        reported_in = (
            self._sec_triples
            .filter(F.col("predicate").isin([
                _FILINGS_HAS_NON_DERIVATIVE_TRANSACTION,
                _FILINGS_HAS_DERIVATIVE_TRANSACTION,
            ]))
            .select(
                F.col("subject").alias("filing"),
                F.col("object").alias("transaction"),
            )
            .distinct()
        )

        filing_owners = (
            self._sec_triples
            .filter(F.col("predicate") == _FILINGS_HAS_REPORTING_OWNER)
            .select(
                F.col("subject").alias("filing"),
                F.col("object").alias("owner"),
            )
            .distinct()
        )

        with_owner = (
            transactions
            .join(reported_in, "transaction", "inner")
            .join(filing_owners, "filing", "inner")
        )

        # hasTransactionDate is the literal branch of _resolve_dates -- 11,651
        # occurrences, never an intermediate node -- and it is the only date
        # predicate here that dates the event.
        dates = (
            self._resolve_dates(
                with_owner.select(F.col("transaction").alias("entity")).distinct(),
                "entity",
                _FILINGS_HAS_TRANSACTION_DATE,
            )
            .withColumnRenamed("entity", "transaction")
            .distinct()
        )

        dated = with_owner.join(dates, "transaction", "inner").withColumn(
            "document_index",
            F.regexp_extract(F.col("transaction"), r"_(\d+)$", 1).cast("int"),
        )

        return self._build_precedes_links(
            dated,
            "transaction",
            ["owner", "transaction_class"],
            order_cols=("filing", "document_index"),
        )

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
