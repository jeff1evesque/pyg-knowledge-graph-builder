"""
Temporal Entity Unifier (PySpark)

Unifies temporal entities (months, years, quarters) across all data sources.
Creates unified temporal entities and links source-specific temporal entities
to them using owl:sameAs.

Also types the SOURCE temporal URIs it links to (temporal:Source{Month,Year,
Quarter}). Those arrive untyped from the sources, and node_mapper only makes
nodes out of typed URIs — so without this neither the measurement->period edges
nor the sameAs links resolve. See _create_source_temporal_types.

All operations run as distributed PySpark DataFrame transformations.

Example output triples:
    cpi:November      rdf:type      temporal:SourceMonth
    cpi:November      rdfs:label    "November"

    unified:November  rdf:type      bls:UnifiedMonth
    unified:November  rdfs:label    "November"
    unified:November  owl:sameAs    cpi:November
    unified:November  owl:sameAs    ppi:November
    unified:November  owl:sameAs    https://jefflevesque.com/id/temporal/market-feeds/November

    unified:Year2024  rdf:type      bls:UnifiedYear
    unified:Year2024  rdfs:label    "2024"
    unified:Year2024  owl:sameAs    cpi:2024
    unified:Year2024  owl:sameAs    https://jefflevesque.com/id/temporal/sec/2024

    unified:Q1        rdf:type      bls:UnifiedQuarter
    unified:Q1        rdfs:label    "Q1"
    unified:Q1        owl:sameAs    wkyeng:Q1
    unified:Q1        bls:coversMonth  unified:January
    unified:Q1        bls:coversMonth  unified:February
    unified:Q1        bls:coversMonth  unified:March
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from functools import reduce
from typing import List, Optional

from spark_jobs.utils.rdf_utils import (
    BLS_ENRICHMENT, UNIFIED, SOURCE_TEMPORAL,
    CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER, WKYENG,
    SEC_FILINGS, SEC_ADMIN, SEC_LIT, SEC_SUSP,
    MARKET_FEEDS, CAP, WEATHER,
    SYNTHETIC_TEMPORAL_IDS,
    identifier_namespace,
)

import logging

logger = logging.getLogger(__name__)

# ============================================
# URI string constants
# ============================================

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"

UNIFIED_MONTH_TYPE = str(BLS_ENRICHMENT.UnifiedMonth)
UNIFIED_YEAR_TYPE = str(BLS_ENRICHMENT.UnifiedYear)
UNIFIED_QUARTER_TYPE = str(BLS_ENRICHMENT.UnifiedQuarter)
COVERS_MONTH_PRED = str(BLS_ENRICHMENT.coversMonth)

# The on-ramp from a dated entity to the period it falls in.
#
# The date-literal sources (SEC, NOAA, market-feeds) state a date and nothing
# else -- unlike the BLS datasets, whose source data already says
# `obs hasMonth cpi:February` and so arrives on the spine under its own power.
# For those three the unifier minted temporal/{source}/February and linked it
# sameAs unified:February, and stopped: nothing pointed AT the period, because
# the collector read the date value and discarded the subject it belonged to.
# The result was a two-node island per period per source -- a bridge with no
# on-ramp -- while the graph still reported unified temporal entities and
# looked connected.
#
# In BLS_ENRICHMENT rather than SOURCE_TEMPORAL because classify_edge_origin
# reads pipeline provenance off the predicate's namespace, and SOURCE_TEMPORAL
# is not one of ENRICHMENT_NAMESPACES -- minting it there would report these
# inferred links as raw source facts. coversMonth above is in the same
# namespace for the same reason and is likewise emitted for every source.
OBSERVED_IN_PERIOD_PRED = str(BLS_ENRICHMENT.observedInPeriod)

# Types for the source-side temporal URIs the unifier links TO — see the
# SOURCE_TEMPORAL note in rdf_utils for why these are their own namespace
# rather than reusing the Unified* types.
SOURCE_MONTH_TYPE = str(SOURCE_TEMPORAL.SourceMonth)
SOURCE_YEAR_TYPE = str(SOURCE_TEMPORAL.SourceYear)
SOURCE_QUARTER_TYPE = str(SOURCE_TEMPORAL.SourceQuarter)

SOURCE_TEMPORAL_TYPE_BY_KIND = {
    "month": SOURCE_MONTH_TYPE,
    "year": SOURCE_YEAR_TYPE,
    "quarter": SOURCE_QUARTER_TYPE,
}

UNIFIED_BASE = str(UNIFIED)

# BLS monthly dataset IDENTIFIER prefixes.
#
# These match period URIs -- id/cpi/February, id/jolts/2024 -- which are
# individuals, not terms. Keying them on the term namespaces (ontology/cpi/)
# matched nothing at all: no BLS period was ever collected, so none was typed
# temporal:SourceMonth, so node_mapper's _CANONICAL_TYPE_PRIORITY had nothing
# to prefer and every period sharded across cpi_Month / jolts_Month /
# empsit_Month / eci_Month. The graph kept its per-source month nodes and lost
# the cross-source bridge, without anything raising.
BLS_MONTHLY_PREFIXES = [
    identifier_namespace(str(ns))
    for ns in (CPI, PPI, ECI, JOLTS, EMPSIT, XIMPIM, LAUS, METRO, REALER)
]

# BLS quarterly dataset namespace prefix
WKYENG_PREFIX = str(WKYENG)
WKYENG_HAS_QUARTER = str(WKYENG.hasQuarter)
WKYENG_HAS_YEAR = str(WKYENG.hasYear)

# Market predicates
MARKET_OBSERVED_AT = str(MARKET_FEEDS.observedAt)
MARKET_PRICE_OBS_TYPE = str(MARKET_FEEDS.PriceObservation)
MARKET_OPTION_CONTRACT_TYPE = str(MARKET_FEEDS.OptionContract)
MARKET_EXPIRATION_DATE = str(MARKET_FEEDS.expirationDate)

# SEC types for detection
SEC_TYPES = [
    str(SEC_FILINGS.Form3), str(SEC_FILINGS.Form4),
    str(SEC_ADMIN.AdministrativeProceeding),
    str(SEC_LIT.LitigationRelease),
    str(SEC_SUSP.TradingSuspension),
]

# SEC date predicates
#
# These carried hasattr()/else-literal guards, which were dead code: rdflib's
# Namespace.__getattr__ resolves any non-dunder name, so hasattr was always
# True and no fallback could run. Worse than redundant -- every fallback still
# spelled a pre-migration sec.gov URI, so had one ever fired it would have
# reintroduced the exact namespace this work removed. NOAA_DATE_PREDS below
# was always written this way.
#
# hasReportDate was dead -- upstream states hasFilingDate, and does so on every
# filing (40 of 40 on the e2e fixtures, against 39 for hasPeriodOfReport), so
# the single most common SEC date was the one missing from this list.
# hasPeriodOfReport was live, which is why SEC still minted period nodes and the
# gap read as "SEC is on the spine" rather than as a defect.
SEC_DATE_PREDS = [
    str(SEC_FILINGS.hasPeriodOfReport),
    str(SEC_FILINGS.hasFilingDate),
    str(SEC_ADMIN.initiationDate),
    str(SEC_LIT.filingDate),
    str(SEC_SUSP.startDate),
]

# NOAA — aligned with updated RML mapper
# In the new mapper, cap:hasSentTime is on the Info subject
# (alert:{alert_id}#info), not on the Alert subject.
# We collect all NOAA date predicates that appear on Info subjects.
# The temporal unifier's _collect_date_based_temporals() extracts
# month/year from the literal values regardless of which subject
# they appear on — it only needs the predicate and object columns.
NOAA_WEATHER_ALERT_TYPE = str(WEATHER.WeatherAlert)
CAP_ALERT_TYPE = str(CAP.Alert)

# All NOAA date predicates that carry temporal information.
# hasSentTime, hasEffectiveTime, hasOnsetTime, hasExpirationTime, hasEndsTime
# are all on the Info subject in the new mapper.
NOAA_DATE_PREDS = [
    str(CAP.hasSentTime),
    str(CAP.hasEffectiveTime),
    str(CAP.hasOnsetTime),
    str(CAP.hasExpirationTime),
    str(CAP.hasEndsTime),
]

# Valid month names for regex matching
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Month number → name mapping for date parsing
MONTH_NUM_TO_NAME = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}

# Quarter → months mapping
QUARTER_MONTH_MAP = {
    "Q1": ["January", "February", "March"],
    "Q2": ["April", "May", "June"],
    "Q3": ["July", "August", "September"],
    "Q4": ["October", "November", "December"],
}


class TemporalUnifier:
    """
    Unifies temporal entities across all data sources using PySpark.

    Strategies:
    1. Collect temporal URIs from BLS (explicit month/year/quarter entities)
    2. Extract month/year from SEC date literals → synthetic temporal URIs
    3. Extract month/year from Market timestamps → synthetic temporal URIs
    4. Extract month/year from NOAA timestamps → synthetic temporal URIs
    5. Group all temporal URIs by normalized name
    6. Produce unified entities with owl:sameAs links
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def enrich(self, triples_df: DataFrame) -> DataFrame:
        """
        Run temporal unification across all sources.

        Args:
            triples_df: DataFrame with columns (subject, predicate, object)

        Returns:
            DataFrame of NEW triples (unified temporal entities + sameAs links)
        """
        empty = self.spark.createDataFrame(
            [], "subject STRING, predicate STRING, object STRING"
        )

        logger.info("=" * 60)
        logger.info("Starting Temporal Unification (PySpark)")
        logger.info("=" * 60)

        triples_df.cache()

        # Collect (temporal_uri, normalized_name, kind) from all sources
        # kind is "month", "year", or "quarter"
        temporal_dfs: List[DataFrame] = []

        logger.info("  Collecting BLS temporal entities...")
        df = self._collect_bls_months_years(triples_df)
        if df is not None:
            temporal_dfs.append(df)

        df = self._collect_bls_quarters(triples_df)
        if df is not None:
            temporal_dfs.append(df)

        # The date-literal sources, kept as (subject, temporal_uri, ...) frames.
        # The subject is dropped for the union below -- which only wants the
        # distinct periods -- but survives here so enrich() can also emit the
        # link from each dated entity to its period.
        dated_dfs: List[DataFrame] = []

        logger.info("  Collecting SEC temporal entities...")
        df = self._collect_date_based_temporals(
            triples_df, SEC_DATE_PREDS, SYNTHETIC_TEMPORAL_IDS["sec"]
        )
        if df is not None:
            dated_dfs.append(df)

        logger.info("  Collecting Market temporal entities...")
        df = self._collect_market_temporals(triples_df)
        if df is not None:
            dated_dfs.append(df)

        logger.info("  Collecting NOAA temporal entities...")
        df = self._collect_date_based_temporals(
            triples_df, NOAA_DATE_PREDS, SYNTHETIC_TEMPORAL_IDS["noaa"]
        )
        if df is not None:
            dated_dfs.append(df)

        # Settle the date-literal frames once. Each is a scan of triples_df and
        # each feeds BOTH the period union below and the period links further
        # down, so without this their subtrees are planned twice over -- the
        # same Catalyst blow-up the all_temporals checkpoint exists to avoid.
        dated: Optional[DataFrame] = None
        if dated_dfs:
            dated = reduce(DataFrame.unionAll, dated_dfs).dropDuplicates(
                ["subject", "temporal_uri", "normalized_name", "kind"]
            ).localCheckpoint(eager=True)
            temporal_dfs.append(
                dated.select("temporal_uri", "normalized_name", "kind")
            )

        if not temporal_dfs:
            logger.info("No temporal entities found in any source")
            return empty

        # Union all temporal entities: (temporal_uri, normalized_name, kind)
        #
        # Truncate the plan here (see EnrichmentPipeline._settle). all_temporals is a
        # union of every collector above -- each one a scan of triples_df -- and the
        # three _create_unified_* branches below each build on it, so without this its
        # whole subtree is re-planned into all three and Catalyst's constraint
        # inference blows the driver heap while *planning* (OutOfMemoryError inside
        # .cache(), before a single task runs). It survives a 1-source run and OOMs a
        # 7-source one. Eager, so it is materialized once and shared by the branches.
        all_temporals = reduce(DataFrame.unionAll, temporal_dfs).dropDuplicates(
            ["temporal_uri", "normalized_name", "kind"]
        ).localCheckpoint(eager=True)

        # Produce unified triples
        new_dfs: List[DataFrame] = []

        new_dfs.append(self._create_source_temporal_types(all_temporals))

        if dated is not None:
            # The on-ramp: each entity that stated a date -> the period it names.
            # Both granularities, matching how the BLS sources arrive (an
            # observation states hasMonth AND hasYear).
            new_dfs.append(
                dated.select(
                    F.col("subject"),
                    F.lit(OBSERVED_IN_PERIOD_PRED).alias("predicate"),
                    F.col("temporal_uri").alias("object"),
                ).dropDuplicates()
            )

        df = self._create_unified_months(all_temporals)
        if df is not None:
            new_dfs.append(df)

        df = self._create_unified_years(all_temporals)
        if df is not None:
            new_dfs.append(df)

        df = self._create_unified_quarters(all_temporals)
        if df is not None:
            new_dfs.append(df)

        if not new_dfs:
            logger.info("No unified temporal triples produced")
            return empty

        result = reduce(DataFrame.unionAll, new_dfs).cache()
        count = result.count()

        logger.info("=" * 60)
        logger.info(f"Temporal Unification Complete: {count} triples")
        logger.info("=" * 60)

        return result

    # ================================================================
    # BLS: Explicit month/year URIs
    # ================================================================

    def _collect_bls_months_years(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Collect month and year URIs from BLS monthly datasets.

        BLS datasets use URI-based temporal entities like:
          id/cpi/November, id/ppi/November, id/cpi/2024, id/ppi/2024

        We find these by looking for URIs under BLS IDENTIFIER prefixes whose
        local name matches a month name or 4-digit year. Identifier, not term:
        periods are things, and nothing is ever minted under ontology/cpi/.
        """
        # Build filter: object URI starts with any BLS monthly prefix
        # and is used as an object in any triple (i.e., referenced as a value)
        bls_prefix_filter = F.lit(False)
        for prefix in BLS_MONTHLY_PREFIXES:
            bls_prefix_filter = bls_prefix_filter | F.col("object").startswith(prefix)

        bls_objects = (
            triples_df
            .filter(bls_prefix_filter)
            .select(F.col("object").alias("temporal_uri"))
            .dropDuplicates()
        )

        if bls_objects.head(1) == []:
            return None

        # Extract local name (everything after the last "/")
        bls_objects = bls_objects.withColumn(
            "local_name",
            F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
        )

        # Month URIs: local name is a valid month name
        month_names_str = "|".join(MONTH_NAMES)
        months = bls_objects.filter(
            F.col("local_name").rlike(f"^({month_names_str})$")
        ).select(
            F.col("temporal_uri"),
            F.col("local_name").alias("normalized_name"),
            F.lit("month").alias("kind"),
        )

        # Year URIs: local name is a 4-digit number
        years = bls_objects.filter(
            F.col("local_name").rlike(r"^\d{4}$")
        ).select(
            F.col("temporal_uri"),
            F.col("local_name").alias("normalized_name"),
            F.lit("year").alias("kind"),
        )

        result = months.unionAll(years)
        if result.head(1) == []:
            return None

        return result

    def _collect_bls_quarters(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Collect quarter and year URIs from WKYENG (quarterly BLS dataset).

        WKYENG uses:
          ?entity wkyeng:hasQuarter wkyeng:Q1
          ?entity wkyeng:hasYear wkyeng:2024
        """
        # Quarters: objects of wkyeng:hasQuarter
        quarters = triples_df.filter(
            F.col("predicate") == WKYENG_HAS_QUARTER
        ).select(
            F.col("object").alias("temporal_uri")
        ).dropDuplicates().withColumn(
            "local_name",
            F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
        ).filter(
            F.col("local_name").isin(["Q1", "Q2", "Q3", "Q4"])
        ).select(
            F.col("temporal_uri"),
            F.col("local_name").alias("normalized_name"),
            F.lit("quarter").alias("kind"),
        )

        # Years: objects of wkyeng:hasYear
        years = triples_df.filter(
            F.col("predicate") == WKYENG_HAS_YEAR
        ).select(
            F.col("object").alias("temporal_uri")
        ).dropDuplicates().withColumn(
            "local_name",
            F.regexp_extract(F.col("temporal_uri"), r"([^/]+)$", 1)
        ).filter(
            F.col("local_name").rlike(r"^\d{4}$")
        ).select(
            F.col("temporal_uri"),
            F.col("local_name").alias("normalized_name"),
            F.lit("year").alias("kind"),
        )

        result = quarters.unionAll(years)
        if result.head(1) == []:
            return None

        return result

    # ================================================================
    # SEC / NOAA: Date literals → synthetic temporal URIs
    # ================================================================

    def _collect_date_based_temporals(
        self,
        triples_df: DataFrame,
        date_predicates: List[str],
        synthetic_prefix: str,
    ) -> Optional[DataFrame]:
        """
        Extract month/year from date literal values and create synthetic
        temporal URIs.

        Works for any source that stores dates as literal values
        (SEC filing dates, NOAA alert timestamps, etc.)

        For NOAA: In the updated RML mapper, cap:hasSentTime and other
        date properties are on the Info subject (alert:{id}#info), not
        the Alert subject. This method keys only on the predicate, so it works
        regardless of which subject the date property appears on -- the entity
        it links to the period is whichever subject stated the date.

        The SUBJECT is carried through, not just the date value. It is what
        OBSERVED_IN_PERIOD_PRED links to the period node in enrich(); without it
        the synthetic period is an island. See that constant for the history.

        Some sources state the date as a URI rather than a literal -- SEC points
        hasFilingDate at sec-common:Date_2026-08-14 -- and the regexes below
        read the date out of either form, so both shapes land on the spine. The
        entity linked to the period is the one that STATED the date (the
        filing), not the date node, so the period sits one hop from the filing
        rather than two.

        Args:
            triples_df: The triples DataFrame
            date_predicates: List of predicate URIs that hold date values
            synthetic_prefix: URI prefix for synthetic temporal entities
                              e.g., "https://jefflevesque.com/id/temporal/sec/"

        Produces (subject, temporal_uri, normalized_name, kind) rows like:
            (filing, ".../temporal/sec/November", "November", "month")
            (filing, ".../temporal/sec/2024", "2024", "year")
        """
        # Filter to triples with the relevant date predicates
        date_triples = triples_df.filter(
            F.col("predicate").isin(date_predicates)
        ).select(
            F.col("subject"),
            F.col("object").alias("date_value"),
        )

        if date_triples.head(1) == []:
            return None

        date_triples = date_triples.dropDuplicates()

        # Extract month number and year from date strings.
        # Handles ISO formats: "2024-11-15", "2024-11-15T10:30:00", etc.
        # Uses substring extraction — no UDF needed.
        parsed = date_triples.withColumn(
            "year_str", F.regexp_extract(F.col("date_value"), r"(\d{4})", 1)
        ).withColumn(
            "month_str", F.regexp_extract(F.col("date_value"), r"\d{4}-(\d{2})", 1)
        ).filter(
            (F.col("year_str") != "") & (F.col("month_str") != "")
        )

        if parsed.head(1) == []:
            return None

        # Map month number to month name using a broadcast join
        month_mapping = self.spark.createDataFrame(
            list(MONTH_NUM_TO_NAME.items()),
            ["month_str", "month_name"]
        )

        parsed = parsed.join(F.broadcast(month_mapping), "month_str", "inner")

        # Produce month rows
        months = parsed.select(
            F.col("subject"),
            F.concat(F.lit(synthetic_prefix), F.col("month_name")).alias("temporal_uri"),
            F.col("month_name").alias("normalized_name"),
            F.lit("month").alias("kind"),
        ).dropDuplicates()

        # Produce year rows
        years = parsed.select(
            F.col("subject"),
            F.concat(F.lit(synthetic_prefix), F.col("year_str")).alias("temporal_uri"),
            F.col("year_str").alias("normalized_name"),
            F.lit("year").alias("kind"),
        ).dropDuplicates()

        return months.unionAll(years)

    # ================================================================
    # Market: Timestamps + expiration dates → synthetic temporal URIs
    # ================================================================

    def _collect_market_temporals(
        self, triples_df: DataFrame
    ) -> Optional[DataFrame]:
        """
        Extract month/year from Market price observation timestamps
        and option expiration dates.

        Price observations use market-feeds:observedAt with dateTime values.
        Option contracts use market-feeds:expirationDate with date values.
        Both are ISO format, so we reuse the date parsing logic.

        This is the FEEDS vocabulary, not the quote-snapshot one -- quotes
        have no observedAt and no PriceObservation. The quote snapshots carry
        captureTime instead and are handled by the intra-source market linker.

        Returns the same (subject, temporal_uri, normalized_name, kind) shape
        as the collector it delegates to, so feeds observations get the same
        on-ramp to the spine that SEC and NOAA entities now do.
        """
        market_date_preds = [MARKET_OBSERVED_AT, MARKET_EXPIRATION_DATE]
        return self._collect_date_based_temporals(
            triples_df, market_date_preds, SYNTHETIC_TEMPORAL_IDS["market-feeds"]
        )

    # ================================================================
    # Produce unified entities
    # ================================================================

    def _create_source_temporal_types(
        self, all_temporals: DataFrame
    ) -> DataFrame:
        """
        Type the SOURCE temporal URIs so they can become graph nodes.

        Source data references periods as bare URIs — cpi:February, eci:2024,
        jolts:August — and none of them carries an rdf:type. node_mapper only
        creates nodes for typed URIs, so they were never nodes, and every triple
        pointing at them (hasMonth / hasYear / hasStartMonth / hasEndMonth /
        hasStartYear / hasEndYear — ~1,205 on the e2e fixtures) was silently
        dropped during edge resolution. The graph had no temporal dimension: no
        measurement was connected to when it happened.

        The unifier's own owl:sameAs links pointed at the same untyped URIs, so
        they were dropped too, leaving the Unified{Month,Year} nodes as isolated
        vertices a GNN can propagate nothing through.

        This method emits the missing rdf:type (plus a label) for exactly the
        set of temporal URIs the unifier already collects — the same set it
        links to — which completes the intended bridge:

            cpi obs -> cpi:February -> unified:February <- eci:February <- eci obs

        Both hops are edges only once BOTH endpoints are typed.

        Produces, per source temporal URI:
            cpi:February  rdf:type    temporal:SourceMonth
            cpi:February  rdfs:label  "February"

        Note the synthetic URIs minted by the date-literal collectors
        (sec/noaa/market temporal/{Month}) are typed here too — they are equally
        untyped otherwise, and their only edges are the sameAs links.
        """
        distinct = all_temporals.select(
            "temporal_uri", "normalized_name", "kind"
        ).dropDuplicates()

        type_expr = None
        for kind, type_uri in SOURCE_TEMPORAL_TYPE_BY_KIND.items():
            cond = F.col("kind") == kind
            type_expr = (
                F.when(cond, F.lit(type_uri)) if type_expr is None
                else type_expr.when(cond, F.lit(type_uri))
            )

        typed = distinct.withColumn("type_uri", type_expr.otherwise(F.lit(None)))
        # An unrecognized kind would otherwise emit `<uri> rdf:type NULL`, which
        # node_mapper would turn into a garbage node type rather than failing.
        typed = typed.filter(F.col("type_uri").isNotNull())

        type_triples = typed.select(
            F.col("temporal_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.col("type_uri").alias("object"),
        )

        label_triples = typed.select(
            F.col("temporal_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        return type_triples.unionAll(label_triples)

    def _create_unified_months(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified month entities and owl:sameAs links.

        For each unique month name, produces:
          unified:{MonthName}  rdf:type    bls:UnifiedMonth
          unified:{MonthName}  rdfs:label  "{MonthName}"
          unified:{MonthName}  owl:sameAs  <source_temporal_uri_1>
          unified:{MonthName}  owl:sameAs  <source_temporal_uri_2>
          ...
        """
        months = all_temporals.filter(F.col("kind") == "month")

        if months.head(1) == []:
            return None

        # Build unified URI
        months = months.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.col("normalized_name"))
        )

        # Type triples (one per unique month name)
        distinct_months = months.select("unified_uri", "normalized_name").dropDuplicates()

        type_triples = distinct_months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_MONTH_TYPE).alias("object"),
        )

        label_triples = distinct_months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        # sameAs triples (one per source temporal URI)
        same_as_triples = months.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        return type_triples.unionAll(label_triples).unionAll(same_as_triples)

    def _create_unified_years(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified year entities and owl:sameAs links.

        For each unique year value, produces:
          unified:Year{YYYY}  rdf:type    bls:UnifiedYear
          unified:Year{YYYY}  rdfs:label  "{YYYY}"
          unified:Year{YYYY}  owl:sameAs  <source_temporal_uri_1>
          ...
        """
        years = all_temporals.filter(F.col("kind") == "year")

        if years.head(1) == []:
            return None

        years = years.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.lit("Year"), F.col("normalized_name"))
        )

        distinct_years = years.select("unified_uri", "normalized_name").dropDuplicates()

        type_triples = distinct_years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_YEAR_TYPE).alias("object"),
        )

        label_triples = distinct_years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        same_as_triples = years.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        return type_triples.unionAll(label_triples).unionAll(same_as_triples)

    def _create_unified_quarters(
        self, all_temporals: DataFrame
    ) -> Optional[DataFrame]:
        """
        Create unified quarter entities, owl:sameAs links, and
        coversMonth links to unified months.

        For each unique quarter label, produces:
          unified:{Q1}  rdf:type           bls:UnifiedQuarter
          unified:{Q1}  rdfs:label         "Q1"
          unified:{Q1}  owl:sameAs         <source_quarter_uri>
          unified:{Q1}  bls:coversMonth    unified:January
          unified:{Q1}  bls:coversMonth    unified:February
          unified:{Q1}  bls:coversMonth    unified:March
        """
        quarters = all_temporals.filter(F.col("kind") == "quarter")

        if quarters.head(1) == []:
            return None

        quarters = quarters.withColumn(
            "unified_uri",
            F.concat(F.lit(UNIFIED_BASE), F.col("normalized_name"))
        )

        distinct_quarters = quarters.select(
            "unified_uri", "normalized_name"
        ).dropDuplicates()

        type_triples = distinct_quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDF_TYPE).alias("predicate"),
            F.lit(UNIFIED_QUARTER_TYPE).alias("object"),
        )

        label_triples = distinct_quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(RDFS_LABEL).alias("predicate"),
            F.col("normalized_name").alias("object"),
        )

        same_as_triples = quarters.select(
            F.col("unified_uri").alias("subject"),
            F.lit(OWL_SAME_AS).alias("predicate"),
            F.col("temporal_uri").alias("object"),
        )

        # coversMonth links: quarter → constituent months
        # Build a small DataFrame of (quarter_label, month_name) pairs
        covers_rows = []
        for q_label, month_names in QUARTER_MONTH_MAP.items():
            for m_name in month_names:
                covers_rows.append((q_label, m_name))

        covers_df = self.spark.createDataFrame(
            covers_rows, ["quarter_label", "month_name"]
        )

        # Join to get (unified_quarter_uri, unified_month_uri)
        covers_joined = distinct_quarters.join(
            F.broadcast(covers_df),
            distinct_quarters.normalized_name == covers_df.quarter_label,
            "inner",
        )

        covers_triples = covers_joined.select(
            F.col("unified_uri").alias("subject"),
            F.lit(COVERS_MONTH_PRED).alias("predicate"),
            F.concat(F.lit(UNIFIED_BASE), F.col("month_name")).alias("object"),
        )

        return (
            type_triples
            .unionAll(label_triples)
            .unionAll(same_as_triples)
            .unionAll(covers_triples)
        )


def unify_temporal_entities(
    spark: SparkSession, triples_df: DataFrame
) -> DataFrame:
    """
    Convenience function for temporal unification.

    Args:
        spark: SparkSession
        triples_df: DataFrame with (subject, predicate, object)

    Returns:
        DataFrame of new temporal unification triples
    """
    unifier = TemporalUnifier(spark)
    return unifier.enrich(triples_df)
